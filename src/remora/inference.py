import os
import torch
from copy import copy
import array as pyarray
from pathlib import Path
from itertools import chain
from threading import Thread
from collections import defaultdict

import pod5
import pysam
import numpy as np
from tqdm import tqdm

from remora import constants, log, RemoraError
from remora.data_chunks import CoreRemoraDataset, DatasetMetadata
from remora.io import (
    ReadIndexedBam,
    iter_signal,
    extract_alignments,
    DuplexRead,
    DuplexPairsBuilder,
)
from remora.util import (
    _queue_iter,
    _put_item,
    MultitaskMap,
    BackgroundIter,
    NamedQueue,
    NamedMPQueue,
    format_mm_ml_tags,
    softmax_axis1,
    get_read_ids,
    Motif,
    revcomp,
    human_format,
)

LOGGER = log.get_logger()
_PROF_PREP_FN = os.getenv("REMORA_INFER_PREP_DATA_PROFILE_FILE")
_PROF_BATCH_FN = os.getenv("REMORA_INFER_BATCH_PROFILE_FILE")
_PROF_MODEL_FN = os.getenv("REMORA_INFER_RUN_MODEL_PROFILE_FILE")
_PROF_UNBATCH_FN = os.getenv("REMORA_INFER_UNBATCH_PROFILE_FILE")
_PROF_MAIN_FN = os.getenv("REMORA_INFER_MAIN_PROFILE_FILE")


################
# POD5+BAM CLI #
################


def mods_tags_to_str(mods_tags):
    # TODO these operations are often quite slow
    return [
        f"MM:Z:{mods_tags[0]}",
        f"ML:B:C,{','.join(map(str, mods_tags[1]))}",
    ]


def prepare_reads(read_errs, model_metadata, ref_anchored):
    out_read_errs = []
    motif_seqs, motif_offsets = zip(*model_metadata["motifs"])
    motifs = [Motif(*mot) for mot in model_metadata["motifs"]]
    md_kwargs = {
        "mod_bases": model_metadata["mod_bases"],
        "mod_long_names": model_metadata["mod_long_names"],
        "motif_sequences": motif_seqs,
        "motif_offsets": motif_offsets,
        "chunk_context": model_metadata["chunk_context"],
        "kmer_context_bases": model_metadata["kmer_context_bases"],
        "extra_arrays": {"read_focus_bases": ("int64", "")},
    }
    for io_read, err in read_errs:
        if io_read is None:
            out_read_errs.append((None, None, err))
            continue
        try:
            remora_read = io_read.into_remora_read(ref_anchored)
        except RemoraError as e:
            LOGGER.debug(f"Remora read prep error: {e}")
            out_read_errs.append((None, None, "Remora read prep error"))
            continue
        remora_read.set_motif_focus_bases(motifs)
        remora_read.refine_signal_mapping(model_metadata["sig_map_refiner"])
        chunks = list(
            remora_read.iter_chunks(
                model_metadata["chunk_context"],
                model_metadata["kmer_context_bases"],
                model_metadata["base_start_justify"],
                model_metadata["offset"],
            )
        )
        if len(chunks) == 0:
            out_read_errs.append((None, None, "No mod calls"))
            continue
        # clear larger memory arrays (for quicker queue transfer)
        io_read.sig_len = io_read.dacs.size
        io_read.dacs = None
        io_read.mv_table = None
        io_read.query_to_signal = None
        io_read.ref_to_signal = None
        # prepare in memory dataset to perform chunk extraction
        num_chunks = len(chunks)
        md_kwargs["allocate_size"] = num_chunks
        md_kwargs["max_seq_len"] = max(c.seq_len for c in chunks)
        # open in-memory dataset
        dataset = CoreRemoraDataset(
            mode="w",
            metadata=DatasetMetadata(**md_kwargs),
            batch_size=num_chunks,
            super_batch_size=num_chunks,
            infinite_iter=False,
        )
        for chunk in chunks:
            dataset.write_chunk(chunk)
        out_read_errs.append((io_read, dataset, None))
    return out_read_errs


if _PROF_PREP_FN:
    _prepare_reads_wrapper = prepare_reads

    def prepare_reads(*args, **kwargs):
        import cProfile

        prof = cProfile.Profile()
        retval = prof.runcall(_prepare_reads_wrapper, *args, **kwargs)
        prof.dump_stats(_PROF_PREP_FN)
        return retval


def prep_nn_input(read_errs):
    # TODO for basecall-anchored calls only call on first read and apply to
    # other mappings
    if len(read_errs) == 0:
        return [
            (None, None, "No valid mappings"),
        ]
    read_nn_inputs = []
    for io_read, read_dataset, err in read_errs:
        if err is not None:
            read_nn_inputs.append((io_read, None, err))
            continue
        r_chunks = next(iter(read_dataset))
        del r_chunks["labels"]
        read_nn_inputs.append((io_read, r_chunks, None))
    return read_nn_inputs


def batch_reads(prepped_nn_inputs, batches_q, batch_size, model_metadata):
    def new_arrays():
        return (
            np.empty(
                (batch_size, 1, model_metadata["chunk_len"]),
                dtype=np.float32,
            ),
            np.empty(
                (
                    batch_size,
                    model_metadata["kmer_len"] * 4,
                    model_metadata["chunk_len"],
                ),
                dtype=np.float32,
            ),
            np.empty(batch_size, dtype=int),
        )

    b_sigs, b_enc_kmers, b_read_pos = new_arrays()
    # position within current batch
    b_pos = 0
    b_reads = []
    for read_nn_inputs in prepped_nn_inputs:
        for io_read, r_chunks, err in read_nn_inputs:
            if err is not None:
                b_reads.append([None, None, None, err])
                continue
            num_chunks = r_chunks["read_focus_bases"].size
            # fill out and yield full batches
            rb_consumed = 0
            # while this read extends through a whole batch continue to
            # supply batches from this read
            while b_pos + num_chunks - rb_consumed > batch_size:
                rb_en = rb_consumed + batch_size - b_pos
                b_sigs[b_pos:] = r_chunks["signal"][rb_consumed:rb_en]
                b_enc_kmers[b_pos:] = r_chunks["enc_kmers"][rb_consumed:rb_en]
                b_read_pos[b_pos:] = r_chunks["read_focus_bases"][
                    rb_consumed:rb_en
                ]
                # batch start is None once the first batch is complete
                # from this read
                b_st = b_pos if rb_consumed == 0 else None
                b_reads.append([io_read, b_st, None, None])
                _put_item((b_sigs, b_enc_kmers, b_read_pos, b_reads), batches_q)
                rb_consumed += batch_size - b_pos
                # new batch
                b_sigs, b_enc_kmers, b_read_pos = new_arrays()
                b_pos = 0
                b_reads = []
            # add rest of read to unfinished batch
            b_en = b_pos + num_chunks - rb_consumed
            b_sigs[b_pos:b_en] = r_chunks["signal"][rb_consumed:]
            b_enc_kmers[b_pos:b_en] = r_chunks["enc_kmers"][rb_consumed:]
            b_read_pos[b_pos:b_en] = r_chunks["read_focus_bases"][rb_consumed:]
            # if read continues from last batch set start to None
            b_st = b_pos if rb_consumed == 0 else None
            b_reads.append([io_read, b_st, b_en, None])
            # set current batch position for next read
            b_pos = b_en
    if b_pos > 0:
        # send last batch
        _put_item(
            (b_sigs[:b_pos], b_enc_kmers[:b_pos], b_read_pos[:b_pos], b_reads),
            batches_q,
        )
    _put_item(StopIteration, batches_q)


if _PROF_BATCH_FN:
    _batch_reads_wrapper = batch_reads

    def batch_reads(*args, **kwargs):
        import cProfile

        prof = cProfile.Profile()
        retval = prof.runcall(_batch_reads_wrapper, *args, **kwargs)
        prof.dump_stats(_PROF_BATCH_FN)
        return retval


def run_model_batched(
    batches_q, called_batches_q, model, model_metadata, batch_size
):
    device = next(model.parameters()).device
    pin_memory = device.type == "cuda"
    t_b_sigs = torch.empty(
        (batch_size, 1, model_metadata["chunk_len"]),
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    t_b_enc_kmers = torch.empty(
        (
            batch_size,
            model_metadata["kmer_len"] * 4,
            model_metadata["chunk_len"],
        ),
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    for b_sigs, b_enc_kmers, b_read_pos, b_reads in _queue_iter(batches_q):
        if b_read_pos.size == batch_size:
            t_b_sigs[:] = torch.from_numpy(b_sigs)
            t_b_enc_kmers[:] = torch.from_numpy(b_enc_kmers)
        else:
            t_b_sigs = torch.from_numpy(b_sigs)
            t_b_enc_kmers = torch.from_numpy(b_enc_kmers)
        nn_out = model(t_b_sigs.to(device), t_b_enc_kmers.to(device))
        _put_item((nn_out, b_read_pos, b_reads), called_batches_q)
    _put_item(StopIteration, called_batches_q)


if _PROF_MODEL_FN:
    _run_model_wrapper = run_model_batched

    def run_model_batched(*args, **kwargs):
        import cProfile

        prof = cProfile.Profile()
        retval = prof.runcall(_run_model_wrapper, *args, **kwargs)
        prof.dump_stats(_PROF_MODEL_FN)
        return retval


def unbatch_reads(curr_read, b_nn_out, b_read_pos, b_reads):
    comp_reads = []
    for io_read, b_st, b_en, err in b_reads:
        if err is not None:
            comp_reads.append((io_read, None, None, err))
            curr_read = None
            continue
        # end of read from previous batch
        if b_st is None:
            assert (
                curr_read is not None
                and curr_read[0].read_id == io_read.read_id
            )
            io_read, r_nn_out, r_read_pos, _ = curr_read
            # update curr_read
            curr_read = (
                io_read,
                np.concatenate([r_nn_out, b_nn_out[:b_en]], axis=0),
                np.concatenate([r_read_pos, b_read_pos[:b_en]]),
                None,
            )
            continue
        if curr_read is not None:
            comp_reads.append(curr_read)
        curr_read = (io_read, b_nn_out[b_st:b_en], b_read_pos[b_st:b_en], None)
    return comp_reads, curr_read


def unbatch(called_batches_q, called_reads_q):
    curr_read = None
    for nn_out, b_read_pos, b_reads in _queue_iter(called_batches_q):
        comp_reads, curr_read = unbatch_reads(
            curr_read, nn_out.cpu().numpy(), b_read_pos, b_reads
        )
        for read in comp_reads:
            _put_item(read, called_reads_q)
    if curr_read is not None:
        _put_item(curr_read, called_reads_q)
    _put_item(StopIteration, called_reads_q)


if _PROF_UNBATCH_FN:
    _unbatch_wrapper = unbatch

    def unbatch(*args, **kwargs):
        import cProfile

        prof = cProfile.Profile()
        retval = prof.runcall(_unbatch_wrapper, *args, **kwargs)
        prof.dump_stats(_PROF_UNBATCH_FN)
        return retval


def post_process_reads(read_mapping, model_metadata, ref_anchored):
    io_read, nn_out, r_poss, err = read_mapping
    if err is not None:
        return None, err
    r_probs = softmax_axis1(nn_out)[:, 1:].astype(np.float64)
    seq = io_read.ref_seq if ref_anchored else io_read.seq
    mod_tags = mods_tags_to_str(
        format_mm_ml_tags(
            seq=seq,
            poss=r_poss,
            probs=r_probs,
            mod_bases=model_metadata["mod_bases"],
            can_base=model_metadata["can_base"],
        )
    )
    io_read.full_align["tags"] = [
        tag
        for tag in io_read.full_align["tags"]
        if not (tag.startswith("MM") or tag.startswith("ML"))
    ]
    io_read.full_align["tags"].extend(mod_tags)
    if ref_anchored:
        io_read.full_align["cigar"] = f"{len(io_read.ref_seq)}M"
        io_read.full_align["seq"] = (
            io_read.ref_seq
            if io_read.ref_reg.strand == "+"
            else revcomp(io_read.ref_seq)
        )
        io_read.full_align["qual"] = "*"
    return io_read, None


def infer_from_pod5_and_bam(
    pod5_path,
    in_bam_path,
    model,
    model_metadata,
    out_bam_path,
    num_reads=None,
    queue_max=1_000,
    num_extract_alignment_workers=1,
    num_prep_read_workers=1,
    num_prep_nn_input_workers=1,
    num_post_process_workers=1,
    batch_size=constants.DEFAULT_BATCH_SIZE,
    skip_non_primary=True,
    ref_anchored=False,
):
    bam_idx = ReadIndexedBam(in_bam_path, skip_non_primary, req_tags={"mv"})
    with pod5.Reader(Path(pod5_path)) as pod5_fh:
        read_ids, num_reads = get_read_ids(bam_idx, pod5_fh, num_reads)

    signals = BackgroundIter(
        iter_signal,
        args=(pod5_path,),
        kwargs={
            "num_reads": num_reads,
            "read_ids": read_ids,
            "rev_sig": model_metadata["reverse_signal"],
        },
        name="ExtractSignal",
        use_process=True,
        q_maxsize=queue_max,
    )
    reads = MultitaskMap(
        extract_alignments,
        signals,
        num_workers=num_extract_alignment_workers,
        args=(bam_idx, model_metadata["reverse_signal"]),
        name="AddAlignments",
        use_process=True,
        use_mp_queue=True,
        q_maxsize=queue_max,
    )
    prepped_reads = MultitaskMap(
        prepare_reads,
        reads,
        num_workers=num_prep_read_workers,
        args=(model_metadata, ref_anchored),
        name="PrepReadData",
        use_process=True,
        use_mp_queue=True,
        q_maxsize=100,
    )
    prepped_nn_input = MultitaskMap(
        prep_nn_input,
        prepped_reads,
        num_workers=num_prep_nn_input_workers,
        name="PrepNNInput",
        use_process=False,
        use_mp_queue=False,
        q_maxsize=10,
    )
    batches_q = NamedQueue(maxsize=4, name="Batches")
    batch_reads_p = Thread(
        target=batch_reads,
        args=(
            _queue_iter(prepped_nn_input.out_q, num_prep_nn_input_workers),
            batches_q,
            batch_size,
            model_metadata,
        ),
        name="batch_reads",
        daemon=True,
    )
    batch_reads_p.start()
    called_batches_q = NamedQueue(maxsize=4, name="CalledBatches")
    call_batches_t = Thread(
        target=run_model_batched,
        args=(batches_q, called_batches_q, model, model_metadata, batch_size),
        name="call_batches",
        daemon=True,
    )
    call_batches_t.start()
    called_reads_q = NamedMPQueue(maxsize=queue_max, name="Unbatch")
    unbatch_p = Thread(
        target=unbatch,
        args=(
            called_batches_q,
            called_reads_q,
        ),
        name="unbatch",
        daemon=True,
    )
    unbatch_p.start()
    final_reads = MultitaskMap(
        post_process_reads,
        _queue_iter(called_reads_q),
        num_workers=num_post_process_workers,
        args=(model_metadata, ref_anchored),
        name="PostProcess",
        use_process=False,
        use_mp_queue=False,
        q_maxsize=queue_max,
    )

    all_qs = [
        signals.out_q,
        reads.out_q,
        prepped_reads.out_q,
        prepped_nn_input.out_q,
        batches_q,
        called_batches_q,
        called_reads_q,
        final_reads.out_q,
    ]
    errs = defaultdict(int)
    pysam_save = pysam.set_verbosity(0)
    sig_called = 0
    in_bam = out_bam = pbar = prev_rid = None
    try:
        in_bam = pysam.AlignmentFile(in_bam_path, "rb")
        out_bam = pysam.AlignmentFile(out_bam_path, "wb", template=in_bam)
        pbar = tqdm(
            smoothing=0,
            total=num_reads,
            dynamic_ncols=True,
            unit=" Reads",
            desc="Inferring mods",
            disable=os.environ.get("LOG_SAFE", False),
        )
        for io_read, err in final_reads:
            LOGGER.debug(
                "QueuesStatus: "
                + "\t".join(
                    [f"{q.name}: {q.queue.qsize()}/{q.maxsize}" for q in all_qs]
                )
            )
            if err is not None:
                errs[err] += 1
                prev_rid = None
                pbar.update()
                continue
            if prev_rid != io_read.read_id:
                pbar.update()
            sig_called += io_read.sig_len
            sps, mag = human_format(sig_called / pbar.format_dict["elapsed"])
            pbar.set_postfix_str(f"{sps:>5.1f} {mag}samps/s", refresh=False)
            out_bam.write(
                pysam.AlignedSegment.from_dict(
                    io_read.full_align, out_bam.header
                )
            )
            prev_rid = io_read.read_id
    finally:
        if pbar is not None:
            pbar.close()
        if in_bam is not None:
            in_bam.close()
        if out_bam is not None:
            out_bam.close()
    pysam.set_verbosity(pysam_save)

    if len(errs) > 0:
        err_types = sorted([(num, err) for err, num in errs.items()])[::-1]
        err_str = "\n".join(f"{num:>7} : {err:<80}" for num, err in err_types)
        LOGGER.info(f"Unsuccessful read reasons:\n{err_str}")
    batch_reads_p.join()
    call_batches_t.join()


if _PROF_MAIN_FN:
    _infer_from_pod5_and_bam_wrapper = infer_from_pod5_and_bam

    def infer_from_pod5_and_bam(*args, **kwargs):
        import cProfile

        prof = cProfile.Profile()
        retval = prof.runcall(_infer_from_pod5_and_bam_wrapper, *args, **kwargs)
        prof.dump_stats(_PROF_MAIN_FN)
        return retval


##########
# Duplex #
##########


def call_read_mods(
    read,
    model,
    model_metadata,
    batch_size=constants.DEFAULT_BATCH_SIZE,
    focus_offset=None,
    return_mm_ml_tags=False,
    return_mod_probs=False,
):
    """Call modified bases on a read.

    Args:
        read (RemoraRead): Read to be called
        model: Compiled inference model
            (see remora.model_util.load_torchscript_model)
        model_metadata: Inference model metadata
        batch_size (int): Number of chunks to call per-batch
        focus_offset (int): Specific base to call within read
            Default: Use motif from model
        return_mm_ml_tags (bool): Return MM and ML tags for SAM tags.
        return_mod_probs (bool): Convert returned neural network score to
            probabilities

    Returns:
        If return_mm_ml_tags, MM string tag and ML array tag
        Else if return_mod_probs, 3-tuple containing:
          1. Modified base probabilities (dim: num_calls, num_mods)
          2. Labels for each base (-1 if labels not provided)
          3. List of positions within the `read.seq`.
       Else, return value from call_read_mods_core
    """
    if focus_offset is None:
        motifs = [Motif(*mot) for mot in model_metadata["motifs"]]
        read.set_motif_focus_bases(motifs)
    else:
        read.focus_bases = np.array([focus_offset])
    read.prepare_batches(model_metadata, batch_size)
    if len(read.batches) == 0:
        return np.array([]), np.array([]), np.array([])
    nn_out, labels, pos = read.run_model(model)
    if not return_mod_probs and not return_mm_ml_tags:
        return nn_out, labels, pos
    probs = softmax_axis1(nn_out)[:, 1:].astype(np.float64)
    if return_mm_ml_tags:
        return format_mm_ml_tags(
            seq=read.str_seq,
            poss=pos,
            probs=probs,
            mod_bases=model_metadata["mod_bases"],
            can_base=model_metadata["can_base"],
        )
    return probs, labels, pos


class DuplexReadModCaller:
    def __init__(self, model, model_metadata):
        self.model = model
        self.model_metadata = model_metadata

    def call_duplex_read_mod_probs(
        self,
        duplex_read: DuplexRead,
    ):
        template_read = duplex_read.template_read.into_remora_read(False)
        complement_read = duplex_read.complement_read.into_remora_read(False)

        template_probs, _, template_positions = call_read_mods(
            read=template_read,
            model=self.model,
            model_metadata=self.model_metadata,
            return_mod_probs=True,
        )
        template_positions = template_positions + duplex_read.template_ref_start

        complement_probs, _, complement_positions = call_read_mods(
            read=complement_read,
            model=self.model,
            model_metadata=self.model_metadata,
            return_mod_probs=True,
        )
        complement_positions = (
            complement_positions + duplex_read.complement_ref_start
        )

        read_sequence = (
            duplex_read.duplex_basecalled_sequence
            if not duplex_read.is_reverse_mapped
            else revcomp(duplex_read.duplex_basecalled_sequence)
        )

        if duplex_read.is_reverse_mapped:
            (template_positions, template_probs), (
                complement_positions,
                complement_probs,
            ) = (complement_positions, complement_probs), (
                template_positions,
                template_probs,
            )

        complement_positions_duplex_orientation = (
            len(read_sequence) - complement_positions - 1
        )

        return {
            "template_probs": template_probs,
            "template_positions": template_positions,
            "complement_probs": complement_probs,
            "complement_positions": complement_positions_duplex_orientation,
            "read_sequence": read_sequence,
        }

    def call_duplex_read_mods(
        self,
        duplex_read: DuplexRead,
    ) -> (str, pyarray.array):
        duplex_read_probs = self.call_duplex_read_mod_probs(duplex_read)

        template_mm, template_ml = format_mm_ml_tags(
            seq=duplex_read_probs["read_sequence"],
            poss=duplex_read_probs["template_positions"],
            probs=duplex_read_probs["template_probs"],
            mod_bases=self.model_metadata["mod_bases"],
            can_base=self.model_metadata["can_base"],
            strand="+",
        )
        complement_mm, complement_ml = format_mm_ml_tags(
            seq=duplex_read_probs["read_sequence"],
            poss=duplex_read_probs["complement_positions"],
            probs=duplex_read_probs["complement_probs"],
            mod_bases=self.model_metadata["mod_bases"],
            can_base=revcomp(self.model_metadata["can_base"]),
            strand="-",
        )
        mm_tag = f"MM:Z:{template_mm + complement_mm}"
        ml_tag_values = template_ml + complement_ml
        ml_tag_stringed = ",".join([str(x) for x in ml_tag_values])
        ml_tag = f"ML:B:C,{ml_tag_stringed}"

        return mm_tag, ml_tag


def check_simplex_alignments(*, simplex_index, duplex_index, pairs):
    """Check that valid pairs are found

    Args:
        simplex_index (ReadIndexedBam): Simplex basecalls bam index
        duplex_index (ReadIndexedBam): Duplex basecalls bam index
        pairs (list): List of read pair strings
    """
    if len(pairs) == 0:
        raise ValueError("no pairs found in file")
    all_paired_read_ids = set(chain(*pairs))
    simplex_read_ids = set(simplex_index.read_ids)
    duplex_read_ids = set(duplex_index.read_ids)
    count_paired_simplex_alignments = sum(
        [1 for read_id in all_paired_read_ids if read_id in simplex_read_ids]
    )
    if count_paired_simplex_alignments == 0:
        raise ValueError("zero simplex alignments found")

    # valid meaning we have all the parts to perform inference
    valid_read_pairs = [
        (t, c)
        for t, c in pairs
        if all(read_id in simplex_read_ids for read_id in (t, c))
        and t in duplex_read_ids
    ]
    return valid_read_pairs, len(valid_read_pairs)


def prep_duplex_read_builder(simplex_index, pod5_path):
    """Prepare a duplex pair builder object. For example pass to MultitaskMap
    prep_func argument.

    Args:
        simplex_index (io.ReadIndexedBam): Read indexed BAM file handle
        pod5_path (pod5.reader.Reader): POD5 file handle
    """
    builder = DuplexPairsBuilder(
        simplex_index=simplex_index,
        pod5_path=pod5_path,
    )
    return [builder], {}


def iter_duplexed_io_reads(read_id_pair, builder):
    return builder.make_read_pair(read_id_pair)


def infer_duplex(
    *,
    simplex_pod5_path: str,
    simplex_bam_path: str,
    duplex_bam_path: str,
    pairs_path: str,
    model,
    model_metadata,
    out_bam,
    num_extract_alignment_threads,
    num_duplex_prep_workers,
    num_infer_threads,
    num_reads=None,
    skip_non_primary=True,
    duplex_deliminator=";",
):
    duplex_bam_index = ReadIndexedBam(
        duplex_bam_path,
        skip_non_primary=skip_non_primary,
        req_tags=set(),
        read_id_converter=lambda k: k.split(duplex_deliminator)[0],
    )

    simplex_bam_index = ReadIndexedBam(
        simplex_bam_path, skip_non_primary=True, req_tags={"mv"}
    )
    with open(pairs_path, "r") as fh:
        pairs = [tuple(line.split()) for line in fh]
    valid_pairs, num_valid_reads = check_simplex_alignments(
        simplex_index=simplex_bam_index,
        duplex_index=duplex_bam_index,
        pairs=pairs,
    )
    if num_reads is not None:
        num_reads = min(num_valid_reads, num_reads)

    # source of pipeline, template, complement read ID pairs
    # produces template, complement read id tuples
    def iter_pairs(pairs, num_reads):
        for pair_num, pair in enumerate(pairs):
            if num_reads is not None and pair_num >= num_reads:
                return
            yield pair

    read_id_pairs = BackgroundIter(
        iter_pairs,
        kwargs=dict(pairs=valid_pairs, num_reads=num_reads),
        use_process=False,
    )

    # consumes: tuple of template, complement read Ids
    # prep: open resources for Pod5 and simplex BAM
    # produces: (io.Read, io.Read), str
    io_read_pairs_results = MultitaskMap(
        iter_duplexed_io_reads,
        read_id_pairs,
        prep_func=prep_duplex_read_builder,
        args=(simplex_bam_index, simplex_pod5_path),
        name="BuildDuplexedIoReads",
        num_workers=num_extract_alignment_threads,
        use_process=False,
        use_mp_queue=False,
    )

    def make_duplex_reads(read_pair_result, duplex_index, bam_file_handle):
        read_pair, err = read_pair_result
        if err is not None or read_pair is None:
            return None, err
        template, complement = read_pair
        if template.read_id not in duplex_index:
            return None, "duplex BAM record not found for read_id"
        for pointer in duplex_index[template.read_id]:
            bam_file_handle.seek(pointer)
            bam_record = next(bam_file_handle)
            duplex_read = DuplexRead.from_reads_and_alignment(
                template_read=template,
                complement_read=complement,
                duplex_alignment=bam_record,
            )

            return duplex_read, None

    # consumes: tuple of io.Reads (template, complement)
    # produces: (DuplexRead, str), for inference by the model
    duplex_aln = pysam.AlignmentFile(duplex_bam_path, "rb", check_sq=False)
    duplex_reads = MultitaskMap(
        make_duplex_reads,
        io_read_pairs_results,
        num_workers=num_duplex_prep_workers,
        args=(duplex_bam_index, duplex_aln),
        name="MakeDuplexReads",
        use_process=False,
        use_mp_queue=False,
    )

    def add_mod_mappings_to_alignment(duplex_read_result, caller):
        duplex_read, err = duplex_read_result
        if err is not None:
            return None, err
        mod_tags = caller.call_duplex_read_mods(duplex_read)
        mod_tags = list(mod_tags)
        assert len(mod_tags) == 2
        record = duplex_read.duplex_alignment
        record = copy(record)
        record["tags"] = [
            tag
            for tag in record["tags"]
            if not (tag.startswith("MM") or tag.startswith("ML"))
        ]
        record["tags"].extend(mod_tags)
        return record, None

    duplex_caller = DuplexReadModCaller(model, model_metadata)

    # consumes: Result[DuplexReads, str]
    # produces: (dict, str) the dict is the BAM record with mods
    # added/substituted mod tags
    alignment_records_with_mod_tags = MultitaskMap(
        add_mod_mappings_to_alignment,
        duplex_reads,
        num_workers=num_infer_threads,
        args=(duplex_caller,),
        name="InferMods",
        use_process=False,
        use_mp_queue=False,
    )

    errs = defaultdict(int)
    pysam_save = pysam.set_verbosity(0)
    with pysam.AlignmentFile(duplex_bam_path, "rb", check_sq=False) as in_bam:
        with pysam.AlignmentFile(out_bam, "wb", template=in_bam) as out_bam:
            for mod_read_mapping, err in tqdm(
                alignment_records_with_mod_tags,
                dynamic_ncols=True,
                total=num_reads,
                disable=os.environ.get("LOG_SAFE", False),
            ):
                if err is not None:
                    errs[err] += 1
                    continue
                out_bam.write(
                    pysam.AlignedSegment.from_dict(
                        mod_read_mapping, out_bam.header
                    )
                )
    pysam.set_verbosity(pysam_save)
    duplex_aln.close()

    if len(errs) > 0:
        err_types = sorted([(num, err) for err, num in errs.items()])[::-1]
        err_str = "\n".join(f"{num:>7} : {err:<80}" for num, err in err_types)
        LOGGER.info(f"Unsuccessful read reasons:\n{err_str}")


if __name__ == "__main__":
    NotImplementedError("This is a module.")
