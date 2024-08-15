import os
import sys
import array
import torch
from copy import copy
import array as pyarray
from pathlib import Path
from threading import Thread
from functools import partial
from itertools import chain, islice
from collections import defaultdict

import pysam
import numpy as np
from tqdm import tqdm
from pod5 import DatasetReader

from remora import constants, log, RemoraError, __version__
from remora.data_chunks import CoreRemoraDataset, DatasetMetadata
from remora.io import (
    ReadIndexedBam,
    get_read_ids,
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


def mods_tags_to_str(mm_tags, ml_arr):
    # TODO these operations are often quite slow
    return [
        f"MM:Z:{''.join(mm_tags)}",
        f"ML:B:C,{','.join(map(str, ml_arr))}",
    ]


def prepare_reads(read_errs, models_metadata, ref_anchored, drop_move_tag=True):
    out_read_errs = []

    models_kwargs = []
    motifs = {}
    for md in models_metadata:
        motif_seqs, motif_offsets = zip(*md["motifs"])
        motifs[md["can_base"]] = [Motif(*mot) for mot in md["motifs"]]
        models_kwargs.append(
            {
                "mod_bases": md["mod_bases"],
                "mod_long_names": md["mod_long_names"],
                "motif_sequences": motif_seqs,
                "motif_offsets": motif_offsets,
                "chunk_context": md["chunk_context"],
                "kmer_context_bases": md["kmer_context_bases"],
                "extra_metadata_arrays": {"read_focus_base": ("int64", "")},
            }
        )
    for io_read, err in read_errs:
        if err is not None:
            io_read.prune(drop_move_tag=drop_move_tag)
            out_read_errs.append((io_read, None, err))
            continue
        drop_tags = set((("MM", "ML", "MN")))
        if drop_move_tag:
            drop_tags.add("mv")
        if ref_anchored:
            # drop all minimap align tags
            drop_tags.update(
                (
                    "NM",
                    "MD",
                    "tp",
                    "cm",
                    "s1",
                    "s2",
                    "AS",
                    "SA",
                    "ms",
                    "nn",
                    "cs",
                    "dv",
                    "de",
                    "rl",
                )
            )
        try:
            remora_read = io_read.into_remora_read(ref_anchored)
        except RemoraError as e:
            io_read.prune(drop_tags)
            LOGGER.debug(f"{io_read.child_read_id} Read prep error: {e}")
            out_read_errs.append((io_read, None, f"Read prep error: {e}"))
            continue
        except Exception as e:
            io_read.prune(drop_tags)
            LOGGER.debug(f"{io_read.child_read_id} Unexpected error: {e}")
            out_read_errs.append((io_read, None, f"Unexpected error: {e}"))
            continue
        # after creating remora read strip IO read of large arrays
        io_read.prune(drop_tags)
        if ref_anchored:
            io_read.full_align["tags"].extend(
                (
                    "NM:i:0",
                    f"MN:i:{io_read.ref_seq_len}",
                    f"MD:Z:{io_read.ref_seq_len}",
                )
            )
        else:
            io_read.full_align["tags"].append(f"MN:i:{io_read.seq_len}")
        datasets = {}
        for md, md_kwargs in zip(models_metadata, models_kwargs):
            mdl_remora_read = remora_read.copy()
            mdl_remora_read.set_motif_focus_bases(motifs[md["can_base"]])
            mdl_remora_read.refine_signal_mapping(md["sig_map_refiner"])
            chunks = list(
                mdl_remora_read.iter_chunks(
                    md["chunk_context"],
                    md["kmer_context_bases"],
                    md["base_start_justify"],
                    md["offset"],
                )
            )
            if len(chunks) == 0:
                LOGGER.debug(
                    f"{io_read.child_read_id} No {md['can_base']} mod calls"
                )
                out_read_errs.append(
                    (io_read, None, f"No {md['can_base']} mod calls")
                )
                continue
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
                return_arrays=[
                    "signal",
                    "enc_kmer",
                    "read_focus_base",
                ],
            )
            for chunk in chunks:
                dataset.write_chunk(chunk)
            datasets[md["can_base"]] = dataset
        out_read_errs.append((io_read, datasets, None))
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
        return [(None, None, "No valid mappings")]
    read_nn_inputs = []
    for io_read, read_datasets, err in read_errs:
        if err is not None:
            read_nn_inputs.append((io_read, None, err))
            continue
        bases_chunks = {}
        for can_base, ds in read_datasets.items():
            base_chunks = next(iter(ds))
            bases_chunks[can_base] = base_chunks
        read_nn_inputs.append((io_read, bases_chunks, None))
    return read_nn_inputs


def batch_reads(prepped_nn_inputs, batches_q, batch_size, models_metadata):
    md_dict = dict((md["can_base"], md) for md in models_metadata)
    can_bases = list(md_dict)

    def new_arrays(can_base):
        return (
            np.empty(
                (batch_size, 1, md_dict[can_base]["chunk_len"]),
                dtype=np.float32,
            ),
            np.empty(
                (
                    batch_size,
                    md_dict[can_base]["kmer_len"] * 4,
                    md_dict[can_base]["chunk_len"],
                ),
                dtype=np.float32,
            ),
            np.empty(batch_size, dtype=int),
        )

    arrs = dict((cb, new_arrays(cb)) for cb in can_bases)
    # position within current batch
    b_poss = dict((cb, 0) for cb in can_bases)
    b_readss = dict((cb, []) for cb in can_bases)
    for read_nn_inputs in prepped_nn_inputs:
        for io_read, bases_chunks, err in read_nn_inputs:
            if err is not None:
                for can_base in can_bases:
                    b_readss[can_base].append([io_read, None, None, err])
                continue
            for can_base, r_chunks in bases_chunks.items():
                num_chunks = r_chunks["read_focus_base"].size
                # fill out and yield full batches
                rb_consumed = 0
                # while this read extends through a whole batch continue to
                # supply batches from this read
                while b_poss[can_base] + num_chunks - rb_consumed >= batch_size:
                    rb_en = rb_consumed + batch_size - b_poss[can_base]
                    arrs[can_base][0][b_poss[can_base] :] = r_chunks["signal"][
                        rb_consumed:rb_en
                    ]
                    arrs[can_base][1][b_poss[can_base] :] = r_chunks[
                        "enc_kmer"
                    ][rb_consumed:rb_en]
                    arrs[can_base][2][b_poss[can_base] :] = r_chunks[
                        "read_focus_base"
                    ][rb_consumed:rb_en]
                    # batch start is None once the first batch is complete
                    # from this read
                    b_st = b_poss[can_base] if rb_consumed == 0 else None
                    b_readss[can_base].append([io_read, b_st, None, None])
                    _put_item(
                        (can_base, *arrs[can_base], b_readss[can_base]),
                        batches_q,
                    )
                    rb_consumed += batch_size - b_poss[can_base]
                    # new batch
                    arrs[can_base] = new_arrays(can_base)
                    b_poss[can_base] = 0
                    b_readss[can_base] = []
                # add rest of read to unfinished batch
                b_en = b_poss[can_base] + num_chunks - rb_consumed
                arrs[can_base][0][b_poss[can_base] : b_en] = r_chunks["signal"][
                    rb_consumed:
                ]
                arrs[can_base][1][b_poss[can_base] : b_en] = r_chunks[
                    "enc_kmer"
                ][rb_consumed:]
                arrs[can_base][2][b_poss[can_base] : b_en] = r_chunks[
                    "read_focus_base"
                ][rb_consumed:]
                # if read continues from last batch set start to None
                b_st = b_poss[can_base] if rb_consumed == 0 else None
                b_readss[can_base].append([io_read, b_st, b_en, None])
                # set current batch position for next read
                b_poss[can_base] = b_en
    for can_base in can_bases:
        if b_poss[can_base] > 0:
            b_sig, b_enc_kmer, b_read_pos = arrs[can_base]
            # send last batch
            _put_item(
                (
                    can_base,
                    b_sig[: b_poss[can_base]],
                    b_enc_kmer[: b_poss[can_base]],
                    b_read_pos[: b_poss[can_base]],
                    b_readss[can_base],
                ),
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
    batches_q, called_batches_q, models, models_metadata, batch_size
):
    md_dict = dict((md["can_base"], md) for md in models_metadata)
    can_bases = list(md_dict)
    devices = dict()
    sig_arrs = dict()
    enc_kmer_arrs = dict()
    for can_base in can_bases:
        devices[can_base] = next(models[can_base].parameters()).device
        pin_memory = devices[can_base].type == "cuda"
        sig_arrs[can_base] = torch.empty(
            (batch_size, 1, md_dict[can_base]["chunk_len"]),
            dtype=torch.float32,
            pin_memory=pin_memory,
        )
        enc_kmer_arrs[can_base] = torch.empty(
            (
                batch_size,
                md_dict[can_base]["kmer_len"] * 4,
                md_dict[can_base]["chunk_len"],
            ),
            dtype=torch.float32,
            pin_memory=pin_memory,
        )
    for can_base, b_sig, b_enc_kmer, b_read_pos, b_reads in _queue_iter(
        batches_q
    ):
        if b_read_pos.size == batch_size:
            sig_arrs[can_base][:] = torch.from_numpy(b_sig)
            enc_kmer_arrs[can_base][:] = torch.from_numpy(b_enc_kmer)
        else:
            sig_arrs[can_base] = torch.from_numpy(b_sig)
            enc_kmer_arrs[can_base] = torch.from_numpy(b_enc_kmer)
        nn_out = models[can_base](
            sig_arrs[can_base].to(devices[can_base]),
            enc_kmer_arrs[can_base].to(devices[can_base]),
        )
        _put_item((can_base, nn_out, b_read_pos, b_reads), called_batches_q)
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
            if curr_read is not None:
                comp_reads.append(curr_read)
            comp_reads.append((io_read, None, None, err))
            curr_read = None
        # end of read from previous batch
        elif b_st is None:
            if curr_read is None:
                LOGGER.debug("Unbatching encountered None read")
                raise RemoraError("Unbatching encountered None read")
            if curr_read[0].read_id != io_read.read_id:
                LOGGER.debug(
                    "Unbatching encountered mismatching reads "
                    f"{curr_read[0].read_id} != {io_read.read_id}"
                )
                raise RemoraError("Unbatching encountered mismatching reads")
            io_read, r_nn_out, r_read_pos, _ = curr_read
            # update curr_read
            curr_read = (
                io_read,
                np.concatenate([r_nn_out, b_nn_out[:b_en]], axis=0),
                np.concatenate([r_read_pos, b_read_pos[:b_en]]),
                None,
            )
        else:
            if curr_read is not None:
                comp_reads.append(curr_read)
            curr_read = (
                io_read,
                b_nn_out[b_st:b_en],
                b_read_pos[b_st:b_en],
                None,
            )
    return comp_reads, curr_read


def unbatch(called_batches_q, called_reads_q, models_metadata):
    def get_return_read(reads):
        mod_calls = []
        r_errs = set()
        for can_base, (io_read, nn_out, r_pos, err) in reads:
            r_errs.add(err)
            if err is None:
                mod_calls.append((can_base, nn_out, r_pos))
        if any(err is None for err in r_errs):
            r_err = None
        else:
            r_err = ",".join(sorted(r_errs))
        return io_read, mod_calls, r_err

    can_bases = [md["can_base"] for md in models_metadata]
    num_can_bases = len(can_bases)
    curr_reads = dict((can_base, None) for can_base in can_bases)
    comp_reads = defaultdict(list)
    for can_base, nn_out, b_read_pos, b_reads in _queue_iter(called_batches_q):
        cb_comp_reads, cb_curr_read = unbatch_reads(
            curr_reads[can_base], nn_out.cpu().numpy(), b_read_pos, b_reads
        )
        curr_reads[can_base] = cb_curr_read
        for comp_read in cb_comp_reads:
            comp_reads[comp_read[0].child_read_id].append((can_base, comp_read))

        # add reads which have completed through all canonical base model
        full_comp_read_ids = [
            rid
            for rid, r_comp_reads in comp_reads.items()
            if len(r_comp_reads) == num_can_bases
        ]
        for rid in full_comp_read_ids:
            _put_item(
                get_return_read(comp_reads[rid]),
                called_reads_q,
            )
            # delete read from completed reads dict
            del comp_reads[rid]
    if curr_reads[can_bases[0]] is not None:
        _put_item(
            get_return_read([(cb, curr_reads[cb]) for cb in can_bases]),
            called_reads_q,
        )
    _put_item(StopIteration, called_reads_q)


if _PROF_UNBATCH_FN:
    _unbatch_wrapper = unbatch

    def unbatch(*args, **kwargs):
        import cProfile

        prof = cProfile.Profile()
        retval = prof.runcall(_unbatch_wrapper, *args, **kwargs)
        prof.dump_stats(_PROF_UNBATCH_FN)
        return retval


def post_process_reads(read_mapping, models_metadata, ref_anchored):
    io_read, mod_calls, err = read_mapping
    if err is not None:
        return io_read, err

    md_dict = dict((md["can_base"], md) for md in models_metadata)
    mm_tags = []
    ml_arr = array.array("B")
    for can_base, nn_out, r_poss in mod_calls:
        r_probs = softmax_axis1(nn_out)[:, 1:].astype(np.float64)
        seq = io_read.ref_seq if ref_anchored else io_read.seq
        cb_mm, cb_ml = format_mm_ml_tags(
            seq=seq,
            poss=r_poss,
            probs=r_probs,
            mod_bases=md_dict[can_base]["mod_bases"],
            can_base=can_base,
        )
        mm_tags.append(cb_mm)
        ml_arr.extend(cb_ml)

    io_read.full_align["tags"].extend(mods_tags_to_str(mm_tags, ml_arr))
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
    models,
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
    drop_move_tag=True,
):
    bam_idx = ReadIndexedBam(in_bam_path, skip_non_primary, req_tags={"mv"})
    if bam_idx.num_records == 0:
        LOGGER.info("No records found in BAM file.")
        sys.exit()
    with DatasetReader(Path(pod5_path)) as pod5_dr:
        read_ids, num_reads = get_read_ids(bam_idx, pod5_dr, num_reads)
    models_metadata = list(zip(*models))[1]
    models = dict((md["can_base"], mdl) for mdl, md in models)
    reverse_signal = models_metadata[0]["reverse_signal"]
    pa_scaling = models_metadata[0]["pa_scaling"]

    signals = BackgroundIter(
        iter_signal,
        args=(pod5_path,),
        kwargs={
            "num_reads": num_reads,
            "read_ids": read_ids,
            "rev_sig": reverse_signal,
            "pa_scaling": pa_scaling,
        },
        name="ExtractSignal",
        use_process=True,
        q_maxsize=queue_max,
    )
    reads = MultitaskMap(
        extract_alignments,
        signals,
        num_workers=num_extract_alignment_workers,
        args=(bam_idx, reverse_signal),
        name="AddAlignments",
        use_process=True,
        use_mp_queue=True,
        q_maxsize=queue_max,
    )
    prepped_reads = MultitaskMap(
        prepare_reads,
        reads,
        num_workers=num_prep_read_workers,
        args=(models_metadata, ref_anchored, drop_move_tag),
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
            models_metadata,
        ),
        name="batch_reads",
        daemon=True,
    )
    batch_reads_p.start()
    called_batches_q = NamedQueue(maxsize=4, name="CalledBatches")
    call_batches_t = Thread(
        target=run_model_batched,
        args=(batches_q, called_batches_q, models, models_metadata, batch_size),
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
            models_metadata,
        ),
        name="unbatch",
        daemon=True,
    )
    unbatch_p.start()
    final_reads = MultitaskMap(
        post_process_reads,
        _queue_iter(called_reads_q),
        num_workers=num_post_process_workers,
        args=(models_metadata, ref_anchored),
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
    for err, cnt in bam_idx.skip_reasons.items():
        errs[err] = cnt
    pysam_save = pysam.set_verbosity(0)
    sig_called = 0
    in_bam = out_bam = pbar = prev_rid = None
    try:
        in_bam = pysam.AlignmentFile(in_bam_path, "rb", check_sq=False)
        header = in_bam.header.to_dict()
        pg_recs = header.get("PG", [])
        remora_pg = {
            "PN": "remora",
            "ID": "remora",
            "CL": " ".join(sys.argv),
            "VN": __version__,
        }
        if len(pg_recs) > 0:
            num_remora = len([ln for ln in pg_recs if "remora" == ln["PN"]])
            if num_remora > 0:
                remora_pg["ID"] = f"remora.{num_remora}"
            remora_pg["PP"] = pg_recs[-1]["ID"]
        header["PG"] = pg_recs + [remora_pg]
        out_bam = pysam.AlignmentFile(out_bam_path, "wb", header=header)
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
                    [f"{q.name}: {q.qsize()}/{q.maxsize}" for q in all_qs]
                )
            )
            if io_read is None:
                # should not reach this block
                errs[err] += 1
                LOGGER.DEBUG("None io_read encountered")
                pbar.update()
                continue
            if prev_rid != io_read.read_id:
                pbar.update()
            sig_called += io_read.sig_len
            sps, mag = human_format(sig_called / pbar.format_dict["elapsed"])
            pbar.set_postfix_str(f"{sps:>5.1f} {mag}samps/s", refresh=False)
            if err is not None:
                errs[err] += 1
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
    num_paired_simplex_alignments = len(
        all_paired_read_ids.intersection(simplex_read_ids)
    )
    LOGGER.debug(
        f"Found {num_paired_simplex_alignments} simplex simplex alignments in "
        f"a pair out of {len(simplex_read_ids)} total simplex reads."
    )
    if num_paired_simplex_alignments == 0:
        raise ValueError("zero simplex alignments found")

    # valid meaning we have all the parts to perform inference
    valid_read_pairs = [
        (t, c)
        for t, c in pairs
        if t in simplex_read_ids
        and c in simplex_read_ids
        and t in duplex_read_ids
    ]
    num_valid_read_pairs = len(valid_read_pairs)
    LOGGER.debug(
        f"Found {num_valid_read_pairs} valid reads out of {len(pairs)} "
        "total pairs"
    )
    return valid_read_pairs, num_valid_read_pairs


def prep_duplex_read_builder(simplex_index, pod5_path):
    """Prepare a duplex pair builder object. For example pass to MultitaskMap
    prep_func argument.

    Args:
        simplex_index (io.ReadIndexedBam): Read indexed BAM file handle
        pod5_path (str): POD5 file handle
    """
    builder = DuplexPairsBuilder(
        simplex_index=simplex_index,
        pod5_path=pod5_path,
    )
    return [builder], {}


def iter_duplexed_io_reads(read_id_pair, builder):
    return builder.make_read_pair(read_id_pair)


def make_duplex_reads(read_pair_result, duplex_index):
    read_pair, err = read_pair_result
    if err is not None or read_pair is None:
        return read_pair, err
    template, complement = read_pair
    if template.read_id not in duplex_index:
        return read_pair, "duplex BAM record not found for read_id"
    for bam_record in duplex_index.get_alignments(template.read_id):
        duplex_read = DuplexRead.from_reads_and_alignment(
            template_read=template,
            complement_read=complement,
            duplex_alignment=bam_record,
        )
        # TODO do we want to return all the duplex mappings?
        return duplex_read, None


def add_mod_mappings_to_alignment(duplex_read_result, caller):
    duplex_read, err = duplex_read_result
    if err is not None:
        return None, err
    mod_tags = caller.call_duplex_read_mods(duplex_read)
    mod_tags = list(mod_tags)
    record = duplex_read.duplex_alignment
    record = copy(record)
    record["tags"] = [
        tag
        for tag in record["tags"]
        if not (tag.startswith("MM") or tag.startswith("ML"))
    ]
    record["tags"].extend(mod_tags)
    return record, None


def duplex_read_id_converter(read_id, duplex_deliminator):
    return read_id.split(duplex_deliminator)[0]


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
    duplex_read_id_converter_delim = partial(
        duplex_read_id_converter, duplex_deliminator=duplex_deliminator
    )
    LOGGER.info("Indexing Duplex BAM")
    duplex_bam_index = ReadIndexedBam(
        duplex_bam_path,
        skip_non_primary=skip_non_primary,
        req_tags=set(),
        read_id_converter=duplex_read_id_converter_delim,
    )
    if duplex_bam_index.num_records == 0:
        LOGGER.info("No records found in duplex BAM file.")
        sys.exit()
    LOGGER.info("Indexing Simplex BAM")
    simplex_bam_index = ReadIndexedBam(
        simplex_bam_path, skip_non_primary=True, req_tags={"mv"}
    )
    if simplex_bam_index.num_records == 0:
        LOGGER.info("No records found in simplex BAM file.")
        sys.exit()
    with open(pairs_path, "r") as fh:
        pairs = [tuple(line.split()) for line in fh]
    valid_pairs, num_valid_reads = check_simplex_alignments(
        simplex_index=simplex_bam_index,
        duplex_index=duplex_bam_index,
        pairs=pairs,
    )
    num_reads = (
        num_valid_reads
        if num_reads is None
        else min(num_valid_reads, num_reads)
    )

    # consumes: tuple of template, complement read Ids
    # prep: open resources for Pod5 and simplex BAM
    # produces: (io.Read, io.Read), str
    io_read_pairs_results = MultitaskMap(
        iter_duplexed_io_reads,
        islice(valid_pairs, num_reads),
        prep_func=prep_duplex_read_builder,
        args=(simplex_bam_index, simplex_pod5_path),
        name="BuildDuplexedIoReads",
        q_maxsize=100,
        num_workers=num_extract_alignment_threads,
        use_process=True,
        use_mp_queue=True,
    )

    # consumes: tuple of io.Reads (template, complement)
    # produces: (DuplexRead, str), for inference by the model
    duplex_reads = MultitaskMap(
        make_duplex_reads,
        io_read_pairs_results,
        num_workers=num_duplex_prep_workers,
        args=(duplex_bam_index,),
        name="MakeDuplexReads",
        q_maxsize=100,
        use_process=True,
        use_mp_queue=True,
    )

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
        q_maxsize=100,
        use_process=False,
        use_mp_queue=False,
    )

    errs = defaultdict(int)
    pysam_save = pysam.set_verbosity(0)
    with pysam.AlignmentFile(duplex_bam_path, "rb", check_sq=False) as in_bam:
        with pysam.AlignmentFile(out_bam, "wb", template=in_bam) as out_bam:
            for mod_read_mapping, err in tqdm(
                alignment_records_with_mod_tags,
                smoothing=0,
                total=num_reads,
                dynamic_ncols=True,
                unit=" Duplex Reads",
                desc="Inferring duplex mods",
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

    if len(errs) > 0:
        err_types = sorted([(num, err) for err, num in errs.items()])[::-1]
        err_str = "\n".join(f"{num:>7} : {err:<80}" for num, err in err_types)
        LOGGER.info(f"Unsuccessful read reasons:\n{err_str}")


if __name__ == "__main__":
    NotImplementedError("This is a module.")
