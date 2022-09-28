import os
from copy import copy
import array as pyarray
from pathlib import Path
from typing import Tuple
from collections import defaultdict

import pysam
import torch
import numpy as np
from tqdm import tqdm
from pod5_format import CombinedReader
from torch.jit._script import RecursiveScriptModule

from remora import constants, log, RemoraError, encoded_kmers
from remora.data_chunks import (
    RemoraDataset,
    RemoraRead,
    compute_ref_to_signal,
)
from remora.io import (
    index_bam,
    iter_signal,
    prep_extract_alignments,
    extract_alignments,
    DuplexRead,
    Read as IoRead,
    DuplexPairsIter,
)
from remora.util import (
    MultitaskMap,
    BackgroundIter,
    format_mm_ml_tags,
    softmax_axis1,
    Motif,
    revcomp,
)

LOGGER = log.get_logger()
_PROF_FN = os.getenv("REMORA_INFER_PROFILE_FILE")


################
# Core Methods #
################


def call_read_mods_core(
    read,
    model,
    model_metadata,
    batch_size=constants.DEFAULT_BATCH_SIZE,
    focus_offset=None,
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

    Returns:
        3-tuple containing:
          1. Modified base predictions (dim: num_calls, num_mods + 1)
          2. Labels for each base (-1 if labels not provided)
          3. List of positions within the read
    """
    device = next(model.parameters()).device
    read.refine_signal_mapping(model_metadata["sig_map_refiner"])
    motifs = [Motif(*mot) for mot in model_metadata["motifs"]]
    bb, ab = model_metadata["kmer_context_bases"]
    if focus_offset is not None:
        read.focus_bases = np.array([focus_offset])
    else:
        read.add_motif_focus_bases(motifs)
    chunks = list(
        read.iter_chunks(
            model_metadata["chunk_context"],
            model_metadata["kmer_context_bases"],
            model_metadata["base_pred"],
            model_metadata["base_start_justify"],
            model_metadata["offset"],
        )
    )
    if len(chunks) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.long),
            [],
        )
    dataset = RemoraDataset.allocate_empty_chunks(
        num_chunks=len(chunks),
        chunk_context=model_metadata["chunk_context"],
        max_seq_len=max(c.seq_len for c in chunks),
        kmer_context_bases=model_metadata["kmer_context_bases"],
        base_pred=model_metadata["base_pred"],
        mod_bases=model_metadata["mod_bases"],
        mod_long_names=model_metadata["mod_long_names"],
        motifs=[mot.to_tuple() for mot in motifs],
        batch_size=batch_size,
        shuffle_on_iter=False,
        drop_last=False,
    )
    for chunk in chunks:
        dataset.add_chunk(chunk)
    dataset.set_nbatches()
    read_outputs, read_poss, read_labels = [], [], []
    for (sigs, seqs, seq_maps, seq_lens), labels, (_, read_pos) in dataset:
        enc_kmers = encoded_kmers.compute_encoded_kmer_batch(
            bb, ab, seqs, seq_maps, seq_lens
        )
        read_outputs.append(
            model.forward(
                sigs=torch.from_numpy(sigs).to(device),
                seqs=torch.from_numpy(enc_kmers).to(device),
            )
            .detach()
            .cpu()
            .numpy()
        )
        read_labels.append(labels)
        read_poss.append(read_pos)
    read_outputs = np.concatenate(read_outputs, axis=0)
    read_labels = np.concatenate(read_labels)
    read_poss = np.concatenate(read_poss)
    return read_outputs, read_labels, read_poss


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
    nn_out, labels, pos = call_read_mods_core(
        read,
        model,
        model_metadata,
    )
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
        LOGGER.debug(
            f"calling mods on duplex read {duplex_read.duplex_read_id}"
        )
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


################
# POD5+BAM CLI #
################


def mods_tags_to_str(mods_tags):
    # TODO these operations are often quite slow
    return [
        f"MM:Z:{mods_tags[0]}",
        f"ML:B:C,{','.join(map(str, mods_tags[1]))}",
    ]


def infer_mods(
    read_errs, model, model_metadata, ref_anchored=False, check_read=False
):
    try:
        read = next((read for read, _ in read_errs if read is not None))
    except StopIteration:
        return read_errs
    trim_signal = read.signal[
        read.query_to_signal[0] : read.query_to_signal[-1]
    ]
    shift_q_to_sig = read.query_to_signal - read.query_to_signal[0]
    if ref_anchored:
        assert (
            read.ref_seq is not None
        ), "cannot do ref_anchored when ref_seq is None"
        read.ref_to_signal = compute_ref_to_signal(
            read.query_to_signal,
            read.cigar,
            query_seq=read.seq,
            ref_seq=read.ref_seq,
        )
        trim_signal = read.signal[
            read.ref_to_signal[0] : read.ref_to_signal[-1]
        ]
        shift_ref_to_sig = read.ref_to_signal - read.ref_to_signal[0]
        remora_read = RemoraRead(
            dacs=trim_signal,
            shift=read.shift_dacs_to_norm,
            scale=read.scale_dacs_to_norm,
            seq_to_sig_map=shift_ref_to_sig,
            str_seq=read.ref_seq,
            read_id=read.read_id,
        )
    else:
        remora_read = RemoraRead(
            dacs=trim_signal,
            shift=read.shift_dacs_to_norm,
            scale=read.scale_dacs_to_norm,
            seq_to_sig_map=shift_q_to_sig,
            str_seq=read.seq,
            read_id=read.read_id,
        )
    try:
        if check_read:
            remora_read.check()
    except RemoraError as e:
        # TODO figure out what exactly is going on here.
        #   hopefully what will happen is err will end up being
        #   carried along when mapping is None below.
        err = f"Remora read prep error: {e}"
        LOGGER.debug(err)

    mod_tags = mods_tags_to_str(
        call_read_mods(
            remora_read,
            model,
            model_metadata,
            return_mm_ml_tags=True,
        )
    )
    mod_read_mappings = []
    for mapping, err in read_errs:
        # TODO add check that seq and cigar are the same
        if mapping is None:
            mod_read_mappings.append(tuple((None, err)))
            continue
        mod_mapping = copy(mapping)
        mod_mapping.full_align["tags"] = [
            tag
            for tag in mod_mapping.full_align["tags"]
            if not (tag.startswith("MM") or tag.startswith("ML"))
        ]
        mod_mapping.full_align["tags"].extend(mod_tags)
        if ref_anchored:
            mod_mapping.full_align["cigar"] = f"{len(mod_mapping.ref_seq)}M"
            mod_mapping.full_align["seq"] = (
                read.ref_seq
                if read.ref_pos.strand == "+"
                else revcomp(read.ref_seq)
            )
            mod_mapping.full_align["qual"] = "*"
        mod_read_mappings.append(tuple((mod_mapping, None)))
    return mod_read_mappings


if _PROF_FN:
    _infer_mods_wrapper = infer_mods

    def infer_mods(*args, **kwargs):
        import cProfile

        prof = cProfile.Profile()
        retval = prof.runcall(_infer_mods_wrapper, *args, **kwargs)
        prof.dump_stats(_PROF_FN)
        return retval


def infer_from_pod5_and_bam(
    pod5_fn,
    bam_fn,
    model,
    model_metadata,
    out_fn,
    num_reads,
    num_extract_alignment_threads,
    num_extract_chunks_threads,
    skip_non_primary=True,
    ref_anchored=False,
):
    bam_idx, num_bam_reads = index_bam(bam_fn, skip_non_primary)
    with CombinedReader(Path(pod5_fn)) as pod5_fp:
        num_pod5_reads = sum(1 for _ in pod5_fp.reads())
        LOGGER.info(
            f"Found {num_bam_reads} BAM records and "
            f"{num_pod5_reads} POD5 reads"
        )
        if num_reads is None:
            num_reads = min(num_pod5_reads, num_bam_reads)
        else:
            num_reads = min(num_reads, num_pod5_reads, num_bam_reads)
    signals = BackgroundIter(
        iter_signal,
        args=(pod5_fn, num_reads, list(bam_idx.keys())),
        name="ExtractSignal",
        use_process=True,
    )
    reads = MultitaskMap(
        extract_alignments,
        signals,
        prep_func=prep_extract_alignments,
        num_workers=num_extract_alignment_threads,
        args=(bam_idx, bam_fn),
        kwargs={"req_tags": {"mv"}},
        name="AddAlignments",
        use_process=True,
    )

    use_process = True
    if isinstance(model, RecursiveScriptModule):
        use_process = next(model.parameters()).device.type == "cpu"

    mod_reads_mappings = MultitaskMap(
        infer_mods,
        reads,
        num_workers=num_extract_chunks_threads,
        args=(model, model_metadata, ref_anchored),
        name="InferMods",
        use_process=use_process,
    )

    errs = defaultdict(int)
    pysam_save = pysam.set_verbosity(0)
    with pysam.AlignmentFile(bam_fn, "rb") as in_bam:
        with pysam.AlignmentFile(out_fn, "wb", template=in_bam) as out_bam:
            sig_called = 0
            progress_bar = tqdm(
                smoothing=0,
                total=num_reads,
                unit=" Reads",
                desc="Inferring mods",
            )
            for mod_read_mappings in mod_reads_mappings:
                progress_bar.update()
                if len(mod_read_mappings) == 0:
                    errs["No valid mappings"] += 1
                    continue

                if mod_read_mappings[0][0] is not None:
                    sig_called += mod_read_mappings[0][0].signal.size
                    msps = (
                        sig_called
                        / 1_000_000
                        / progress_bar.format_dict["elapsed"]
                    )
                    progress_bar.set_postfix_str(
                        f"{msps:.2f} Msamps/s", refresh=False
                    )

                for mod_mapping, err in mod_read_mappings:
                    if mod_mapping is None:
                        errs[err] += 1
                        continue
                    out_bam.write(
                        pysam.AlignedSegment.from_dict(
                            mod_mapping.full_align, out_bam.header
                        )
                    )
            progress_bar.close()
            pysam.set_verbosity(pysam_save)

    if len(errs) > 0:
        err_types = sorted([(num, err) for err, num in errs.items()])[::-1]
        err_str = "\n".join(f"{num:>7} : {err:<80}" for num, err in err_types)
        LOGGER.info(f"Unsuccessful read reasons:\n{err_str}")


def infer_duplex(
    *,
    simplex_pod5_fp: str,
    simplex_bam_fp: str,
    duplex_bam_fp: str,
    pairs_fp: str,
    model,
    model_metadata,
    out_fn,
    num_extract_alignment_threads,
    num_infer_threads,
    skip_non_primary=True,
):
    duplex_bam_index, _ = index_bam(
        duplex_bam_fp, skip_non_primary=skip_non_primary, req_tags=set()
    )

    def iterate_duplex_pairs(pairs, pod5, simplex):
        duplex_pairs_iter = DuplexPairsIter(
            pairs_fp=pairs,
            pod5_fp=pod5,
            simplex_bam_fp=simplex,
        )
        for read_pair in duplex_pairs_iter:
            yield read_pair

    read_pairs = BackgroundIter(
        iterate_duplex_pairs,
        args=(pairs_fp, simplex_pod5_fp, simplex_bam_fp),
        name="DuplexPairsIter",
        use_process=True,
    )

    def make_duplex_reads(
        read_pair: Tuple[IoRead, IoRead], duplex_index, bam_file_handle
    ):
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

    # not sure if this really needs to be another step, could just make duplex
    # read assembly part of the first step since it will likely be the slow part
    duplex_aln = pysam.AlignmentFile(duplex_bam_fp, "rb", check_sq=False)
    duplex_reads = MultitaskMap(
        make_duplex_reads,
        read_pairs,
        num_workers=num_extract_alignment_threads,
        args=(duplex_bam_index, duplex_aln),
        name="MakeDuplexReads",
        use_process=True,
    )

    def add_mod_mappings_to_alignment(
        duplex_read_result: Tuple[DuplexRead, Exception],
        caller: DuplexReadModCaller,
    ):
        duplex_read, exc = duplex_read_result
        if exc is not None:
            return None, exc
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
    use_process = True
    if isinstance(model, RecursiveScriptModule):
        use_process = next(model.parameters()).device.type == "cpu"
    alignment_records_with_mod_tags = MultitaskMap(
        add_mod_mappings_to_alignment,
        duplex_reads,
        num_workers=num_infer_threads,
        args=(duplex_caller,),
        name="InferMods",
        use_process=use_process,
    )

    errs = defaultdict(int)
    pysam_save = pysam.set_verbosity(0)
    with pysam.AlignmentFile(duplex_bam_fp, "rb", check_sq=False) as in_bam:
        with pysam.AlignmentFile(out_fn, "wb", template=in_bam) as out_bam:
            for mod_read_mapping, err in tqdm(alignment_records_with_mod_tags):
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
        # TODO make a proper run report that tabulates the errors by read_id
        # to make forensics easier
        LOGGER.info(f"Unsuccessful read reasons:\n{err_str}")


if __name__ == "__main__":
    NotImplementedError("This is a module.")
