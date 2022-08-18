import os
from copy import copy
from pathlib import Path
from collections import defaultdict

import pysam
import torch
import numpy as np
from tqdm import tqdm
from pod5_format import CombinedReader
from torch.jit._script import RecursiveScriptModule

from remora import constants, log, RemoraError, encoded_kmers
from remora.data_chunks import RemoraDataset, RemoraRead, compute_ref_to_signal
from remora.io import (
    index_bam,
    iter_signal,
    prep_extract_alignments,
    extract_alignments,
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
        model (ort.InferenceSession): Inference model
            (see remora.model_util.load_onnx_model)
        model_metadata (ort.InferenceSession): Inference model metadata
        batch_size (int): Number of chunks to call per-batch
        focus_offset (int): Specific base to call within read
            Default: Use motif from model

    Returns:
        3-tuple containing:
          1. Modified base predictions (dim: num_calls, num_mods + 1)
          2. Labels for each base (-1 if labels not provided)
          3. List of positions within the read
    """
    is_torch_model = isinstance(model, RecursiveScriptModule)
    if is_torch_model:
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
        if is_torch_model:
            read_outputs.append(
                model.forward(
                    sigs=torch.from_numpy(sigs).to(device),
                    seqs=torch.from_numpy(enc_kmers).to(device),
                )
                .detach()
                .cpu()
                .numpy()
            )
        else:
            read_outputs.append(
                model.run([], {"sig": sigs, "seq": enc_kmers})[0]
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
        model (ort.InferenceSession): Inference model
            (see remora.model_util.load_onnx_model)
        model_metadata (ort.InferenceSession): Inference model metadata
        batch_size (int): Number of chunks to call per-batch
        focus_offset (int): Specific base to call within read
            Default: Use motif from model
        return_mm_ml_tags (bool): Return MM and ML tags for SAM tags.
        return_mod_probs (bool): Convert returned neural network score to
            probabilities

    Returns:
        If return_mm_ml_tags, MM string tag and ML array tag
        Else if return_mod_probs, 3-tuple containing:
          1. Modified base probabilties (dim: num_calls, num_mods)
          2. Labels for each base (-1 if labels not provided)
          3. List of positions within the read
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
            read.str_seq,
            pos,
            probs,
            model_metadata["mod_bases"],
            model_metadata["can_base"],
        )
    return probs, labels, pos


################
# POD5+BAM CLI #
################


def mods_tags_to_str(mods_tags):
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
        read.ref_to_signal = compute_ref_to_signal(
            read.query_to_signal, read.cigar, len(read.ref_seq)
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
        err = f"Remora read prep error: {e}"
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
    in_bam = pysam.AlignmentFile(bam_fn, "rb")
    out_bam = pysam.AlignmentFile(out_fn, "wb", template=in_bam)
    sig_called = 0
    bar = tqdm(
        smoothing=0,
        total=num_reads,
        unit=" Reads",
        desc="Inferring mods",
    )
    for mod_read_mappings in mod_reads_mappings:
        bar.update()
        if len(mod_read_mappings) == 0:
            errs["No valid mappings"] += 1
            continue
        if mod_read_mappings[0][0] is not None:
            sig_called += mod_read_mappings[0][0].signal.size
            msps = sig_called / 1_000_000 / bar.format_dict["elapsed"]
            bar.set_postfix_str(f"{msps:.2f} Msamps/s", refresh=False)
        for mod_mapping, err in mod_read_mappings:
            if mod_mapping is None:
                errs[err] += 1
                continue
            out_bam.write(
                pysam.AlignedSegment.from_dict(
                    mod_mapping.full_align, out_bam.header
                )
            )
    bar.close()
    pysam.set_verbosity(pysam_save)
    if len(errs) > 0:
        err_types = sorted([(num, err) for err, num in errs.items()])[::-1]
        err_str = "\n".join(f"{num:>7} : {err:<80}" for num, err in err_types)
        LOGGER.info(f"Unsuccessful read reasons:\n{err_str}")


if __name__ == "__main__":
    NotImplementedError("This is a module.")
