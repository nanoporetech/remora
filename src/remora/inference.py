import os
from copy import copy
import array as pyarray
from pathlib import Path
from typing import Tuple
from collections import defaultdict

import pysam
import numpy as np
from tqdm import tqdm
from pod5_format import CombinedReader
from torch.jit._script import RecursiveScriptModule

from remora import constants, log, RemoraError
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
_PROF_PREP_FN = os.getenv("REMORA_INFER_PREP_DATA_PROFILE_FILE")
_PROF_MODEL_FN = os.getenv("REMORA_INFER_RUN_MODEL_PROFILE_FILE")
_PROF_MAIN_FN = os.getenv("REMORA_INFER_MAIN_PROFILE_FILE")


################
# Core Methods #
################


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


################
# Duplex Infer #
################


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


def prepare_batches(read_errs, model_metadata, batch_size, ref_anchored=False):
    out_read_errs = []
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
        motifs = [Motif(*mot) for mot in model_metadata["motifs"]]
        remora_read.set_motif_focus_bases(motifs)
        remora_read.prepare_batches(model_metadata, batch_size)
        out_read_errs.append((copy(io_read), remora_read, None))
    return out_read_errs


if _PROF_PREP_FN:
    _prepare_batches_wrapper = prepare_batches

    def prepare_batches(*args, **kwargs):
        import cProfile

        prof = cProfile.Profile()
        retval = prof.runcall(_prepare_batches_wrapper, *args, **kwargs)
        prof.dump_stats(_PROF_PREP_FN)
        return retval


def run_model(read_errs, model, model_metadata, ref_anchored):
    out_read_errs = []
    for io_read, remora_read, err in read_errs:
        if err is not None:
            out_read_errs.append((None, err))
            continue
        # TODO super-batch reads to optimize performance for short reads
        nn_out, labels, pos = remora_read.run_model(model)
        probs = softmax_axis1(nn_out)[:, 1:].astype(np.float64)
        mod_tags = mods_tags_to_str(
            format_mm_ml_tags(
                seq=remora_read.str_seq,
                poss=pos,
                probs=probs,
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
                if io_read.ref_pos.strand == "+"
                else revcomp(io_read.ref_seq)
            )
            io_read.full_align["qual"] = "*"
        out_read_errs.append((io_read, None))
    return out_read_errs


if _PROF_MODEL_FN:
    _run_model_wrapper = run_model

    def run_model(*args, **kwargs):
        import cProfile

        prof = cProfile.Profile()
        retval = prof.runcall(_run_model_wrapper, *args, **kwargs)
        prof.dump_stats(_PROF_MODEL_FN)
        return retval


def infer_from_pod5_and_bam(
    pod5_path,
    in_bam_path,
    model,
    model_metadata,
    out_bam_path,
    num_reads=None,
    num_extract_alignment_workers=1,
    num_prep_batch_workers=1,
    num_infer_workers=1,
    batch_size=constants.DEFAULT_BATCH_SIZE,
    skip_non_primary=True,
    ref_anchored=False,
):
    bam_idx, num_bam_reads = index_bam(in_bam_path, skip_non_primary)
    with CombinedReader(Path(pod5_path)) as pod5_fh:
        num_pod5_reads = sum(1 for _ in pod5_fh.reads())
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
        args=(pod5_path, num_reads, list(bam_idx.keys())),
        name="ExtractSignal",
        use_process=True,
    )
    reads = MultitaskMap(
        extract_alignments,
        signals,
        prep_func=prep_extract_alignments,
        num_workers=num_extract_alignment_workers,
        args=(bam_idx, in_bam_path),
        kwargs={"req_tags": {"mv"}},
        name="AddAlignments",
        use_process=True,
    )
    reads = MultitaskMap(
        prepare_batches,
        reads,
        num_workers=num_prep_batch_workers,
        args=(model_metadata, batch_size),
        name="PrepBatches",
        use_process=True,
    )

    use_process = True
    if isinstance(model, RecursiveScriptModule):
        use_process = next(model.parameters()).device.type == "cpu"

    mod_reads_mappings = MultitaskMap(
        run_model,
        reads,
        num_workers=num_infer_workers,
        args=(model, model_metadata, ref_anchored),
        name="InferMods",
        use_process=use_process,
    )

    errs = defaultdict(int)
    pysam_save = pysam.set_verbosity(0)
    sig_called = 0
    in_bam = out_bam = pbar = None
    try:
        in_bam = pysam.AlignmentFile(in_bam_path, "rb")
        out_bam = pysam.AlignmentFile(out_bam_path, "wb", template=in_bam)
        pbar = tqdm(
            smoothing=0,
            total=num_reads,
            unit=" Reads",
            desc="Inferring mods",
        )
        for read_errs in mod_reads_mappings:
            pbar.update()
            if len(read_errs) == 0:
                errs["No valid mappings"] += 1
                continue

            sig_called += sum(
                read.signal.size for read, _ in read_errs if read is not None
            )
            msps = sig_called / 1_000_000 / pbar.format_dict["elapsed"]
            pbar.set_postfix_str(f"{msps:.2f} Msamps/s", refresh=False)

            for io_read, err in read_errs:
                if io_read is None:
                    errs[err] += 1
                    continue
                out_bam.write(
                    pysam.AlignedSegment.from_dict(
                        io_read.full_align, out_bam.header
                    )
                )
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


if _PROF_MAIN_FN:
    _infer_from_pod5_and_bam_wrapper = infer_from_pod5_and_bam

    def infer_from_pod5_and_bam(*args, **kwargs):
        import cProfile

        prof = cProfile.Profile()
        retval = prof.runcall(_infer_from_pod5_and_bam_wrapper, *args, **kwargs)
        prof.dump_stats(_PROF_MAIN_FN)
        return retval


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
    num_infer_threads,
    skip_non_primary=True,
):
    duplex_bam_index, _ = index_bam(
        duplex_bam_path, skip_non_primary=skip_non_primary, req_tags=set()
    )

    def iterate_duplex_pairs(pairs, pod5, simplex):
        duplex_pairs_iter = DuplexPairsIter(
            pairs_path=pairs,
            pod5_path=pod5,
            simplex_bam_path=simplex,
        )
        for read_pair in duplex_pairs_iter:
            yield read_pair

    read_pairs = BackgroundIter(
        iterate_duplex_pairs,
        args=(pairs_path, simplex_pod5_path, simplex_bam_path),
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
    duplex_aln = pysam.AlignmentFile(duplex_bam_path, "rb", check_sq=False)
    duplex_reads = MultitaskMap(
        make_duplex_reads,
        read_pairs,
        num_workers=num_extract_alignment_threads,
        args=(duplex_bam_index, duplex_aln),
        name="MakeDuplexReads",
        use_process=True,
    )

    def add_mod_mappings_to_alignment(
        duplex_read_result: Tuple[DuplexRead, str],
        caller: DuplexReadModCaller,
    ):
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
    with pysam.AlignmentFile(duplex_bam_path, "rb", check_sq=False) as in_bam:
        with pysam.AlignmentFile(out_bam, "wb", template=in_bam) as out_bam:
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
