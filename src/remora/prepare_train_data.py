from pathlib import Path
from collections import defaultdict

import pod5
import pysam
import numpy as np
from tqdm import tqdm

from remora import log, RemoraError
from remora.util import MultitaskMap, BackgroundIter
from remora.data_chunks import (
    RemoraRead,
    RemoraDataset,
    compute_ref_to_signal,
)
from remora.io import (
    index_bam,
    iter_signal,
    extract_signal,
    iter_alignments,
    read_is_primary,
    extract_alignments,
    prep_extract_signal,
    prep_extract_alignments,
)

LOGGER = log.get_logger()


####################
# Chunk extraction #
####################


def extract_chunks(
    read_errs,
    int_label,
    motifs,
    focus_ref_pos,
    sig_map_refiner,
    max_chunks_per_read,
    chunk_context,
    kmer_context_bases,
    base_pred,
    base_start_justify,
    offset,
    basecall_anchored,
):
    read_chunks = []
    for read_idx, (io_read, err) in enumerate(read_errs):
        if io_read is None:
            read_chunks.append(tuple((io_read, err)))
            continue
        if io_read.ref_seq is None:
            read_chunks.append(
                tuple((None, "No reference sequence (missing MD tag)"))
            )
            continue
        if basecall_anchored:
            remora_read = io_read.into_remora_read(use_reference_anchor=False)
            remora_read.focus_bases = (
                io_read.get_base_call_anchored_focus_bases(
                    motifs=motifs,
                    select_focus_reference_positions=focus_ref_pos,
                )
            )
            remora_read.labels = np.full(len(io_read.seq), int_label, dtype=int)
        else:
            io_read.ref_to_signal = compute_ref_to_signal(
                io_read.query_to_signal,
                io_read.cigar,
                query_seq=io_read.seq,
                ref_seq=io_read.ref_seq,
            )
            trim_signal = io_read.signal[
                io_read.ref_to_signal[0] : io_read.ref_to_signal[-1]
            ]
            shift_ref_to_sig = io_read.ref_to_signal - io_read.ref_to_signal[0]
            remora_read = RemoraRead(
                dacs=trim_signal,
                shift=io_read.shift_dacs_to_norm,
                scale=io_read.scale_dacs_to_norm,
                seq_to_sig_map=shift_ref_to_sig,
                str_seq=io_read.ref_seq,
                labels=np.full(len(io_read.ref_seq), int_label, dtype=int),
                read_id=io_read.read_id,
            )
            if focus_ref_pos is not None:
                # todo(arand) make a test that exercises this code path
                remora_read.focus_bases = io_read.get_filtered_focus_positions(
                    focus_ref_pos
                )
            else:
                remora_read.set_motif_focus_bases(motifs)

        remora_read.refine_signal_mapping(sig_map_refiner)
        remora_read.downsample_focus_bases(max_chunks_per_read)
        try:
            remora_read.check()
        except RemoraError as e:
            LOGGER.debug(f"Read prep failed: {e}")
            continue
        read_align_chunks = list(
            remora_read.iter_chunks(
                chunk_context,
                kmer_context_bases,
                base_pred,
                base_start_justify,
                offset,
                check_chunks=True,
            )
        )
        LOGGER.debug(
            f"extracted {len(read_align_chunks)} chunks from {io_read.read_id} "
            f"alignment {read_idx}"
        )
        read_chunks.append(tuple((read_align_chunks, None)))

    return read_chunks


############
# Pipeline #
############


def extract_chunk_dataset(
    bam_fn,
    pod5_path,
    out_fn,
    mod_base,
    mod_base_control,
    motifs,
    focus_ref_pos,
    chunk_context,
    min_samps_per_base,
    max_chunks_per_read,
    sig_map_refiner,
    base_pred,
    kmer_context_bases,
    base_start_justify,
    offset,
    num_reads,
    num_extract_alignment_threads,
    num_extract_chunks_threads,
    signal_first=True,
    skip_non_primary=True,
    base_call_anchor=False,
):
    if signal_first:
        bam_idx, num_bam_reads = index_bam(bam_fn, skip_non_primary)
        with pod5.Reader(Path(pod5_path)) as pod5_fp:
            num_pod5_reads = sum(1 for _ in pod5_fp.reads())
            LOGGER.info(
                f"Found {num_bam_reads} BAM records and "
                f"{num_pod5_reads} POD5 reads"
            )
            if num_reads is None:
                num_reads = min(num_pod5_reads, num_bam_reads)
            else:
                num_reads = min(num_reads, num_pod5_reads, num_bam_reads)
    else:
        LOGGER.info("Counting reads in BAM file")
        pysam_save = pysam.set_verbosity(0)
        with pysam.AlignmentFile(bam_fn, mode="rb", check_sq=False) as bam_fp:
            num_bam_reads = bam_fp.count(
                until_eof=True, read_callback=read_is_primary
            )
            LOGGER.info(f"Found {num_bam_reads} BAM records")
            if num_reads is None:
                num_reads = num_bam_reads
            else:
                num_reads = min(num_reads, num_bam_reads)
        pysam.set_verbosity(pysam_save)

    LOGGER.info(
        f"Making {'base call' if base_call_anchor else 'reference'}-"
        f"anchored training data"
    )
    LOGGER.info("Allocating memory for output tensors")
    # initialize empty dataset with pre-allocated memory
    dataset = RemoraDataset.allocate_empty_chunks(
        num_chunks=max_chunks_per_read * num_reads,
        chunk_context=chunk_context,
        kmer_context_bases=kmer_context_bases,
        min_samps_per_base=min_samps_per_base,
        base_pred=base_pred,
        mod_bases=[] if mod_base_control else [mod_base[0]],
        mod_long_names=[] if mod_base_control else [mod_base[1]],
        motifs=[mot.to_tuple() for mot in motifs],
        sig_map_refiner=sig_map_refiner,
        base_start_justify=base_start_justify,
        offset=offset,
    )

    assert len(bam_idx.keys()) > 0

    LOGGER.info("Processing reads")
    if signal_first:
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
            num_workers=num_extract_alignment_threads,
            args=(bam_idx, bam_fn),
            name="AddAlignments",
            use_process=True,
        )
    else:
        mappings = BackgroundIter(
            iter_alignments,
            args=(bam_fn, num_reads, skip_non_primary),
            name="ExtractMappings",
            use_process=True,
        )
        reads = MultitaskMap(
            extract_signal,
            mappings,
            prep_func=prep_extract_signal,
            num_workers=num_extract_alignment_threads,
            args=(pod5_path,),
            name="AddSignal",
            use_process=True,
        )
    chunks = MultitaskMap(
        extract_chunks,
        reads,
        num_workers=num_extract_chunks_threads,
        args=[
            0 if mod_base_control else 1,
            motifs,
            focus_ref_pos,
            sig_map_refiner,
            max_chunks_per_read,
            chunk_context,
            kmer_context_bases,
            base_pred,
            base_start_justify,
            offset,
            base_call_anchor,
        ],
        name="ExtractChunks",
        use_process=True,
    )

    errs = defaultdict(int)
    for read_chunks in tqdm(
        chunks,
        total=num_reads,
        smoothing=0,
        unit=" Reads",
        desc="Extracting chunks",
    ):
        if len(read_chunks) == 0:
            errs["No chunks extracted"] += 1
            continue
        for read_align_chunks, err in read_chunks:
            if read_align_chunks is None:
                errs[err] += 1
                continue
            for chunk in read_align_chunks:
                try:
                    dataset.add_chunk(chunk)
                except RemoraError as e:
                    errs[str(e)] += 1

    if len(errs) > 0:
        err_types = sorted([(num, err) for err, num in errs.items()])[::-1]
        err_str = "\n".join(f"{num:>7} : {err:<80}" for num, err in err_types)
        LOGGER.info(f"Unsuccessful read/chunk reasons:\n{err_str}")

    dataset.clip_chunks()
    dataset.shuffle()
    dataset.save(out_fn)

    LOGGER.info(f"Extracted {dataset.nchunks} chunks from {num_reads} reads.")
    LOGGER.info(f"Label distribution: {dataset.get_label_counts()}")
