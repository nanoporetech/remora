from pathlib import Path
from collections import defaultdict

import pysam
import pod5_format
import numpy as np
from tqdm import tqdm

from remora import log, RemoraError
from remora.util import MultitaskMap, BackgroundIter
from remora.data_chunks import RemoraDataset, RemoraRead, compute_ref_to_signal
from remora.io import (
    index_bam,
    read_is_primary,
    iter_signal,
    prep_extract_alignments,
    extract_alignments,
    iter_alignments,
    prep_extract_signal,
    extract_signal,
)

LOGGER = log.get_logger()


####################
# Chunk extraction #
####################


def add_focus_ref_pos(read, focus_ref_pos, ref_pos):
    try:
        cs_focus_pos = focus_ref_pos[(ref_pos.ctg, ref_pos.strand)]
    except KeyError:
        # no focus positions on contig/strand
        return
    read_len = read.int_seq.size
    read_focus_ref_pos = np.array(
        sorted(
            set(range(ref_pos.start, ref_pos.start + read_len)).intersection(
                cs_focus_pos
            )
        )
    )
    if read_focus_ref_pos.size == 0:
        return
    read.focus_bases = (
        read_focus_ref_pos - ref_pos.start
        if ref_pos.strand == "+"
        else ref_pos.start + read_len - read_focus_ref_pos[::-1] - 1
    )


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
):
    read_chunks = []
    for read_idx, (read, err) in enumerate(read_errs):
        if read is None:
            read_chunks.append(tuple((read, err)))
            continue
        if read.ref_seq is None:
            read_chunks.append(
                tuple((None, "No reference sequence (missing MD tag)"))
            )
            continue
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
            labels=np.full(len(read.ref_seq), int_label, dtype=int),
            read_id=read.read_id,
        )
        if focus_ref_pos is not None:
            add_focus_ref_pos(remora_read, focus_ref_pos, read.ref_pos)
        else:
            remora_read.add_motif_focus_bases(motifs)
        remora_read.refine_signal_mapping(sig_map_refiner)
        remora_read.downsample_focus_bases(max_chunks_per_read)
        read_align_chunks = list(
            remora_read.iter_chunks(
                chunk_context,
                kmer_context_bases,
                base_pred,
                base_start_justify,
                offset,
            )
        )
        LOGGER.debug(
            f"extracted {len(read_align_chunks)} chunks from {read.read_id} "
            f"alignment {read_idx}"
        )
        read_chunks.append(tuple((read_align_chunks, None)))
    return read_chunks


############
# Pipeline #
############


def extract_chunk_dataset(
    bam_fn,
    pod5_fn,
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
):
    if signal_first:
        bam_idx, num_bam_reads = index_bam(bam_fn, skip_non_primary)
        with pod5_format.CombinedReader(Path(pod5_fn)) as pod5_fp:
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

    LOGGER.info("Processing reads")
    if signal_first:
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
            args=(pod5_fn,),
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
