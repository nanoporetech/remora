import os
from pathlib import Path
from collections import defaultdict

import pod5
import numpy as np
from tqdm import tqdm

from remora import log, RemoraError
from remora.util import MultitaskMap, BackgroundIter, get_read_ids
from remora.io import ReadIndexedBam, iter_signal, extract_alignments
from remora.data_chunks import RemoraRead, RemoraDataset, compute_ref_to_signal

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
    basecall_anchor,
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
        if basecall_anchor:
            remora_read = io_read.into_remora_read(use_reference_anchor=False)
            remora_read.focus_bases = io_read.get_basecall_anchored_focus_bases(
                motifs=motifs,
                select_focus_reference_positions=focus_ref_pos,
            )
            remora_read.labels = np.full(len(io_read.seq), int_label, dtype=int)
        else:
            io_read.ref_to_signal = compute_ref_to_signal(
                io_read.query_to_signal,
                io_read.cigar,
            )
            assert io_read.ref_to_signal.size == len(io_read.ref_seq) + 1, (
                "discordant ref seq lengths: move+cigar:"
                f"{io_read.ref_to_signal.size} ref_seq:{len(io_read.ref_seq)}"
            )
            trim_dacs = io_read.dacs[
                io_read.ref_to_signal[0] : io_read.ref_to_signal[-1]
            ]
            shift_ref_to_sig = io_read.ref_to_signal - io_read.ref_to_signal[0]
            remora_read = RemoraRead(
                dacs=trim_dacs,
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
    skip_non_primary=True,
    basecall_anchor=False,
    rev_sig=False,
):
    bam_idx = ReadIndexedBam(bam_fn, skip_non_primary)
    with pod5.Reader(Path(pod5_path)) as pod5_fh:
        read_ids, num_reads = get_read_ids(bam_idx, pod5_fh, num_reads)
    if num_reads == 0:
        return

    LOGGER.info(
        f"Making {'basecall' if basecall_anchor else 'reference'}-"
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
        reverse_signal=rev_sig,
        sig_map_refiner=sig_map_refiner,
        base_start_justify=base_start_justify,
        offset=offset,
    )

    LOGGER.info("Processing reads")
    signals = BackgroundIter(
        iter_signal,
        args=(pod5_path,),
        kwargs={
            "num_reads": num_reads,
            "read_ids": read_ids,
            "rev_sig": rev_sig,
        },
        name="ExtractSignal",
        use_process=True,
    )
    reads = MultitaskMap(
        extract_alignments,
        signals,
        num_workers=num_extract_alignment_threads,
        args=(bam_idx, rev_sig),
        name="AddAlignments",
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
            basecall_anchor,
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
        disable=os.environ.get("LOG_SAFE", False),
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
