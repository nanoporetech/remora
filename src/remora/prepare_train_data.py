import os
import sys
from pathlib import Path
from collections import defaultdict

import pod5
import numpy as np
from tqdm import tqdm

from remora import log, RemoraError, constants
from remora.util import MultitaskMap, BackgroundIter
from remora.io import (
    ReadIndexedBam,
    get_read_ids,
    iter_signal,
    extract_alignments,
)
from remora.data_chunks import (
    DatasetMetadata,
    RemoraRead,
    CoreRemoraDataset,
    compute_ref_to_signal,
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
    base_start_justify,
    offset,
    basecall_anchor,
):
    read_chunks = []
    for read_idx, (io_read, err) in enumerate(read_errs):
        if err is not None:
            read_chunks.append((None, err))
            continue
        if not io_read.is_mapped:
            read_chunks.append(((None, "Unmapped read")))
            continue
        if io_read.ref_seq is None:
            read_chunks.append(
                ((None, "No reference sequence (unmapped or missing MD tag)"))
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
                read_metrics=io_read.read_metrics,
            )
            if focus_ref_pos is None:
                remora_read.set_motif_focus_bases(motifs)
            else:
                # todo(arand) make a test that exercises this code path
                remora_read.focus_bases = io_read.get_filtered_focus_positions(
                    focus_ref_pos
                )

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
                base_start_justify,
                offset,
                check_chunks=True,
                motifs=motifs,
            )
        )
        LOGGER.debug(
            f"extracted {len(read_align_chunks)} chunks from {io_read.read_id} "
            f"alignment {read_idx}"
        )
        read_chunks.append((read_align_chunks, None))

    return read_chunks


############
# Pipeline #
############


def extract_chunk_dataset(
    bam_path,
    pod5_path,
    out_path,
    mod_base,
    mod_base_control,
    motifs,
    focus_ref_pos,
    chunk_context,
    min_samps_per_base,
    max_chunks_per_read,
    pa_scaling,
    sig_map_refiner,
    kmer_context_bases,
    base_start_justify,
    offset,
    num_reads,
    num_extract_alignment_threads,
    num_extract_chunks_threads,
    skip_non_primary=True,
    basecall_anchor=False,
    rev_sig=False,
    save_every=100_000,
    skip_shuffle=False,
):
    bam_idx = ReadIndexedBam(bam_path, skip_non_primary)
    if bam_idx.num_records == 0:
        LOGGER.info("No records found in BAM file.")
        sys.exit()
    with pod5.DatasetReader(Path(pod5_path)) as pod5_dr:
        read_ids, num_reads = get_read_ids(
            bam_idx, pod5_dr, num_reads, return_num_bam_reads=True
        )
    if num_reads == 0:
        return

    LOGGER.info(
        f"Making {'basecall' if basecall_anchor else 'reference'}-"
        f"anchored training data"
    )
    LOGGER.info("Opening dataset for output")
    max_seq_len = sum(chunk_context) // min_samps_per_base
    LOGGER.debug(f"Maximum chunk sequence length set to {max_seq_len}")
    extra_metadata_arrays = {
        "modbase_label": ("int64", "Modified base label"),
        "read_id": ("<U36", "Read identifier"),
        "read_focus_base": (
            "int64",
            "Position within read training sequence",
        ),
        "percent_identity": (
            "float32",
            "Reference mapping percent identity",
        ),
        "start_time": (
            "uint32",
            "Read start time in seconds since first read in dataset",
        ),
        "duration": ("int64", "Number of samples in trimmed read"),
    }
    dataset = CoreRemoraDataset(
        data_path=out_path,
        mode="w",
        metadata=DatasetMetadata(
            allocate_size=max_chunks_per_read * num_reads,
            max_seq_len=max_seq_len,
            mod_bases=[] if mod_base_control else [mod_base[0]],
            mod_long_names=[] if mod_base_control else [mod_base[1]],
            motif_sequences=[motif.raw_motif for motif in motifs],
            motif_offsets=[motif.focus_pos for motif in motifs],
            extra_metadata_arrays=extra_metadata_arrays,
            chunk_context=chunk_context,
            kmer_context_bases=kmer_context_bases,
            reverse_signal=rev_sig,
            pa_scaling=pa_scaling,
            sig_map_refiner=sig_map_refiner,
            base_start_justify=base_start_justify,
            offset=offset,
        ),
        return_arrays=list(extra_metadata_arrays),
    )

    LOGGER.info("Processing reads")
    signals = BackgroundIter(
        iter_signal,
        args=(pod5_path,),
        kwargs={
            "num_reads": num_reads,
            "read_ids": read_ids,
            "rev_sig": rev_sig,
            "pa_scaling": pa_scaling,
        },
        name="ExtractSignal",
        use_process=True,
        q_maxsize=1000,
    )
    reads = MultitaskMap(
        extract_alignments,
        signals,
        num_workers=num_extract_alignment_threads,
        args=(bam_idx, rev_sig),
        name="AddAlignments",
        use_process=True,
        q_maxsize=1000,
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
            base_start_justify,
            offset,
            basecall_anchor,
        ],
        name="ExtractChunks",
        use_process=True,
        q_maxsize=1000,
    )

    errs = defaultdict(int)
    for read_chunks in tqdm(
        chunks,
        total=len(read_ids),
        smoothing=0,
        dynamic_ncols=True,
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
                if chunk.seq_len > max_seq_len:
                    errs["Sequence too long"] += 1
                    continue
                try:
                    dataset.write_chunk(chunk)
                    if dataset.size % save_every == 0:
                        dataset.flush()
                        dataset.write_metadata()
                except RemoraError as e:
                    errs[str(e)] += 1

    if len(errs) > 0:
        err_types = sorted([(num, err) for err, num in errs.items()])[::-1]
        err_str = "\n".join(f"{num:>7,} : {err:<80}" for num, err in err_types)
        LOGGER.info(f"Unsuccessful read/chunk reasons:\n{err_str}")

    dataset.write_metadata()
    LOGGER.info(f"Extracted {dataset.size:,} chunks from {num_reads:,} reads.")
    LOGGER.info(f"Label distribution: {dataset.modbase_label_summary}")
    if not skip_shuffle:
        LOGGER.info("Shuffling dataset")
        dataset.shuffle()


#############################
# Basecall chunk extraction #
#############################


def extract_basecall_chunks(
    read_errs,
    sig_map_refiner,
    max_chunks_per_read,
    chunk_context,
    kmer_context_bases,
    basecall_anchor,
):
    read_chunks = []
    for read_idx, (io_read, err) in enumerate(read_errs):
        if err is not None:
            read_chunks.append((None, err))
            continue
        if io_read.ref_seq is None:
            read_chunks.append(
                ((None, "No reference sequence (unmapped or missing MD tag)"))
            )
            continue
        if basecall_anchor:
            remora_read = io_read.into_remora_read(use_reference_anchor=False)
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
                read_id=io_read.read_id,
                read_metrics=io_read.read_metrics,
            )

        remora_read.refine_signal_mapping(sig_map_refiner)
        remora_read.downsample_focus_bases(max_chunks_per_read)
        try:
            remora_read.check()
        except RemoraError as e:
            LOGGER.debug(f"Read prep failed: {e}")
            continue
        read_align_chunks = list(
            remora_read.iter_basecall_chunks(
                chunk_context,
                kmer_context_bases,
                max_chunks_per_read,
                check_chunks=True,
            )
        )
        LOGGER.debug(
            f"extracted {len(read_align_chunks)} chunks from {io_read.read_id} "
            f"alignment {read_idx}"
        )
        read_chunks.append((read_align_chunks, None))

    return read_chunks


#####################
# Basecall pipeline #
#####################


def extract_basecall_chunk_dataset(
    bam_path,
    pod5_path,
    out_path,
    chunk_context,
    min_samps_per_base,
    max_chunks_per_read,
    pa_scaling,
    sig_map_refiner,
    kmer_context_bases,
    num_reads,
    num_extract_alignment_threads,
    num_extract_chunks_threads,
    skip_non_primary=True,
    basecall_anchor=False,
    rev_sig=False,
    save_every=100_000,
    skip_shuffle=False,
    shuffle_batch_size=200_000,
):
    bam_idx = ReadIndexedBam(bam_path, skip_non_primary)
    if bam_idx.num_records == 0:
        LOGGER.info("No records found in BAM file.")
        sys.exit()
    with pod5.DatasetReader(Path(pod5_path)) as pod5_dr:
        read_ids, num_reads = get_read_ids(
            bam_idx, pod5_dr, num_reads, return_num_bam_reads=True
        )
    if num_reads == 0:
        return

    LOGGER.info(
        f"Making {'basecall' if basecall_anchor else 'reference'}-"
        f"anchored basecaller training data"
    )
    LOGGER.info("Opening dataset for output")
    max_seq_len = sum(chunk_context) // min_samps_per_base
    LOGGER.debug(f"Maximum chunk sequence length set to {max_seq_len}")
    extra_metadata_arrays = {
        "read_id": ("<U36", "Read identifier"),
        "percent_identity": (
            "float32",
            "Reference mapping percent identity",
        ),
        "start_time": (
            "uint32",
            "Read start time in seconds since first read in dataset",
        ),
        "duration": ("int64", "Number of samples in trimmed read"),
    }
    # to add more extra metadata values see io.READ_METRICS
    dataset = CoreRemoraDataset(
        data_path=out_path,
        mode="w",
        metadata=DatasetMetadata(
            allocate_size=max_chunks_per_read * num_reads,
            max_seq_len=max_seq_len,
            dataset_type=constants.DATASET_TYPE_SEQ,
            extra_metadata_arrays=extra_metadata_arrays,
            chunk_context=chunk_context,
            kmer_context_bases=kmer_context_bases,
            reverse_signal=rev_sig,
            pa_scaling=pa_scaling,
            sig_map_refiner=sig_map_refiner,
        ),
        return_arrays=list(extra_metadata_arrays),
    )

    LOGGER.info("Processing reads")
    signals = BackgroundIter(
        iter_signal,
        args=(pod5_path,),
        kwargs={
            "num_reads": num_reads,
            "read_ids": read_ids,
            "rev_sig": rev_sig,
            "pa_scaling": pa_scaling,
        },
        name="ExtractSignal",
        use_process=True,
        q_maxsize=1000,
    )
    reads = MultitaskMap(
        extract_alignments,
        signals,
        num_workers=num_extract_alignment_threads,
        args=(bam_idx, rev_sig),
        name="AddAlignments",
        use_process=True,
        q_maxsize=1000,
    )
    chunks = MultitaskMap(
        extract_basecall_chunks,
        reads,
        num_workers=num_extract_chunks_threads,
        args=[
            sig_map_refiner,
            max_chunks_per_read,
            chunk_context,
            kmer_context_bases,
            basecall_anchor,
        ],
        name="ExtractChunks",
        use_process=True,
        q_maxsize=1000,
    )

    errs = defaultdict(int)
    for read_chunks in tqdm(
        chunks,
        total=len(read_ids),
        smoothing=0,
        dynamic_ncols=True,
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
                if chunk.seq_len > max_seq_len:
                    errs["Sequence too long"] += 1
                    continue
                try:
                    dataset.write_chunk(chunk)
                    if dataset.size % save_every == 0:
                        dataset.flush()
                        dataset.write_metadata()
                except RemoraError as e:
                    errs[str(e)] += 1

    # manually shift start time array to difference from dataset min
    dataset.start_time[
        dataset.metadata.dataset_start : dataset.metadata.dataset_end
    ] = dataset.start_time[
        dataset.metadata.dataset_start : dataset.metadata.dataset_end
    ].min()

    if len(errs) > 0:
        err_types = sorted([(num, err) for err, num in errs.items()])[::-1]
        err_str = "\n".join(f"{num:>7,} : {err:<80}" for num, err in err_types)
        LOGGER.info(f"Unsuccessful read/chunk reasons:\n{err_str}")

    dataset.write_metadata()
    LOGGER.info(f"Extracted {dataset.size:,} chunks from {num_reads:,} reads.")
    if not skip_shuffle:
        LOGGER.info("Shuffling dataset")
        dataset.shuffle(shuffle_batch_size)
