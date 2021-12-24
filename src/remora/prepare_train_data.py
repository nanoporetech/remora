import numpy as np
from tqdm import tqdm

from remora import log, RemoraError
from remora.util import Motif, get_can_converter
from remora.data_chunks import RemoraRead, RemoraDataset

LOGGER = log.get_logger()


def fill_dataset(
    input_msf,
    dataset,
    num_reads,
    can_conv,
    label_conv,
    max_chunks_per_read,
):
    motif = [Motif(*mot) for mot in dataset.motifs]
    num_failed_reads = num_short_chunks = 0
    for read in tqdm(input_msf, smoothing=0, total=num_reads, unit="reads"):
        try:
            read = RemoraRead.from_taiyaki_read(read, can_conv, label_conv)
        except RemoraError:
            num_failed_reads += 1
            continue
        motif_hits = []
        for mot in motif:
            motif_hits.append(
                np.fromiter(read.iter_motif_hits(mot), int) + mot.focus_pos
            )
        motif_hits = np.concatenate(motif_hits)
        focus_base_indices = np.random.choice(
            motif_hits,
            size=min(max_chunks_per_read, motif_hits.size),
            replace=False,
        )
        for chunk in read.iter_chunks(
            focus_base_indices,
            dataset.chunk_context,
            dataset.kmer_context_bases,
            dataset.base_pred,
        ):
            if chunk.seq_len > dataset.max_seq_len:
                num_short_chunks += 1
                LOGGER.debug(
                    f"Short chunk: {read.read_id} {chunk.sig_focus_pos}"
                )
                continue
            dataset.add_chunk(chunk)
    dataset.clip_chunks()
    LOGGER.info(
        f"Processing encountered {num_failed_reads} invalid reads from "
        f"Taiyaki and {num_short_chunks} short chunks which were discarded."
    )


def extract_chunk_dataset(
    input_msf,
    output_filename,
    motif,
    chunk_context,
    min_samps_per_base,
    max_chunks_per_read,
    label_conv,
    base_pred,
    mod_bases,
    mod_long_names,
    kmer_context_bases,
):
    alphabet_info = input_msf.get_alphabet_information()
    can_conv = get_can_converter(
        alphabet_info.alphabet, alphabet_info.collapse_alphabet
    )

    LOGGER.info("Allocating memory for output tensors")
    num_reads = len(input_msf.get_read_ids())
    # initialize empty dataset with pre-allocated memory
    dataset = RemoraDataset.allocate_empty_chunks(
        num_chunks=max_chunks_per_read * num_reads,
        chunk_context=chunk_context,
        min_samps_per_base=min_samps_per_base,
        kmer_context_bases=kmer_context_bases,
        base_pred=base_pred,
        mod_bases=mod_bases,
        mod_long_names=mod_long_names,
        motifs=[mot.to_tuple() for mot in motif],
    )
    LOGGER.info("Processing reads")
    fill_dataset(
        input_msf,
        dataset,
        num_reads,
        can_conv,
        label_conv,
        max_chunks_per_read,
    )
    dataset.shuffle_dataset()
    dataset.save_dataset(output_filename)

    LOGGER.info(f"Extracted {dataset.nchunks} chunks from {num_reads} reads.")
    LOGGER.info(f"Label distribution: {dataset.get_label_counts()}")
