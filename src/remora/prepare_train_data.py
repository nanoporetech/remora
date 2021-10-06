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
    motif = Motif(*dataset.motif)
    for read in tqdm(input_msf, smoothing=0, total=num_reads, unit="reads"):
        try:
            read = RemoraRead.from_taiyaki_read(read, can_conv, label_conv)
        except RemoraError:
            # TODO log these failed reads to track down errors
            continue
        if motif.any_context:
            motif_hits = np.arange(
                motif.focus_pos,
                read.can_seq.size - motif.num_bases_after_focus,
            )
        else:
            motif_hits = np.fromiter(read.iter_motif_hits(motif), int)
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
            if chunk.seq_len <= dataset.max_seq_len:
                dataset.add_chunk(chunk)
    dataset.clip_chunks()


def extract_chunk_dataset(
    input_msf,
    output_filename,
    motif,
    chunk_context,
    max_seq_len,
    max_chunks_per_read,
    label_conv,
    base_pred,
    mod_bases,
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
        max_seq_len=max_seq_len,
        kmer_context_bases=kmer_context_bases,
        base_pred=base_pred,
        mod_bases=mod_bases,
        motif=motif.to_tuple(),
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