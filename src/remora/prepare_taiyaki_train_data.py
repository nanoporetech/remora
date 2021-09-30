from collections import Counter

import numpy as np
from tqdm import tqdm

from remora import log
from remora.util import to_str

LOGGER = log.get_logger()


def extract_chunk_dataset(
    input_msf,
    output_msf,
    motif,
    context_bases,
    max_chunks_per_read,
):
    alphabet_info = input_msf.get_alphabet_information()

    LOGGER.info("Processing reads")
    num_reads = num_chunks = 0
    labels = []
    for read in tqdm(input_msf, smoothing=0):
        seq = "".join(
            (alphabet_info.collapse_alphabet[b] for b in read.Reference)
        )
        if motif.any_context:
            motif_hits = np.arange(
                max(motif.focus_pos, context_bases),
                len(seq)
                - max(motif.num_bases_after_focus - 1, context_bases - 1),
            )
        else:
            motif_hits = np.array(
                [
                    m.start() + motif.focus_pos
                    for m in motif.pattern.finditer(seq)
                ]
            )
            motif_hits = motif_hits[
                np.logical_and(
                    motif_hits > context_bases,
                    motif_hits < read.Reference.size - context_bases - 1,
                )
            ]
        if motif_hits.size == 0:
            continue
        num_reads += 1
        read_dict = read.get_read_dictionary()
        for focus_pos in np.random.choice(
            motif_hits,
            size=min(max_chunks_per_read, motif_hits.size),
            replace=False,
        ):
            chunk_dict = read_dict.copy()
            # trim signal and adjust Ref_to_signal mapping
            ref_st = focus_pos - context_bases
            ref_en = focus_pos + context_bases + 1
            sig_st = read.Ref_to_signal[ref_st]
            sig_en = read.Ref_to_signal[ref_en]
            # remove chunks with more signal than bases
            # TODO add more stringent filtering (maybe wait for
            # on-the-fly-chunk extraction)
            if sig_en - sig_st < ref_en - ref_st:
                continue
            chunk_dict["read_id"] = f"{to_str(read.read_id)}:::pos_{focus_pos}"
            chunk_dict["Dacs"] = read.Dacs[sig_st:sig_en]
            chunk_dict["Ref_to_signal"] = (
                read.Ref_to_signal[ref_st:ref_en] - sig_st
            )
            chunk_dict["Reference"] = read.Reference[
                focus_pos - context_bases : focus_pos + context_bases + 1
            ]
            labels.append(chunk_dict["Reference"][context_bases])
            num_chunks += 1
            output_msf.write_read(chunk_dict)

    LOGGER.info(f"Extracted {num_chunks} chunks from {num_reads} reads.")
    label_counts = Counter(labels)
    LOGGER.info(f"Label distribution: {label_counts}")
