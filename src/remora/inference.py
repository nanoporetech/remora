import atexit
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from remora import constants, log, RemoraError, encoded_kmers
from remora.data_chunks import RemoraDataset, RemoraRead
from remora.util import (
    format_mm_ml_tags,
    get_can_converter,
    softmax_axis1,
    Motif,
    validate_mod_bases,
)
from remora.model_util import load_model

LOGGER = log.get_logger()


class resultsWriter:
    def __init__(self, output_path):
        self.sep = "\t"
        self.out_fp = open(output_path, "w")
        df = pd.DataFrame(
            columns=[
                "read_id",
                "read_pos",
                "label",
                "class_pred",
                "class_probs",
            ]
        )
        df.to_csv(self.out_fp, sep=self.sep, index=False)

    def write_results(self, output, labels, read_pos, read_id):
        class_preds = output.argmax(axis=1)
        str_probs = [",".join(map(str, r)) for r in softmax_axis1(output)]
        pd.DataFrame(
            {
                "read_id": read_id,
                "read_pos": read_pos,
                "label": labels,
                "class_pred": class_preds,
                "class_probs": str_probs,
            }
        ).to_csv(self.out_fp, header=False, index=False, sep=self.sep)

    def close(self):
        self.out_fp.close()


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
    read.refine_signal_mapping(model_metadata["sig_map_refiner"])
    read_outputs, all_read_data, read_labels = [], [], []
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
        store_read_data=True,
        batch_size=batch_size,
        shuffle_on_iter=False,
        drop_last=False,
    )
    for chunk in chunks:
        dataset.add_chunk(chunk)
    dataset.set_nbatches()
    for (sigs, seqs, seq_maps, seq_lens), labels, read_data in dataset:
        enc_kmers = encoded_kmers.compute_encoded_kmer_batch(
            bb, ab, seqs, seq_maps, seq_lens
        )
        read_outputs.append(model.run([], {"sig": sigs, "seq": enc_kmers})[0])
        read_labels.append(labels)
        all_read_data.extend(read_data)
    read_outputs = np.concatenate(read_outputs, axis=0)
    read_labels = np.concatenate(read_labels)
    return read_outputs, read_labels, list(zip(*all_read_data))[1]


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


def infer(
    input_msf,
    out_path,
    onnx_model_path,
    batch_size,
    device,
    focus_offset,
    pore,
    basecall_model_type,
    basecall_model_version,
    modified_bases,
    remora_model_type,
    remora_model_version,
):
    LOGGER.info("Performing Remora inference")
    alphabet_info = input_msf.get_alphabet_information()
    alphabet, collapse_alphabet = (
        alphabet_info.alphabet,
        alphabet_info.collapse_alphabet,
    )

    if focus_offset is not None:
        focus_offset = np.array([focus_offset])

    rw = resultsWriter(os.path.join(out_path, "results.tsv"))
    atexit.register(rw.close)

    LOGGER.info("Loading model")
    model, model_metadata = load_model(
        onnx_model_path,
        pore=pore,
        basecall_model_type=basecall_model_type,
        basecall_model_version=basecall_model_version,
        modified_bases=modified_bases,
        remora_model_type=remora_model_type,
        remora_model_version=remora_model_version,
        device=device,
    )

    if model_metadata["base_pred"]:
        if alphabet != "ACGT":
            raise ValueError(
                "Base prediction is not compatible with modified base "
                "training data. It requires a canonical alphabet."
            )
        label_conv = get_can_converter(alphabet, collapse_alphabet)
    else:
        try:
            motifs = [Motif(*mot) for mot in model_metadata["motifs"]]
            label_conv = validate_mod_bases(
                model_metadata["mod_bases"], motifs, alphabet, collapse_alphabet
            )
        except RemoraError:
            label_conv = None

    can_conv = get_can_converter(
        alphabet_info.alphabet, alphabet_info.collapse_alphabet
    )
    num_reads = len(input_msf.get_read_ids())
    for read in tqdm(input_msf, smoothing=0, total=num_reads, unit="reads"):
        try:
            read = RemoraRead.from_taiyaki_read(read, can_conv, label_conv)
        except RemoraError:
            # TODO log these failed reads to track down errors
            continue
        output, labels, read_pos = call_read_mods(
            read,
            model,
            model_metadata,
            batch_size=batch_size,
            focus_offset=focus_offset,
        )
        rw.write_results(output, labels, read_pos, read.read_id)


if __name__ == "__main__":
    NotImplementedError("This is a module.")
