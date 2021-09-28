from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from taiyaki.mapped_signal_files import MappedSignalReader
import torch
import torch.nn.utils.rnn as rnn

from remora import constants, log, RemoraError

LOGGER = log.get_logger()

DEFAULT_BATCH_SIZE = 1024

BASE_ENCODINGS = {
    "A": np.array([1, 0, 0, 0]),
    "C": np.array([0, 1, 0, 0]),
    "G": np.array([0, 0, 1, 0]),
    "T": np.array([0, 0, 0, 1]),
    "N": np.array([0, 0, 0, 0]),
}


def get_motif_pos(ref, motif, motif_offset=0):
    return (
        np.where(
            np.all(
                np.stack(
                    [
                        motif[offset]
                        == ref[offset : ref.size - motif.size + offset + 1]
                        for offset in range(motif.size)
                    ]
                ),
                axis=0,
            )
        )[0]
        + motif_offset
    )


def validate_motif(input_msf, motif):
    mod_base, can_motif, motif_offset = motif
    try:
        motif_offset = int(motif_offset)
    except ValueError:
        raise RemoraError(f'Motif offset not an integer: "{motif_offset}"')
    if motif_offset >= len(motif):
        raise RemoraError("Motif offset is past the end of the motif")
    alphabet_info = input_msf.get_alphabet_information()
    if mod_base not in alphabet_info.alphabet:
        raise RemoraError("Modified base provided not found in alphabet")
    if any(b not in alphabet_info.alphabet for b in can_motif):
        raise RemoraError(
            "Base(s) in motif provided not found in alphabet "
            f'"{set(can_motif).difference(alphabet_info.alphabet)}"'
        )
    can_base = can_motif[motif_offset]
    mod_can_equiv = alphabet_info.collapse_alphabet[
        alphabet_info.alphabet.find(mod_base)
    ]
    if can_base != mod_can_equiv:
        raise RemoraError(
            f"Canonical base within motif ({can_base}) does not match "
            f"canonical equivalent for modified base ({mod_can_equiv})"
        )
    int_can_motif = np.array(
        [alphabet_info.alphabet.find(b) for b in can_motif]
    )

    return mod_base, int_can_motif, motif_offset


@dataclass
class Chunk:
    """Chunk of signal and associated sequence for training/infering results in
    Remora. Chunks are selected either with a given signal or bases location
    from a read.

    Args:
        signal (np.array): Normalized signal
        sequence (str): Sequence assigned to chunk
        seq_to_sig_map (np.array): Array of length one more than sequence with
            values representing indices into signal for each base.
        before_context_seq (str): Context sequence for computing seq_nn_input
        after_context_seq (str): Context sequence for computing seq_nn_input
        sig_focus_pos (int): Index within signal array on which the chunk is
            focuesed for prediction. May be used in model architecture in the
            future.
        label (int): Integer label for training/validation.
        read_id (str): Read ID
        read_seq_pos (int): Position within read for validation purposes
    """

    signal: np.ndarray
    sequence: str
    seq_to_sig_map: np.ndarray
    before_context_seq: str
    after_context_seq: str
    sig_focus_pos: int
    label: int
    read_id: str
    read_seq_pos: int
    _seq_nn_input: np.ndarray = None

    def __post_init__(self):
        # compute seq nn input to raise any potential errors at chunk
        # generation time. instead of access time
        self.seq_nn_input

    @classmethod
    def extract_chunk_from_read(
        cls,
        read_sig,
        read_seq,
        seq_to_sig_map,
        kmer_context_bases,
        sig_start,
        sig_end,
        label=None,
        seq_start=None,
        seq_end=None,
        sig_focus_pos=None,
        read_id=None,
        read_seq_pos=None,
    ):
        """Extract a chunk of data from relevant read information given signal
        start and end coordinate.

        Args:
            read_sig (np.array): Normalized signal for the portion mapped to
                sequence
            read_seq (str): Sequence for read (either reference or basecalled
                sequence)
            seq_to_sig_map (np.array): Array of length of read_seq + 1 with
                values representing indices into read_sig.
            kmer_context_bases (tuple): 2-tuple containing number of context
                bases before and after to include in returned chunk.
            sig_start (int): Start index in read_sig array
            sig_end (int): End index (via python slice) in read_sig array
            label (int): Chunk output label for training/validation
            seq_start (int): Start index in read_seq array
            seq_end (int): End index (via python slice) in read_seq array
            sig_focus_pos (int): Focus index within signal array. May be used
                by network architecture.
            read_id (str): Read ID
            read_seq_pos (int): Position within read (for validation).
        """

        if seq_start is None or seq_end is None:
            seq_start = (
                np.searchsorted(seq_to_sig_map, sig_start, side="right") - 1
            )
            seq_end = np.searchsorted(seq_to_sig_map, sig_end, side="left")

        chunk_seq_to_sig = seq_to_sig_map[seq_start : seq_end + 1]
        chunk_seq_to_sig[0] = sig_start
        chunk_seq_to_sig[-1] = sig_end
        chunk_seq_to_sig -= sig_start

        kmer_before_bases, kmer_after_bases = kmer_context_bases
        before_context_seq = read_seq[seq_start - kmer_before_bases : seq_start]
        after_context_seq = read_seq[seq_end : seq_end + kmer_after_bases]

        return cls(
            read_sig[sig_start:sig_end],
            read_seq[seq_start:seq_end],
            chunk_seq_to_sig,
            before_context_seq,
            after_context_seq,
            sig_focus_pos - sig_start,
            label,
            read_id,
            read_seq_pos,
        )

    @property
    def seq_nn_input(self):
        """Representation of chunk sequence presented to neural network. The
        length of the second dimension should match chunk.signal.shape[0]. The
        first dimension represents the number of input features.
        """
        if self._seq_nn_input is None:
            seq_w_context = (
                self.before_context_seq + self.sequence + self.after_context_seq
            )
            kmer_len = len(seq_w_context) - len(self.sequence) + 1
            enc_seq_w_context = np.array(
                [BASE_ENCODINGS[b] for b in seq_w_context]
            )
            enc_kmers = np.stack(
                [
                    enc_seq_w_context[offset : len(self.sequence) + offset]
                    for offset in range(kmer_len)
                ],
                axis=1,
            ).reshape(-1, kmer_len * 4)
            self._seq_nn_input = np.repeat(
                enc_kmers, np.diff(self.seq_to_sig_map), axis=0
            ).T
        return self._seq_nn_input

    @property
    def accepted(self):
        # TODO add other filters for failed chunks
        return self.signal.shape[0] > 0


def load_chunks(
    dataset_path,
    num_chunks,
    mod_motif,
    chunk_context,
    kmer_context_bases,
    focus_offset=None,
    fixed_seq_len_chunks=False,
    base_pred=False,
    full=False,
):
    """
    Args:
        dataset_path: HDF5 file generated by `remora prepare_taiyaki_train_data`
        num_chunks: Total maximum number of chunks to return
        mod_motif: 3-tuple representing modified base motif. (mod_base,
            can_motif, mod_offset)
        chunk_context: 2-tuple containing context signal or bases for each
            chunk
        kmer_context_bases: Number of bases before and after to included in
            encoded reference
        focus_offset: index of (mod)base in reference
        fixed_seq_len_chunks: return chunks with fixed sequence length and
            variable signal length. Default returns chunks with fixed signal
            length and variable length sequences.
        base_pred: Is this a base prediction model? Default: mods
        full: Is the input data a full read Taiyaki mapped signal file?
            Default: chunks

    Returns:
        sigs: list of signal chunks
        labels: list of mod/unmod labels for the corresponding chunks
        refs: list of reference sequences for each chunk
        base_locs: location for each base in the corersponing chunk
    """

    # TODO include a padding option
    read_data = MappedSignalReader(dataset_path)
    n_reads = len(read_data.get_read_ids())
    if num_chunks is None or num_chunks == 0 or num_chunks > n_reads:
        num_chunks = n_reads
    if not isinstance(num_chunks, int):
        raise ValueError("num_chunks must be an integer")
    if len(chunk_context) != 2:
        raise ValueError("chunk_context must be length 2")
    if any(not isinstance(cc, int) for cc in chunk_context):
        raise ValueError("chunk_context must be integers")
    if focus_offset is None and not full:
        raise ValueError("Either --focus-offset or --full need to be set.")

    # TODO allow multiple modified bases
    alphabet_info = read_data.get_alphabet_information()
    mod, int_can_motif, motif_offset = validate_motif(read_data, mod_motif)

    if base_pred and alphabet_info.alphabet != "ACGT":
        raise ValueError(
            "Base prediction is not compatible with modified base "
            "training data. It requires a canonical alphabet."
        )

    mod_idx = alphabet_info.alphabet.find(mod)

    chunks = []

    reject_reasons = defaultdict(int)
    for read in read_data:
        sig = read.get_current(read.get_mapped_dacs_region())
        seq = "".join(
            alphabet_info.collapse_alphabet[b] for b in read.Reference
        )
        read_seq_to_sig = read.Ref_to_signal - read.Ref_to_signal[0]
        if len(read_seq_to_sig) > len(set(read_seq_to_sig)):
            continue

        if full:
            motif_pos = get_motif_pos(
                read.Reference, int_can_motif, motif_offset
            )
        else:
            motif_pos = [focus_offset]
        sig_start = sig_end = seq_start = seq_end = None
        for m_pos in motif_pos:
            # compute position at center of central base
            sig_focus_pos = (
                read_seq_to_sig[m_pos] + read_seq_to_sig[m_pos + 1]
            ) // 2
            if fixed_seq_len_chunks:
                seq_start = m_pos - chunk_context[0]
                seq_end = m_pos + chunk_context[1] + 1
                if seq_start <= 0:
                    reject_reasons["invalid_base_start"] += 1
                    continue
                if seq_end >= len(read_seq_to_sig):
                    reject_reasons["invalid_base_end"] += 1
                    continue
                sig_start = read_seq_to_sig[seq_start]
                sig_end = read_seq_to_sig[seq_end]
            else:
                sig_start = sig_focus_pos - chunk_context[0]
                sig_end = sig_focus_pos + chunk_context[1]
                if sig_start < 0:
                    reject_reasons["invalid_signal_start"] += 1
                    continue
                if sig_end > read_seq_to_sig[-1]:
                    reject_reasons["invalid_signal_end"] += 1
                    continue
                if sig_start >= sig_end:
                    reject_reasons["empty signal"] += 1
                    continue

            if base_pred:
                label = read.Reference[m_pos]
            else:
                label = int(read.Reference[m_pos] == mod_idx)
            try:
                chunks.append(
                    Chunk.extract_chunk_from_read(
                        sig,
                        seq,
                        read_seq_to_sig,
                        kmer_context_bases,
                        sig_start,
                        sig_end,
                        label=label,
                        seq_start=seq_start,
                        seq_end=seq_end,
                        sig_focus_pos=sig_focus_pos,
                        read_id=read.read_id,
                        read_seq_pos=m_pos,
                    )
                )
                reject_reasons["success"] += 1
                if reject_reasons["success"] >= num_chunks:
                    break
            except ValueError:
                LOGGER.debug(
                    f"FAILED_READ: {read.read_id} sig: {sig_start}-{sig_end} "
                    f"base: {seq_start}-{seq_end} "
                )
                reject_reasons["broken ref encoding"] += 1

    rej_summ = "\n".join(
        f"\t{count}\t{reason}"
        for count, reason in sorted(
            (count, reason) for reason, count in reject_reasons.items()
        )
    )
    LOGGER.info(f"Chunk selection summary:\n{rej_summ}\n")

    return chunks


def collate_var_len_input(batch_chunks):
    """Pads batch of variable sequence lengths

    note: the output is passed to the pack_padded_sequence,
        so that variable sequence lenghts can be handled by
        the RNN
    """
    # get sequence lengths
    lens = torch.tensor(
        [c.signal.shape[0] for c in batch_chunks if c.accepted], dtype=np.long
    )

    # padding
    sigs = torch.nn.utils.rnn.pad_sequence(
        [
            torch.Tensor(c.signal).unsqueeze(1)
            for c in batch_chunks
            if c.accepted
        ]
    )
    sigs = rnn.pack_padded_sequence(sigs, lens, enforce_sorted=False)
    seqs = torch.nn.utils.rnn.pad_sequence(
        [
            torch.Tensor(c.seq_nn_input).permute(1, 0)
            for c in batch_chunks
            if c.accepted
        ]
    )
    seqs = rnn.pack_padded_sequence(seqs, lens, enforce_sorted=False)
    # get labels
    labels = torch.tensor(
        np.array([c.label for c in batch_chunks if c.accepted], dtype=np.long)
    )
    read_data = [
        (c.read_id, c.read_seq_pos) for c in batch_chunks if c.accepted
    ]

    return (sigs, seqs, lens), labels, read_data


def collate_fixed_len_input(batch_chunks):
    """Collate data with fixed width inputs

    Note that inputs with be in Time x Batch x Features (TBF) dimension order
    """
    # convert inputs to TBF
    sigs = (
        torch.Tensor([c.signal for c in batch_chunks if c.accepted])
        .permute(1, 0)
        .unsqueeze(2)
    )
    seqs = torch.Tensor(
        [c.seq_nn_input for c in batch_chunks if c.accepted]
    ).permute(2, 0, 1)
    labels = torch.tensor(
        np.array([c.label for c in batch_chunks if c.accepted], dtype=np.long)
    )
    read_data = [
        (c.read_id, c.read_seq_pos) for c in batch_chunks if c.accepted
    ]
    return (sigs, seqs), labels, read_data


class RemoraDataset(torch.utils.data.Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __getitem__(self, index):
        return self.chunks[index]

    def __len__(self):
        return len(self.chunks)


def load_datasets(
    dataset_path,
    chunk_context,
    focus_offset=None,
    batch_size=DEFAULT_BATCH_SIZE,
    num_chunks=None,
    fixed_seq_len_chunks=False,
    mod_motif=None,
    base_pred=True,
    val_prop=0.0,
    num_data_workers=0,
    kmer_context_bases=constants.DEFAULT_CONTEXT_BASES,
    full=False,
):
    chunks = load_chunks(
        dataset_path,
        num_chunks,
        mod_motif,
        chunk_context,
        kmer_context_bases,
        focus_offset,
        fixed_seq_len_chunks,
        base_pred,
        full,
    )
    label_counts = Counter(c.label for c in chunks)
    if len(label_counts) <= 1:
        raise ValueError(
            "One or fewer output labels found. Ensure --focus-offset and "
            "--mod are specified correctly"
        )
    LOGGER.info(f"Label distribution: {label_counts}")

    collate_fn = (
        collate_var_len_input
        if fixed_seq_len_chunks
        else collate_fixed_len_input
    )

    if val_prop <= 0.0:
        dl_val = dl_val_trn = None
        trn_set = RemoraDataset(chunks)
    else:
        np.random.shuffle(chunks)
        val_idx = int(len(chunks) * val_prop)
        val_set = RemoraDataset(chunks[:val_idx])
        val_trn_set = RemoraDataset(chunks[val_idx : val_idx + val_idx])
        trn_set = RemoraDataset(chunks[val_idx:])
        dl_val = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_data_workers,
            drop_last=False,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        dl_val_trn = torch.utils.data.DataLoader(
            val_trn_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_data_workers,
            drop_last=False,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # shuffle and drop_last only if this is a split train/validation set.
    dl_trn = torch.utils.data.DataLoader(
        trn_set,
        batch_size=batch_size,
        shuffle=val_prop > 0,
        num_workers=num_data_workers,
        drop_last=val_prop > 0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return dl_trn, dl_val, dl_val_trn
