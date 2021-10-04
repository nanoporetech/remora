from collections import defaultdict, Counter
from dataclasses import dataclass

import numpy as np
from taiyaki.mapped_signal_files import MappedSignalReader
import torch

from remora import constants, log, RemoraError, util

LOGGER = log.get_logger()

DEFAULT_BATCH_SIZE = 1024

BASE_ENCODINGS = {
    "A": np.array([1, 0, 0, 0], dtype=np.bool8),
    "C": np.array([0, 1, 0, 0], dtype=np.bool8),
    "G": np.array([0, 0, 1, 0], dtype=np.bool8),
    "T": np.array([0, 0, 0, 1], dtype=np.bool8),
    "N": np.array([0, 0, 0, 0], dtype=np.bool8),
}
ENCODING_LEN = BASE_ENCODINGS["A"].shape[0]


@dataclass
class RemoraRead:
    """Object to hold information about a read relevant to Remora training and
    inference.

    Args:
        sig (np.ndarray): Normalized signal
        seq (str): Canonical sequence
        seq_to_sig_map (np.ndarray): Position within signal array assigned to
            each base in seq
        read_id (str): Read identifier
        int_seq (np.ndarray): Encoded sequence for training/validation
            (optional)
    """

    sig: np.ndarray
    seq: str
    seq_to_sig_map: np.ndarray
    read_id: str = None
    int_seq: np.ndarray = None

    @classmethod
    def from_taiyaki_read(cls, read, collapse_alphabet):
        sig = read.get_current(read.get_mapped_dacs_region())
        seq = "".join(collapse_alphabet[b] for b in read.Reference)
        seq_to_sig_map = read.Ref_to_signal - read.Ref_to_signal[0]
        int_seq = read.Reference
        return cls(sig, seq, seq_to_sig_map, util.to_str(read.read_id), int_seq)


def load_taiyaki_dataset(dataset_path):
    input_msf = MappedSignalReader(dataset_path)
    alphabet_info = input_msf.get_alphabet_information()
    reads = [
        RemoraRead.from_taiyaki_read(tai_read, alphabet_info.collapse_alphabet)
        for tai_read in input_msf
    ]
    return reads, alphabet_info.alphabet, alphabet_info.collapse_alphabet


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
        seq_focus_pos (int): Index within sequence on which the chunk is
            focuesed for prediction.
        label (int): Integer label for training/validation.
        read_id (str): Read ID
        read_seq_pos (int): Position within read for validation purposes
    """

    # TODO include a padding option

    signal: np.ndarray
    sequence: str
    seq_to_sig_map: np.ndarray
    before_context_seq: str
    after_context_seq: str
    sig_focus_pos: int
    seq_focus_pos: int
    label: int
    read_id: str
    read_seq_pos: int
    _base_sig_lens: np.ndarray = None
    _seq_nn_input: np.ndarray = None

    def mask_focus_base(self):
        if self.sequence[self.seq_focus_pos] == "N":
            return
        self.sequence = (
            f"{self.sequence[:self.seq_focus_pos]}"
            f"N{self.sequence[self.seq_focus_pos + 1:]}"
        )
        self._seq_nn_input = None

    def check(self):
        if self.signal.size <= 0:
            LOGGER.debug(
                f"FAILED_CHUNK: no_sig {self.read_id} {self.read_seq_pos}"
            )
            raise RemoraError("No signal for chunk")
        if self.seq_nn_input.shape[1] != self.signal.size:
            LOGGER.debug(
                f"FAILED_CHUNK: seq_nn_len {self.read_id} {self.read_seq_pos}"
            )
            raise RemoraError("Invalid encoded sig length")
        if len(self.sequence) != self.seq_to_sig_map.shape[0] - 1:
            LOGGER.debug(
                f"FAILED_CHUNK: map_len {self.read_id} {self.read_seq_pos}"
            )
            raise RemoraError("Invalid sig to seq map length")

    @classmethod
    def extract_chunk_from_read(
        cls,
        read,
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
            read_sig (RemoraRead): Read object
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
                np.searchsorted(read.seq_to_sig_map, sig_start, side="right")
                - 1
            )
            seq_end = np.searchsorted(read.seq_to_sig_map, sig_end, side="left")

        chunk_seq_to_sig = read.seq_to_sig_map[seq_start : seq_end + 1].copy()
        chunk_seq_to_sig[0] = sig_start
        chunk_seq_to_sig[-1] = sig_end
        chunk_seq_to_sig -= sig_start
        chunk_seq_to_sig = chunk_seq_to_sig.astype(np.int32)

        kmer_before_bases, kmer_after_bases = kmer_context_bases
        before_context_seq = read.seq[
            max(0, seq_start - kmer_before_bases) : seq_start
        ]
        after_context_seq = read.seq[seq_end : seq_end + kmer_after_bases]
        if (
            len(before_context_seq) != kmer_before_bases
            or len(after_context_seq) != kmer_after_bases
        ):
            LOGGER.debug(
                f"FAILED_CHUNK: context_seq {read_id} {read_seq_pos}  "
                f"{seq_start} {kmer_before_bases}"
            )
            raise RemoraError("Invalid context seq extracted")

        return cls(
            read.sig[sig_start:sig_end],
            read.seq[seq_start:seq_end],
            chunk_seq_to_sig,
            before_context_seq,
            after_context_seq,
            sig_focus_pos - sig_start,
            read_seq_pos - seq_start,
            label,
            read_id,
            read_seq_pos,
        )

    @property
    def kmer_len(self):
        return len(self.before_context_seq) + len(self.after_context_seq) + 1

    @property
    def seq_w_context(self):
        return self.before_context_seq + self.sequence + self.after_context_seq

    @property
    def base_sig_lens(self):
        if self._base_sig_lens is None:
            self._base_sig_lens = np.diff(self.seq_to_sig_map)
        return self._base_sig_lens

    @property
    def seq_nn_input(self):
        """Representation of chunk sequence presented to neural network. The
        length of the second dimension should match chunk.signal.shape[0]. The
        first dimension represents the number of input features.
        """
        if self._seq_nn_input is None:
            enc_kmers = np.concatenate(
                [
                    np.stack(
                        [
                            BASE_ENCODINGS[b]
                            for b in self.seq_w_context[
                                offset : offset + len(self.sequence)
                            ]
                        ],
                        axis=1,
                    )
                    for offset in range(self.kmer_len)
                ]
            )
            self._seq_nn_input = np.repeat(
                enc_kmers, self.base_sig_lens, axis=1
            )
        return self._seq_nn_input


def load_chunks(
    reads,
    motif,
    label_conv,
    num_chunks=None,
    chunk_context=constants.DEFAULT_CHUNK_CONTEXT,
    kmer_context_bases=constants.DEFAULT_KMER_CONTEXT_BASES,
    focus_offset=constants.DEFAULT_FOCUS_OFFSET,
    fixed_seq_len_chunks=False,
    base_pred=False,
    full=False,
):
    """
    Args:
        reads (iterable): Iteratable of RemoraRead objects from which to load
            chunks
        motif (util.Motif): Remora motif object
        label_conv (np.array): Convert reference labels to Remora labels to
            predict. -1 indicates invalid Remora label
        num_chunks (int): Total maximum number of chunks to return
        chunk_context (tuple): 2-tuple containing context signal or bases for
            each chunk
        kmer_context_bases (tuple): Number of bases before and after to included
            in encoded reference
        focus_offset (int): Index of focus position in each reference chunk
        fixed_seq_len_chunks (bool): return chunks with fixed sequence length
            and variable signal length. Default returns chunks with fixed signal
            length and variable length sequences.
        base_pred (bool): Is this a base prediction model? Default: mods
        full (bool): Is the input data a full read Taiyaki mapped signal file?
            Default: chunks

    Returns:
        List of Chunk objects
    """

    if num_chunks is not None and not isinstance(num_chunks, int):
        raise ValueError(
            f"num_chunks must be an integer or None ({num_chunks})"
        )
    if len(chunk_context) != 2:
        raise ValueError("chunk_context must be length 2")
    if any(not isinstance(cc, int) for cc in chunk_context):
        raise ValueError("chunk_context must be integers")
    if focus_offset is None and not full:
        raise ValueError("Either --focus-offset or --full need to be set.")

    chunks = []

    reject_reasons = defaultdict(int)
    for read in reads:
        if full:
            motif_pos = [
                m.start() + motif.focus_pos
                for m in motif.pattern.finditer(read.seq)
            ]
        else:
            if focus_offset >= len(read.seq):
                reject_reasons["[--focus-offset] past read seq"] += 1
                continue
            motif_pos = [focus_offset]
            read_motif = read.seq[
                focus_offset
                - motif.focus_pos : focus_offset
                + motif.num_bases_after_focus
                + 1
            ]
            if not motif.pattern.match(read_motif):
                reject_reasons["Focus position motif mismatch"] += 1
                continue
        sig_start = sig_end = seq_start = seq_end = None
        for m_pos in motif_pos:
            # compute position at center of central base
            sig_focus_pos = (
                read.seq_to_sig_map[m_pos] + read.seq_to_sig_map[m_pos + 1]
            ) // 2
            if fixed_seq_len_chunks:
                seq_start = m_pos - chunk_context[0]
                seq_end = m_pos + chunk_context[1] + 1
                if seq_start <= 0:
                    reject_reasons["Invalid base start"] += 1
                    continue
                if seq_end >= len(read.seq_to_sig_map):
                    reject_reasons["Invalid base end"] += 1
                    continue
                sig_start = read.seq_to_sig_map[seq_start]
                sig_end = read.seq_to_sig_map[seq_end]
            else:
                sig_start = sig_focus_pos - chunk_context[0]
                sig_end = sig_focus_pos + chunk_context[1]
                if sig_start < 0:
                    reject_reasons["Invalid signal start"] += 1
                    continue
                if sig_end > read.seq_to_sig_map[-1]:
                    reject_reasons["Invalid signal end"] += 1
                    continue
                if sig_start >= sig_end:
                    reject_reasons["Empty signal"] += 1
                    continue

            label = (
                -1 if read.int_seq is None else label_conv[read.int_seq[m_pos]]
            )
            try:
                chunk = Chunk.extract_chunk_from_read(
                    read,
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
                if base_pred:
                    chunk.mask_focus_base()
                # check that chunk is valid
                chunk.check()
                chunks.append(chunk)
                reject_reasons["success"] += 1
                if num_chunks is not None and len(chunks) >= num_chunks:
                    break
            except Exception as e:
                LOGGER.debug(
                    f"FAILED_READ: {read.read_id} sig: {sig_start}-{sig_end} "
                    f"base: {seq_start}-{seq_end}. ERROR: {e}"
                )
                # TODO add full traceback to better track down failed reads
                reject_reasons[str(e)] += 1
                continue

    rej_summ = "\n".join(
        f"\t{count}\t{reason}"
        for count, reason in sorted(
            (count, reason) for reason, count in reject_reasons.items()
        )
    )
    LOGGER.info(f"Chunk selection summary:\n{rej_summ}\n")
    if len(chunks) == 0:
        raise RemoraError("No valid chunks extracted")

    return chunks


# TODO add version for variable length datasets
@dataclass
class RemoraDataset:
    # data attributes
    sig_tensor: torch.Tensor
    seq_tensor: torch.Tensor
    labels: torch.LongTensor
    read_data: list

    # scalar metadata attributes
    chunk_context: tuple = constants.DEFAULT_CHUNK_CONTEXT
    kmer_context_bases: tuple = constants.DEFAULT_KMER_CONTEXT_BASES
    base_pred: bool = False
    mod_bases: str = ""

    # batch attributes (can change after data loading)
    store_read_data: bool = False
    batch_size: int = constants.DEFAULT_BATCH_SIZE
    shuffle_on_iter: bool = False
    drop_last: bool = False

    def __post_init__(self):
        self.dataset_len = self.sig_tensor.shape[0]
        self.n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if not self.drop_last and remainder > 0:
            self.n_batches += 1

    def shuffle_data(self):
        shuf_idx = torch.randperm(self.dataset_len)
        self.sig_tensor = self.sig_tensor[shuf_idx]
        self.seq_tensor = self.seq_tensor[shuf_idx]
        self.labels = self.labels[shuf_idx]
        if self.store_read_data:
            self.read_data = [self.read_data[si] for si in shuf_idx]

    def __iter__(self):
        if self.shuffle_on_iter:
            self.shuffle_data()
        self._batch_i = 0
        return self

    def __next__(self):
        if self._batch_i >= self.n_batches:
            raise StopIteration
        b_st = self._batch_i * self.batch_size
        b_en = b_st + self.batch_size
        self._batch_i += 1
        batch_read_data = None
        if self.store_read_data:
            batch_read_data = self.read_data[b_st:b_en]
        return (
            (self.sig_tensor[b_st:b_en], self.seq_tensor[b_st:b_en]),
            self.labels[b_st:b_en],
            batch_read_data,
        )

    def __len__(self):
        return self.n_batches

    def get_label_counts(self):
        return Counter(self.labels.numpy())

    def split_data(self, val_prop):
        if self.dataset_len < int(1 / val_prop) * 2:
            raise RemoraError(
                "Too few chunks to extract validation proportion "
                f"({self.dataset_len} < {int(1 / val_prop) * 2})"
            )
        common_kwargs = {
            "chunk_context": self.chunk_context,
            "kmer_context_bases": self.kmer_context_bases,
            "base_pred": self.base_pred,
            "mod_bases": self.mod_bases,
            "store_read_data": self.store_read_data,
            "batch_size": self.batch_size,
        }
        val_idx = int(self.dataset_len * val_prop)
        val_ds = RemoraDataset(
            self.sig_tensor[:val_idx],
            self.seq_tensor[:val_idx],
            self.labels[:val_idx],
            self.read_data[:val_idx] if self.store_read_data else None,
            shuffle_on_iter=False,
            drop_last=False,
            **common_kwargs,
        )
        val_trn_ds = RemoraDataset(
            self.sig_tensor[val_idx : val_idx + val_idx],
            self.seq_tensor[val_idx : val_idx + val_idx],
            self.labels[val_idx : val_idx + val_idx],
            self.read_data[val_idx : val_idx + val_idx]
            if self.store_read_data
            else None,
            shuffle_on_iter=False,
            drop_last=False,
            **common_kwargs,
        )
        trn_ds = RemoraDataset(
            self.sig_tensor[val_idx:],
            self.seq_tensor[val_idx:],
            self.labels[val_idx:],
            self.read_data[val_idx:] if self.store_read_data else None,
            shuffle_on_iter=False,
            drop_last=False,
            **common_kwargs,
        )
        return trn_ds, val_trn_ds, val_ds

    def save_dataset(self, filename):
        np.savez(
            filename,
            sig_tensor=self.sig_tensor,
            seq_tensor=self.seq_tensor,
            labels=self.labels,
            read_data=self.read_data,
            chunk_context=self.chunk_context,
            kmer_context_bases=self.kmer_context_bases,
            base_pred=self.base_pred,
            mod_bases=self.mod_bases,
        )

    @classmethod
    def load_from_file(cls, filename, *args, **kwargs):
        # use allow_pickle=True to allow None type in read_data
        data = np.load(filename, allow_pickle=True)
        sig_tensor = torch.from_numpy(data["sig_tensor"])
        seq_tensor = torch.from_numpy(data["seq_tensor"])
        labels = torch.from_numpy(data["labels"])
        read_data = data["read_data"].tolist()
        chunk_context = tuple(data["chunk_context"].tolist())
        kmer_context_bases = tuple(data["kmer_context_bases"].tolist())
        base_pred = data["base_pred"].item()
        mod_bases = data["mod_bases"].item()
        return cls(
            sig_tensor,
            seq_tensor,
            labels,
            read_data,
            chunk_context=chunk_context,
            kmer_context_bases=kmer_context_bases,
            base_pred=base_pred,
            mod_bases=mod_bases,
            store_read_data=read_data is not None,
            *args,
            **kwargs,
        )

    @classmethod
    def load_from_chunks(cls, chunks, store_read_data=False, *args, **kwargs):
        # pre-compute full tensors from which to load data
        sig_tensor = torch.Tensor(
            np.stack([c.signal for c in chunks])
        ).unsqueeze(1)
        seq_tensor = torch.from_numpy(
            np.stack([c.seq_nn_input for c in chunks])
        ).type(torch.FloatTensor)
        labels = torch.LongTensor([c.label for c in chunks])
        read_data = None
        if store_read_data:
            read_data = [(c.read_id, c.read_seq_pos) for c in chunks]
        return cls(
            sig_tensor,
            seq_tensor,
            labels,
            read_data,
            store_read_data=store_read_data,
            *args,
            **kwargs,
        )
