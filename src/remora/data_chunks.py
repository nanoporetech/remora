from collections import Counter
from dataclasses import dataclass

import numpy as np
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
ENCODING_LEN = BASE_ENCODINGS["A"].size


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
        # TODO move this assert to a check method
        assert seq_to_sig_map.size == len(seq) + 1
        return cls(sig, seq, seq_to_sig_map, util.to_str(read.read_id), int_seq)

    def iter_motif_hits(self, motif):
        yield from (
            m.start() + motif.focus_pos
            for m in motif.pattern.finditer(self.seq)
        )

    def extract_chunk(
        self,
        focus_sig_idx,
        chunk_context,
        kmer_context_bases,
        label=-1,
        seq_start=None,
        seq_end=None,
        base_pred=False,
        read_seq_pos=-1,
        check_chunk=True,
    ):
        chunk_len = sum(chunk_context)
        sig_start = focus_sig_idx - chunk_context[0]
        sig_end = focus_sig_idx + chunk_context[1]
        seq_to_sig_offset = 0
        if sig_start >= 0 and sig_end <= self.sig.size:
            # chunk boundaries are within read signal
            chunk_sig = self.sig[sig_start:sig_end].copy()
        else:
            # if signal is not available for full chunk pad with zeros
            chunk_sig = np.zeros(chunk_len, dtype=np.float32)
            fill_st = 0
            fill_en = chunk_len
            if sig_start < 0:
                fill_st = -sig_start
                seq_to_sig_offset = -sig_start
                sig_start = 0
            if sig_end > self.sig.size:
                fill_en = self.sig.size - sig_start
                sig_end = self.sig.size
            chunk_sig[fill_st:fill_en] = self.sig[sig_start:sig_end]

        if seq_start is None or seq_end is None:
            seq_start = (
                np.searchsorted(self.seq_to_sig_map, sig_start, side="right")
                - 1
            )
            seq_end = np.searchsorted(self.seq_to_sig_map, sig_end, side="left")

        LOGGER.debug(f"sig: {sig_start}-{sig_end} seq: {seq_start}-{seq_end}")
        chunk_seq_to_sig = self.seq_to_sig_map[seq_start : seq_end + 1].copy()
        chunk_seq_to_sig[0] = sig_start - seq_to_sig_offset
        chunk_seq_to_sig -= sig_start - seq_to_sig_offset
        chunk_seq_to_sig[-1] = chunk_len
        chunk_seq_to_sig = chunk_seq_to_sig.astype(np.int32)

        # extract context sequence
        kmer_before_bases, kmer_after_bases = kmer_context_bases
        before_context_seq = self.seq[
            max(0, seq_start - kmer_before_bases) : seq_start
        ]
        # if not enough before sequence, pad with Ns
        if len(before_context_seq) < kmer_before_bases:
            before_context_seq = (
                "N" * (kmer_before_bases - len(before_context_seq))
                + before_context_seq
            )
        after_context_seq = self.seq[seq_end : seq_end + kmer_after_bases]
        if len(after_context_seq) < kmer_after_bases:
            after_context_seq = after_context_seq + "N" * (
                kmer_after_bases - len(after_context_seq)
            )

        chunk = Chunk(
            chunk_sig,
            self.seq[seq_start:seq_end],
            chunk_seq_to_sig,
            before_context_seq,
            after_context_seq,
            focus_sig_idx - sig_start,
            read_seq_pos - seq_start,
            label,
            self.read_id,
            read_seq_pos,
        )
        if check_chunk:
            chunk.check()
        if base_pred:
            chunk.mask_focus_base()
        return chunk

    def iter_chunks(
        self,
        focus_base_indices,
        chunk_context,
        label_conv,
        kmer_context_bases,
        base_pred,
    ):
        for focus_base_idx in focus_base_indices:
            # compute position at center of central base
            focus_sig_idx = (
                self.seq_to_sig_map[focus_base_idx]
                + self.seq_to_sig_map[focus_base_idx + 1]
            ) // 2
            label = (
                -1
                if self.int_seq is None or label_conv is None
                else label_conv[self.int_seq[focus_base_idx]]
            )
            yield self.extract_chunk(
                focus_sig_idx,
                chunk_context,
                kmer_context_bases,
                label=label,
                base_pred=base_pred,
                read_seq_pos=focus_base_idx,
            )


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


@dataclass
class RemoraDataset:
    """Remora dataset

    Args:
        sig_tensor (torch.FloatTensor): Signal chunks (dims: nchunks, 1, ntime)
        seq_tensor (torch.BoolTensor): Encoded sequence chunks
            (dims: nchunks, kmer_len*4, ntime)
        labels: torch.LongTensor
        read_data (list): Read data (read_ids, and focus positions
        curr_chunk (int): Current chunk. If None (default), will be set to
            nchunks. If set, only chunks up to this value are assumed to be
            valid.
        chunk_context (tuple): 2-tuple containing the number of signal points
            before and after the central position.
        kmer_context_bases (tuple): 2-tuple containing the bases to include in
            the encoded k-mer presented as input.
        base_pred (bool): Are labels predicting base? Default is mod_bases.
        mod_bases (str): Modified base single letter codes represented by labels
        store_read_data (bool): Is read data stored? Mostly for validation
        batch_size (int): Size of batches to be produced
        shuffle_on_iter (bool): Shuffle data before each iteration over batches
        drop_last (bool): Drop the last batch of each iteration
    """

    # data attributes
    sig_tensor: torch.Tensor
    seq_tensor: torch.Tensor
    labels: torch.LongTensor
    read_data: list
    nchunks: int = None

    # scalar metadata attributes
    chunk_context: tuple = constants.DEFAULT_CHUNK_CONTEXT
    kmer_context_bases: tuple = constants.DEFAULT_KMER_CONTEXT_BASES
    base_pred: bool = False
    mod_bases: str = ""
    motif: tuple = ("N", 0)

    # batch attributes (defaults set for training)
    store_read_data: bool = False
    batch_size: int = constants.DEFAULT_BATCH_SIZE
    shuffle_on_iter: bool = True
    drop_last: bool = True

    def set_nbatches(self):
        self.nbatches, remainder = divmod(self.nchunks, self.batch_size)
        if not self.drop_last and remainder > 0:
            self.nbatches += 1

    def __post_init__(self):
        if self.nchunks is None:
            self.nchunks = self.sig_tensor.shape[0]
        elif self.nchunks > self.sig_tensor.shape[0]:
            raise RemoraError("More chunks indicated than provided.")
        self.set_nbatches()

    def add_chunk(self, chunk):
        if self.nchunks >= self.sig_tensor.shape[0]:
            raise RemoraError("Cannot add chunk to currently allocated tensors")
        self.sig_tensor[self.nchunks, 0] = torch.from_numpy(chunk.signal)
        self.seq_tensor[self.nchunks] = torch.from_numpy(chunk.seq_nn_input)
        self.labels[self.nchunks] = chunk.label
        if self.store_read_data:
            self.read_data.append((chunk.read_id, chunk.read_seq_pos))
        self.nchunks += 1

    def trim_dataset(self):
        if self.nchunks == self.sig_tensor.shape[0]:
            self.set_nbatches()
            return
        elif self.nchunks > self.sig_tensor.shape[0]:
            raise RemoraError("More chunks indicated than provided.")
        self.sig_tensor = self.sig_tensor[: self.nchunks]
        self.seq_tensor = self.seq_tensor[: self.nchunks]
        self.labels = self.labels[: self.nchunks]
        # reset nbatches after trimming
        self.set_nbatches()
        if self.store_read_data and len(self.read_data) != self.nchunks:
            raise RemoraError("More chunks indicated than read_data provided.")

    def shuffle_dataset(self):
        if not self.is_trimmed:
            raise RemoraError("Cannot shuffle an untrimmed dataset.")
        shuf_idx = torch.randperm(self.nchunks)
        self.sig_tensor = self.sig_tensor[shuf_idx]
        self.seq_tensor = self.seq_tensor[shuf_idx]
        self.labels = self.labels[shuf_idx]
        if self.store_read_data:
            self.read_data = [self.read_data[si] for si in shuf_idx]

    def __iter__(self):
        if self.shuffle_on_iter:
            self.shuffle_dataset()
        self._batch_i = 0
        return self

    def __next__(self):

        if self._batch_i >= self.nbatches:
            raise StopIteration
        b_st = self._batch_i * self.batch_size
        b_en = b_st + self.batch_size
        self._batch_i += 1
        batch_read_data = None
        if self.store_read_data:
            batch_read_data = self.read_data[b_st:b_en]
        return (
            (
                self.sig_tensor[b_st:b_en],
                self.seq_tensor[b_st:b_en].type(torch.FloatTensor),
            ),
            self.labels[b_st:b_en],
            batch_read_data,
        )

    def __len__(self):
        return self.nbatches

    def get_label_counts(self):
        return Counter(self.labels.numpy())

    def split_data(self, val_prop):
        if not self.is_trimmed:
            raise RemoraError("Cannot split an untrimmed dataset.")
        elif self.sig_tensor.shape[0] < int(1 / val_prop) * 2:
            raise RemoraError(
                "Too few chunks to extract validation proportion "
                f"({self.sig_tensor.shape[0]} < {int(1 / val_prop) * 2})"
            )
        common_kwargs = {
            "chunk_context": self.chunk_context,
            "kmer_context_bases": self.kmer_context_bases,
            "base_pred": self.base_pred,
            "mod_bases": self.mod_bases,
            "store_read_data": self.store_read_data,
            "batch_size": self.batch_size,
        }
        val_idx = int(self.sig_tensor.shape[0] * val_prop)
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
        self.trim_dataset()
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
            motif=self.motif[0],
            motif_offset=self.motif[1],
        )

    @property
    def is_trimmed(self):
        return self.nchunks == self.sig_tensor.shape[0]

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
        motif = (data["motif"].item(), int(data["motif_offset"].item()))
        return cls(
            sig_tensor,
            seq_tensor,
            labels,
            read_data,
            chunk_context=chunk_context,
            kmer_context_bases=kmer_context_bases,
            base_pred=base_pred,
            mod_bases=mod_bases,
            motif=motif,
            store_read_data=read_data is not None,
            *args,
            **kwargs,
        )

    @classmethod
    def allocate_empty_chunks(
        cls,
        num_chunks,
        chunk_context,
        kmer_context_bases,
        store_read_data=False,
        *args,
        **kwargs,
    ):
        sig_dim_len = sum(chunk_context)
        seq_dim_len = ENCODING_LEN * (sum(kmer_context_bases) + 1)
        sig_tensor = torch.empty(
            (num_chunks, 1, sig_dim_len), dtype=torch.float32
        )
        seq_tensor = torch.empty(
            (num_chunks, seq_dim_len, sig_dim_len),
            dtype=torch.bool,
        )
        labels = torch.empty(num_chunks, dtype=torch.long)
        read_data = [] if store_read_data else None
        return cls(
            sig_tensor,
            seq_tensor,
            labels,
            read_data,
            nchunks=0,
            store_read_data=store_read_data,
            chunk_context=chunk_context,
            kmer_context_bases=kmer_context_bases,
            *args,
            **kwargs,
        )
