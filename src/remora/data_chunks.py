from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from remora import constants, log, RemoraError, util

LOGGER = log.get_logger()

DEFAULT_BATCH_SIZE = 1024


@dataclass
class RemoraRead:
    """Object to hold information about a read relevant to Remora training and
    inference.

    Args:
        sig (np.ndarray): Normalized signal
        seq (np.ndarray): Encoded sequence for training/validation
        seq_to_sig_map (np.ndarray): Position within signal array assigned to
            each base in seq
        read_id (str): Read identifier
    """

    sig: np.ndarray
    can_seq: np.ndarray
    seq_to_sig_map: np.ndarray
    read_id: str = None
    labels: np.ndarray = None

    @classmethod
    def from_taiyaki_read(cls, read, can_conv, label_conv, check_read=True):
        sig = read.get_current(read.get_mapped_dacs_region())
        can_seq = can_conv[read.Reference]
        seq_to_sig_map = read.Ref_to_signal - read.Ref_to_signal[0]
        labels = None if label_conv is None else label_conv[read.Reference]
        read = cls(
            sig, can_seq, seq_to_sig_map, util.to_str(read.read_id), labels
        )
        if check_read:
            read.check()
        return read

    def check(self):
        if self.seq_to_sig_map.size != self.can_seq.size + 1:
            LOGGER.debug(
                "Invalid read: seq and mapping mismatch "
                f"{self.seq_to_sig_map.size} != {self.can_seq.size + 1}"
            )
            raise RemoraError("Invalid read: seq and mapping mismatch")
        if self.seq_to_sig_map[0] != 0:
            LOGGER.debug(
                "Invalid read: invalid mapping start "
                f"{self.seq_to_sig_map[0]} != 0"
            )
            raise RemoraError("Invalid read: mapping start")
        if self.seq_to_sig_map[-1] != self.sig.size:
            LOGGER.debug(
                "Invalid read: invalid mapping end "
                f"{self.seq_to_sig_map[-1]} != {self.sig.size}"
            )
            raise RemoraError("Invalid read: mapping end")
        if self.can_seq.max() > 3:
            LOGGER.debug("Invalid read: Invalid base {self.can_seq.max()}")
            raise RemoraError("Invalid read: Invalid base")
        if self.can_seq.min() < -1:
            LOGGER.debug("Invalid read: Invalid base {self.can_seq.min()}")
            raise RemoraError("Invalid read: Invalid base")
        # TODO add more logic to these tests

    def iter_motif_hits(self, motif):
        yield from np.where(
            np.logical_and.reduce(
                [
                    np.isin(
                        self.can_seq[
                            po : self.can_seq.size
                            - len(motif.int_pattern)
                            + po
                            + 1
                        ],
                        pi,
                    )
                    for po, pi in enumerate(motif.int_pattern)
                ]
            )
        )[0]

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
        if (
            seq_start >= kmer_before_bases
            and seq_end + kmer_after_bases <= self.can_seq.size
        ):
            chunk_seq = self.can_seq[
                seq_start - kmer_before_bases : seq_end + kmer_after_bases
            ]
        else:
            chunk_seq = np.full(
                seq_end - seq_start + sum(kmer_context_bases),
                -1,
                dtype=np.byte,
            )
            fill_st = 0
            fill_en = seq_end - seq_start + sum(kmer_context_bases)
            chunk_seq_st = seq_start - kmer_before_bases
            chunk_seq_en = seq_end + kmer_after_bases
            if seq_start < kmer_before_bases:
                fill_st = kmer_before_bases - seq_start
                chunk_seq_st = 0
            if seq_end + kmer_after_bases > self.can_seq.size:
                fill_en -= seq_end + kmer_after_bases - self.can_seq.size
                chunk_seq_en = self.can_seq.size
            chunk_seq[fill_st:fill_en] = self.can_seq[chunk_seq_st:chunk_seq_en]
        chunk = Chunk(
            chunk_sig,
            chunk_seq,
            chunk_seq_to_sig,
            kmer_context_bases,
            focus_sig_idx - sig_start,
            read_seq_pos - seq_start,
            read_seq_pos,
            self.read_id,
            label,
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
        kmer_context_bases,
        base_pred=False,
    ):
        for focus_base_idx in focus_base_indices:
            # compute position at center of central base
            focus_sig_idx = (
                self.seq_to_sig_map[focus_base_idx]
                + self.seq_to_sig_map[focus_base_idx + 1]
            ) // 2
            label = -1 if self.labels is None else self.labels[focus_base_idx]
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
        seq_w_context (str): Integer encoded sequence including context basees
            for kmer extraction. Note that seq_to_sig_map only corresponds to
            the central sequence without the context sequence
        seq_to_sig_map (np.array): Array of length one more than sequence with
            values representing indices into signal for each base.
        kmer_context_bases (tuple): Number of context bases included before and
            after the chunk sequence
        sig_focus_pos (int): Index within signal array on which the chunk is
            focuesed for prediction. May be used in model architecture in the
            future.
        seq_focus_pos (int): Index within sequence (without context bases) on
            which the chunk is focuesed for prediction.
        read_seq_pos (int): Position within read for validation purposes
        read_id (str): Read ID
        label (int): Integer label for training/validation.
    """

    signal: np.ndarray
    seq_w_context: np.ndarray
    seq_to_sig_map: np.ndarray
    kmer_context_bases: tuple
    sig_focus_pos: int
    seq_focus_pos: int
    read_seq_pos: int
    read_id: str = None
    label: int = None
    _base_sig_lens: np.ndarray = None

    def mask_focus_base(self):
        self.seq_w_context[self.seq_focus_pos + self.kmer_context_bases[0]] = -1

    def check(self):
        if self.signal.size <= 0:
            LOGGER.debug(
                f"FAILED_CHUNK: no_sig {self.read_id} {self.read_seq_pos}"
            )
            raise RemoraError("No signal for chunk")
        if (
            self.seq_w_context.size - sum(self.kmer_context_bases)
            != self.seq_to_sig_map.size - 1
        ):
            LOGGER.debug(
                f"FAILED_CHUNK: map_len {self.read_id} {self.read_seq_pos}"
            )
            raise RemoraError("Invalid sig to seq map length")
        # TODO add more logic to these checks

    @property
    def kmer_len(self):
        return sum(self.kmer_context_bases) + 1

    @property
    def seq_len(self):
        return self.seq_w_context.size - sum(self.kmer_context_bases)

    @property
    def base_sig_lens(self):
        if self._base_sig_lens is None:
            self._base_sig_lens = np.diff(self.seq_to_sig_map)
        return self._base_sig_lens


@dataclass
class RemoraDataset:
    """Remora dataset

    Args:
        sig_tensor (torch.FloatTensor): Signal chunks (dims: nchunks, 1, ntime)
        seq_array (np.array): Variable width integer encoded sequence chunks
            dtype : np.byte
            dims  : nchunks, max_seq_len + sum(kmer_context_bases)
        seq_mappings (np.array): Mapping from seq positions to signal tensor
            Note only sequence without kmer_context_bases bases is included in
            mappings
            dtype : np.short
            dims  : nchunks, max_seq_len + 1
        seq_lens (np.array): Length of sequences (without kmer context bases)
            dtype : np.short
            dims  : nchunks
        labels: torch.LongTensor
        read_data (list): Read data (read_ids, and focus positions
        nchunks (int): Current number of chunks. If None (default), will be set
            to sig_tensor.shape[0]. If set, only chunks up to this value are
            assumed to be valid.
        chunk_context (tuple): 2-tuple containing the number of signal points
            before and after the central position.
        max_seq_len (int): Maximum sequence length of a chunk (used to set
            dimension for seq arrays.
        kmer_context_bases (tuple): 2-tuple containing the bases to include in
            the encoded k-mer presented as input.
        base_pred (bool): Are labels predicting base? Default is mod_bases.
        mod_bases (str): Modified base single letter codes represented by labels
        store_read_data (bool): Is read data stored? Mostly for validation
        batch_size (int): Size of batches to be produced
        shuffle_on_iter (bool): Shuffle data before each iteration over batches
        drop_last (bool): Drop the last batch of each iteration

    Yields:
        Batches of data for training or inference
    """

    # data attributes
    sig_tensor: torch.Tensor
    seq_array: np.ndarray
    seq_mappings: np.ndarray
    seq_lens: np.ndarray
    labels: torch.LongTensor
    read_data: list
    nchunks: int = None

    # scalar metadata attributes
    chunk_context: tuple = constants.DEFAULT_CHUNK_CONTEXT
    max_seq_len: int = constants.DEFAULT_MAX_SEQ_LEN
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
        if chunk.seq_len > self.max_seq_len:
            raise RemoraError("Chunk sequence too long")
        self.sig_tensor[self.nchunks, 0] = torch.from_numpy(chunk.signal)
        self.seq_array[
            self.nchunks, : chunk.seq_w_context.size
        ] = chunk.seq_w_context
        self.seq_mappings[
            self.nchunks, : chunk.seq_to_sig_map.size
        ] = chunk.seq_to_sig_map
        self.seq_lens[self.nchunks] = chunk.seq_len
        self.labels[self.nchunks] = chunk.label
        if self.store_read_data:
            self.read_data.append((chunk.read_id, chunk.read_seq_pos))
        self.nchunks += 1

    def clip_chunks(self):
        if self.nchunks == self.sig_tensor.shape[0]:
            self.set_nbatches()
            return
        elif self.nchunks > self.sig_tensor.shape[0]:
            raise RemoraError("More chunks indicated than provided.")
        self.sig_tensor = self.sig_tensor[: self.nchunks]
        self.seq_array = self.seq_array[: self.nchunks]
        self.seq_mappings = self.seq_mappings[: self.nchunks]
        self.seq_lens = self.seq_lens[: self.nchunks]
        self.labels = self.labels[: self.nchunks]
        # reset nbatches after trimming
        self.set_nbatches()
        if self.store_read_data and len(self.read_data) != self.nchunks:
            raise RemoraError("More chunks indicated than read_data provided.")

    def trim_kmer_context_bases(self, new_kmer_context_bases):
        if new_kmer_context_bases is None:
            return
        if (
            new_kmer_context_bases[0] == self.kmer_context_bases[0]
            and new_kmer_context_bases[1] == self.kmer_context_bases[1]
        ):
            return
        if (
            new_kmer_context_bases[0] > self.kmer_context_bases[0]
            or new_kmer_context_bases[1] > self.kmer_context_bases[1]
        ):
            raise RemoraError(
                f"Cannot expand kmer context (prev:{self.kmer_context_bases} ; "
                f"requested_new:{new_kmer_context_bases})"
            )
        if new_kmer_context_bases[0] < self.kmer_context_bases[0]:
            seq_diff = self.kmer_context_bases[0] - new_kmer_context_bases[0]
            self.seq_array = np.ascontiguousarray(self.seq_array[:, seq_diff:])
        self.kmer_context_bases = new_kmer_context_bases

    def trim_chunk_context(self, new_chunk_context):
        if new_chunk_context is None:
            return
        if (
            new_chunk_context[0] == self.chunk_context[0]
            and new_chunk_context[1] == self.chunk_context[1]
        ):
            return
        if (
            new_chunk_context[0] > self.chunk_context[0]
            or new_chunk_context[1] > self.chunk_context[1]
        ):
            raise RemoraError(
                f"Cannot expand chunk_context (prev:{self.chunk_context} ; "
                f"requested_new:{new_chunk_context})"
            )
        raise NotImplementedError("Cannot currently trim chunk context.")

    def shuffle_dataset(self):
        if not self.is_trimmed:
            raise RemoraError("Cannot shuffle an untrimmed dataset.")
        shuf_idx = torch.randperm(self.nchunks)
        self.sig_tensor = self.sig_tensor[shuf_idx]
        self.seq_array = self.seq_array[shuf_idx]
        self.seq_mappings = self.seq_mappings[shuf_idx]
        self.seq_lens = self.seq_lens[shuf_idx]
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
                self.seq_array[b_st:b_en],
                self.seq_mappings[b_st:b_en],
                self.seq_lens[b_st:b_en],
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
            self.seq_array[:val_idx],
            self.seq_mappings[:val_idx],
            self.seq_lens[:val_idx],
            self.labels[:val_idx],
            self.read_data[:val_idx] if self.store_read_data else None,
            shuffle_on_iter=False,
            drop_last=False,
            **common_kwargs,
        )
        val_trn_ds = RemoraDataset(
            self.sig_tensor[val_idx : val_idx + val_idx],
            self.seq_array[val_idx : val_idx + val_idx],
            self.seq_mappings[val_idx : val_idx + val_idx],
            self.seq_lens[val_idx : val_idx + val_idx],
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
            self.seq_array[val_idx:],
            self.seq_mappings[val_idx:],
            self.seq_lens[val_idx:],
            self.labels[val_idx:],
            self.read_data[val_idx:] if self.store_read_data else None,
            shuffle_on_iter=False,
            drop_last=False,
            **common_kwargs,
        )
        return trn_ds, val_trn_ds, val_ds

    def save_dataset(self, filename):
        self.clip_chunks()
        np.savez(
            filename,
            sig_tensor=self.sig_tensor,
            seq_array=self.seq_array,
            seq_mappings=self.seq_mappings,
            seq_lens=self.seq_lens,
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
        seq_array = data["seq_array"]
        seq_mappings = data["seq_mappings"]
        seq_lens = data["seq_lens"]
        labels = torch.from_numpy(data["labels"])
        read_data = data["read_data"].tolist()
        chunk_context = tuple(data["chunk_context"].tolist())
        kmer_context_bases = tuple(data["kmer_context_bases"].tolist())
        base_pred = data["base_pred"].item()
        mod_bases = data["mod_bases"].item()
        motif = (data["motif"].item(), int(data["motif_offset"].item()))
        return cls(
            sig_tensor,
            seq_array,
            seq_mappings,
            seq_lens,
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
        max_seq_len,
        kmer_context_bases,
        store_read_data=False,
        *args,
        **kwargs,
    ):
        sig_tensor = torch.empty(
            (num_chunks, 1, sum(chunk_context)), dtype=torch.float32
        )
        seq_array = np.empty(
            (num_chunks, max_seq_len + sum(kmer_context_bases)),
            dtype=np.byte,
        )
        seq_mappings = np.empty(
            (num_chunks, max_seq_len + 1),
            dtype=np.short,
        )
        seq_lens = np.empty(num_chunks, dtype=np.short)
        labels = torch.empty(num_chunks, dtype=torch.long)
        read_data = [] if store_read_data else None
        return cls(
            sig_tensor,
            seq_array,
            seq_mappings,
            seq_lens,
            labels,
            read_data,
            nchunks=0,
            store_read_data=store_read_data,
            chunk_context=chunk_context,
            max_seq_len=max_seq_len,
            kmer_context_bases=kmer_context_bases,
            *args,
            **kwargs,
        )
