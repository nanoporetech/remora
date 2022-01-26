from collections import Counter
from dataclasses import dataclass

import numpy as np

from remora import constants, log, RemoraError, util

LOGGER = log.get_logger()

DEFAULT_BATCH_SIZE = 1024


@dataclass
class RemoraRead:
    """Object to hold information about a read relevant to Remora training and
    inference.

    Args:
        sig (np.ndarray): Normalized signal
        seq_to_sig_map (np.ndarray): Position within signal array assigned to
            each base in seq
        int_seq (np.ndarray): Encoded sequence for training/validation.
            See remora.util.seq_to_int
        str_seq (str): String sequence for training/validation. Ignored if
            int_seq is provided.
        read_id (str): Read identifier

    Note: Must provide either int_seq or str_seq. If str_seq is provided
    int_seq will be derived on init.
    """

    sig: np.ndarray
    seq_to_sig_map: np.ndarray
    int_seq: np.ndarray = None
    str_seq: str = None
    read_id: str = None
    labels: np.ndarray = None

    def __post_init__(self):
        if self.int_seq is None:
            if self.str_seq is None:
                raise RemoraError(
                    "Must provide sequence to initialize RemoraRead"
                )
            # if int_seq is not set, set from str_seq provided
            self.int_seq = util.seq_to_int(self.str_seq)
        else:
            # set str_seq from int_seq to ensure the sequences match
            self.str_seq = util.int_to_seq(self.int_seq)

    @classmethod
    def from_taiyaki_read(cls, read, can_conv, label_conv, check_read=True):
        sig = read.get_current(read.get_mapped_dacs_region())
        int_seq = can_conv[read.Reference]
        seq_to_sig_map = read.Ref_to_signal - read.Ref_to_signal[0]
        labels = None if label_conv is None else label_conv[read.Reference]
        read = cls(
            sig,
            seq_to_sig_map,
            int_seq=int_seq,
            read_id=util.to_str(read.read_id),
            labels=labels,
        )
        if check_read:
            read.check()
        return read

    @classmethod
    def test_read(cls, nbases=20, signal_per_base=10):
        """Spoofed read for testing"""
        return cls(
            np.zeros(nbases * signal_per_base),
            np.arange(nbases * signal_per_base + 1, step=signal_per_base),
            np.arange(nbases) % 4,
            "read0",
            np.zeros(nbases, dtype=np.long),
        )

    def check(self):
        if self.seq_to_sig_map.size != self.int_seq.size + 1:
            LOGGER.debug(
                "Invalid read: seq and mapping mismatch "
                f"{self.seq_to_sig_map.size} != {self.int_seq.size + 1}"
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
        if self.int_seq.max() > 3:
            LOGGER.debug(f"Invalid read: Invalid base {self.int_seq.max()}")
            raise RemoraError("Invalid read: Invalid base")
        if self.int_seq.min() < -1:
            LOGGER.debug(f"Invalid read: Invalid base {self.int_seq.min()}")
            raise RemoraError("Invalid read: Invalid base")
        # TODO add more logic to these tests

    def iter_motif_hits(self, motif):
        yield from np.where(
            np.logical_and.reduce(
                [
                    np.isin(
                        self.int_seq[
                            po : self.int_seq.size
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
                # record offset value by which to shift seq_to_sig_map
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

        # extract/compute sequence to signal mapping for this chunk
        chunk_seq_to_sig = self.seq_to_sig_map[seq_start : seq_end + 1].copy()
        # shift mapping relative to the chunk
        chunk_seq_to_sig -= sig_start - seq_to_sig_offset
        # set chunk ends to chunk boundaries
        chunk_seq_to_sig[0] = 0
        chunk_seq_to_sig[-1] = chunk_len
        chunk_seq_to_sig = chunk_seq_to_sig.astype(np.int32)

        # extract context sequence
        kmer_before_bases, kmer_after_bases = kmer_context_bases
        if (
            seq_start >= kmer_before_bases
            and seq_end + kmer_after_bases <= self.int_seq.size
        ):
            chunk_seq = self.int_seq[
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
            if seq_end + kmer_after_bases > self.int_seq.size:
                fill_en -= seq_end + kmer_after_bases - self.int_seq.size
                chunk_seq_en = self.int_seq.size
            chunk_seq[fill_st:fill_en] = self.int_seq[chunk_seq_st:chunk_seq_en]
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
            try:
                yield self.extract_chunk(
                    focus_sig_idx,
                    chunk_context,
                    kmer_context_bases,
                    label=label,
                    base_pred=base_pred,
                    read_seq_pos=focus_base_idx,
                )
            except Exception as e:
                LOGGER.debug(f"FAILED_CHUNK_EXTRACT {e}")


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
        mod_long_names (list): Modified base long names represented by labels
        store_read_data (bool): Is read data stored? Mostly for validation
        batch_size (int): Size of batches to be produced
        shuffle_on_iter (bool): Shuffle data before each iteration over batches
        drop_last (bool): Drop the last batch of each iteration

    Yields:
        Batches of data for training or inference
    """

    # data attributes
    sig_tensor: np.ndarray
    seq_array: np.ndarray
    seq_mappings: np.ndarray
    seq_lens: np.ndarray
    labels: np.ndarray
    read_data: list
    nchunks: int = None

    # scalar metadata attributes
    chunk_context: tuple = constants.DEFAULT_CHUNK_CONTEXT
    max_seq_len: int = None
    kmer_context_bases: tuple = constants.DEFAULT_KMER_CONTEXT_BASES
    base_pred: bool = False
    mod_bases: str = ""
    mod_long_names: list = None
    motifs: tuple = ("N", 0)

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
        if self.max_seq_len is None:
            self.max_seq_len = self.seq_mappings.shape[1] - 1
        if self.nchunks is None:
            self.nchunks = self.sig_tensor.shape[0]
        elif self.nchunks > self.sig_tensor.shape[0]:
            raise RemoraError("More chunks indicated than provided.")
        if not self.base_pred and len(self.mod_bases) != len(
            self.mod_long_names
        ):
            raise RemoraError(
                f"mod_long_names ({self.mod_long_names}) must be same length "
                f"as mod_bases ({self.mod_bases})"
            )
        self.set_nbatches()
        self.shuffled = False

    def add_chunk(self, chunk):
        if self.nchunks >= self.labels.size:
            raise RemoraError("Cannot add chunk to currently allocated tensors")
        if chunk.seq_len > self.max_seq_len:
            raise RemoraError("Chunk sequence too long to store")
        self.sig_tensor[self.nchunks, 0] = chunk.signal
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

    def add_batch(self, b_sig, b_seq, b_ss_map, b_seq_lens, b_labels, b_rd):
        batch_size = b_labels.size
        b_st, b_en = self.nchunks, self.nchunks + batch_size
        if self.nchunks + batch_size > self.labels.size:
            raise RemoraError("Cannot add batch to currently allocated tensors")
        # TODO check that applicable dims are compatible
        self.sig_tensor[b_st:b_en] = b_sig
        self.seq_array[b_st:b_en] = b_seq
        self.seq_mappings[b_st:b_en] = b_ss_map
        self.seq_lens[b_st:b_en] = b_seq_lens
        self.labels[b_st:b_en] = b_labels
        if self.store_read_data:
            self.read_data.extend(b_rd)
        self.nchunks += batch_size

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
        shuf_idx = np.random.permutation(self.nchunks)
        self.sig_tensor = self.sig_tensor[shuf_idx]
        self.seq_array = self.seq_array[shuf_idx]
        self.seq_mappings = self.seq_mappings[shuf_idx]
        self.seq_lens = self.seq_lens[shuf_idx]
        self.labels = self.labels[shuf_idx]
        if self.store_read_data:
            self.read_data = [self.read_data[si] for si in shuf_idx]
        self.shuffled = True

    def head(self, val_prop=0.01, shuffle_on_iter=False, drop_last=False):
        common_kwargs = {
            "chunk_context": self.chunk_context,
            "max_seq_len": self.max_seq_len,
            "kmer_context_bases": self.kmer_context_bases,
            "base_pred": self.base_pred,
            "mod_bases": self.mod_bases,
            "mod_long_names": self.mod_long_names,
            "motifs": self.motifs,
            "store_read_data": self.store_read_data,
            "batch_size": self.batch_size,
        }
        val_trn_slice = int(val_prop * self.nchunks)
        val_trn_ds = RemoraDataset(
            self.sig_tensor[:val_trn_slice].copy(),
            self.seq_array[:val_trn_slice].copy(),
            self.seq_mappings[:val_trn_slice].copy(),
            self.seq_lens[:val_trn_slice].copy(),
            self.labels[:val_trn_slice].copy(),
            [self.read_data[idx] for idx in np.arange(val_trn_slice)]
            if self.read_data
            else None,
            shuffle_on_iter=shuffle_on_iter,
            drop_last=drop_last,
            **common_kwargs,
        )
        return val_trn_ds

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
        return Counter(self.labels[: self.nchunks])

    def split_data(self, val_prop, stratified=True):
        if not self.is_trimmed:
            raise RemoraError("Cannot split an untrimmed dataset.")
        elif self.sig_tensor.shape[0] < int(1 / val_prop) * 2:
            raise RemoraError(
                "Too few chunks to extract validation proportion "
                f"({self.sig_tensor.shape[0]} < {int(1 / val_prop) * 2})"
            )
        elif val_prop > 0.5:
            raise RemoraError(
                "Validation set size must be less than half of full training set"
            )
        common_kwargs = {
            "chunk_context": self.chunk_context,
            "max_seq_len": self.max_seq_len,
            "kmer_context_bases": self.kmer_context_bases,
            "base_pred": self.base_pred,
            "mod_bases": self.mod_bases,
            "mod_long_names": self.mod_long_names,
            "motifs": self.motifs,
            "store_read_data": self.store_read_data,
            "batch_size": self.batch_size,
        }
        if not self.shuffled:
            self.shuffle_dataset()
        val_idx = int(self.sig_tensor.shape[0] * val_prop)
        if stratified:
            class_counts = self.get_label_counts()
            class_num_vals = (
                val_prop * np.fromiter(class_counts.values(), dtype=int)
            ).astype(int)

            val_indices = []
            trn_indices = []

            for class_label in range(len(class_counts)):
                class_indices = np.where(self.labels == class_label)[0]
                np.random.shuffle(class_indices)
                val_indices.append(class_indices[: class_num_vals[class_label]])
                trn_indices.append(class_indices[class_num_vals[class_label] :])

            val_indices = np.concatenate(val_indices)
            trn_indices = np.concatenate(trn_indices)
        else:
            val_indices = np.arange(0, val_idx)
            trn_indices = np.arange(val_idx, self.sig_tensor.shape[0])

        val_ds = RemoraDataset(
            self.sig_tensor[val_indices],
            self.seq_array[val_indices],
            self.seq_mappings[val_indices],
            self.seq_lens[val_indices],
            self.labels[val_indices],
            [self.read_data[idx] for idx in val_indices]
            if self.store_read_data
            else None,
            shuffle_on_iter=False,
            drop_last=False,
            **common_kwargs,
        )
        trn_ds = RemoraDataset(
            self.sig_tensor[trn_indices],
            self.seq_array[trn_indices],
            self.seq_mappings[trn_indices],
            self.seq_lens[trn_indices],
            self.labels[trn_indices],
            [self.read_data[idx] for idx in trn_indices]
            if self.store_read_data
            else None,
            shuffle_on_iter=True,
            drop_last=False,
            **common_kwargs,
        )
        return trn_ds, val_ds

    def split_by_label(self):
        labels = "ACGT" if self.base_pred else self.mod_long_names
        label_datasets = []
        common_kwargs = {
            "chunk_context": self.chunk_context,
            "max_seq_len": self.max_seq_len,
            "kmer_context_bases": self.kmer_context_bases,
            "base_pred": self.base_pred,
            "mod_bases": self.mod_bases,
            "mod_long_names": self.mod_long_names,
            "motifs": self.motifs,
            "store_read_data": self.store_read_data,
            "batch_size": self.batch_size,
        }
        for int_label, label in enumerate(labels):
            if not self.base_pred:
                int_label += 1
            label_indices = np.equal(self.labels, int_label)
            label_datasets.append(
                (
                    label,
                    RemoraDataset(
                        self.sig_tensor[label_indices],
                        self.seq_array[label_indices],
                        self.seq_mappings[label_indices],
                        self.seq_lens[label_indices],
                        self.labels[label_indices],
                        self.read_data[label_indices]
                        if self.store_read_data
                        else None,
                        shuffle_on_iter=False,
                        drop_last=False,
                        **common_kwargs,
                    ),
                )
            )
        return label_datasets

    def balance_classes(self):

        min_class_len = min(self.get_label_counts().values())
        if self.base_pred:
            outs = 4
        else:
            outs = self.num_labels

        choices = []
        for i in range(outs):
            choices.append(
                np.random.choice(
                    np.where(self.labels == i)[0],
                    min_class_len,
                    replace=False,
                )
            )
        choices = np.concatenate(choices)
        b_rd = (
            None
            if self.read_data is None
            else [self.read_data[idx] for idx in choices]
        )

        balanced_dataset = RemoraDataset(
            sig_tensor=self.sig_tensor[choices],
            seq_array=self.seq_array[choices],
            seq_mappings=self.seq_mappings[choices],
            seq_lens=self.seq_lens[choices],
            labels=self.labels[choices],
            read_data=b_rd,
            nchunks=outs * min_class_len,
            chunk_context=self.chunk_context,
            max_seq_len=self.max_seq_len,
            kmer_context_bases=self.kmer_context_bases,
            base_pred=self.base_pred,
            mod_bases=self.mod_bases,
            mod_long_names=self.mod_long_names,
            motifs=self.motifs,
        )

        return balanced_dataset

    def filter(self, indices):

        if len(indices) > self.sig_tensor.shape[0]:
            raise RemoraError(
                "Filter indices cannot be longer than dataset size"
            )

        common_kwargs = {
            "chunk_context": self.chunk_context,
            "max_seq_len": self.max_seq_len,
            "kmer_context_bases": self.kmer_context_bases,
            "base_pred": self.base_pred,
            "mod_bases": self.mod_bases,
            "mod_long_names": self.mod_long_names,
            "motifs": self.motifs,
            "store_read_data": self.store_read_data,
            "batch_size": self.batch_size,
        }

        return RemoraDataset(
            self.sig_tensor[indices],
            self.seq_array[indices],
            self.seq_mappings[indices],
            self.seq_lens[indices],
            self.labels[indices],
            [self.read_data[idx] for idx in indices]
            if self.store_read_data
            else None,
            shuffle_on_iter=False,
            drop_last=False,
            **common_kwargs,
        )

    def add_fake_base(self, new_mod_long_names, new_mod_bases):
        if not set(self.mod_long_names).issubset(new_mod_long_names):
            raise RemoraError(
                "There are no mods in common between the model being traind "
                "and the external validation set."
            )
        for mod in self.mod_long_names:
            new_index = new_mod_long_names.index(mod) + 1
            old_index = self.mod_long_names.index(mod) + 1
            self.labels[np.where(self.labels == old_index)] = new_index
        self.mod_long_names = new_mod_long_names
        self.mod_bases = new_mod_bases

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
            mod_long_names=self.mod_long_names,
            motifs=[mot[0] for mot in self.motifs],
            motif_offset=[mot[1] for mot in self.motifs],
        )

    @property
    def can_base(self):
        return self.motifs[0][0][self.motifs[0][1]]

    @property
    def is_trimmed(self):
        return self.nchunks == self.sig_tensor.shape[0]

    @property
    def is_multiclass(self):
        return self.base_pred or len(self.mod_bases) > 1

    @property
    def num_motifs(self):
        return len(self.motifs)

    @property
    def num_labels(self):
        return len(self.mod_bases) + 1

    @classmethod
    def load_from_file(cls, filename, *args, **kwargs):
        # use allow_pickle=True to allow None type in read_data
        data = np.load(filename, allow_pickle=True)
        read_data = data["read_data"].tolist()
        chunk_context = tuple(data["chunk_context"].tolist())
        kmer_context_bases = tuple(data["kmer_context_bases"].tolist())
        base_pred = data["base_pred"].item()
        mod_bases = data["mod_bases"].item()
        mod_long_names = tuple(data["mod_long_names"].tolist())
        if isinstance(data["motif_offset"].tolist(), int):
            motifs = [
                (mot, mot_off)
                for mot, mot_off in zip(
                    [data["motif"].tolist()], [data["motif_offset"].tolist()]
                )
            ]
        else:
            motifs = [
                (mot, mot_off)
                for mot, mot_off in zip(
                    data["motifs"].tolist(), data["motif_offset"].tolist()
                )
            ]
        return cls(
            data["sig_tensor"],
            data["seq_array"],
            data["seq_mappings"],
            data["seq_lens"],
            data["labels"],
            read_data,
            chunk_context=chunk_context,
            kmer_context_bases=kmer_context_bases,
            base_pred=base_pred,
            mod_bases=mod_bases,
            mod_long_names=mod_long_names,
            motifs=motifs,
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
        max_seq_len=None,
        min_samps_per_base=None,
        store_read_data=False,
        *args,
        **kwargs,
    ):
        if max_seq_len is None and min_samps_per_base is None:
            raise RemoraError(
                "Must set either max_seq_len or min_samps_per_base"
            )
        if max_seq_len is None:
            max_seq_len = sum(chunk_context) // min_samps_per_base
        sig_tensor = np.empty(
            (num_chunks, 1, sum(chunk_context)), dtype=np.float32
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
        labels = np.empty(num_chunks, dtype=np.long)
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


def merge_datasets(input_datasets, balance=False):
    def load_dataset(ds_path, num_chunks):
        dataset = RemoraDataset.load_from_file(
            ds_path,
            shuffle_on_iter=False,
            drop_last=False,
        )
        if num_chunks < dataset.nchunks:
            dataset.shuffle_dataset()
        else:
            num_chunks = dataset.nchunks
        return dataset, num_chunks

    # load first file to determine base_pred or mod_bases
    datasets = [load_dataset(*input_datasets[0])]
    base_pred = datasets[0][0].base_pred
    chunk_context = datasets[0][0].chunk_context
    max_seq_len = datasets[0][0].max_seq_len
    kmer_context_bases = datasets[0][0].kmer_context_bases
    motifs = datasets[0][0].motifs
    for ds_path, num_chunks in input_datasets[1:]:
        datasets.append(load_dataset(ds_path, num_chunks))
        if datasets[-1][0].base_pred != base_pred:
            raise RemoraError("All datasets must be base-pred or modbase")
        if datasets[-1][0].chunk_context != chunk_context:
            raise RemoraError(
                "All datasets must have the same chunk_context "
                f"({datasets[-1][0].chunk_context} != {chunk_context})"
            )
        if datasets[-1][0].max_seq_len != max_seq_len:
            raise RemoraError(
                "All datasets must have the same max_seq_len "
                f"({datasets[-1][0].max_seq_len} != {max_seq_len})"
            )
        if datasets[-1][0].kmer_context_bases != kmer_context_bases:
            raise RemoraError(
                "All datasets must have the same kmer_context_bases "
                f"({datasets[-1][0].kmer_context_bases} != {kmer_context_bases})"
            )
        if datasets[-1][0].motifs != motifs:
            LOGGER.info(
                f"WARNING: Datasets have different motifs. Merging motifs {motifs} with motifs {datasets[-1][0].motifs}"
            )

            motifs = list(
                set(
                    motif for dataset in datasets for motif in dataset[0].motifs
                )
            )

    all_mod_bases = ""
    if base_pred:
        all_mod_long_names = None
    else:
        all_mod_long_names = []
        for ds, _ in datasets:
            for mod_base, mln in zip(ds.mod_bases, ds.mod_long_names):
                if mod_base not in all_mod_bases:
                    all_mod_bases += mod_base
                    all_mod_long_names.append(mln)

    total_chunks = sum(ds[1] for ds in datasets)
    output_dataset = RemoraDataset.allocate_empty_chunks(
        num_chunks=total_chunks,
        chunk_context=chunk_context,
        max_seq_len=max_seq_len,
        kmer_context_bases=kmer_context_bases,
        base_pred=base_pred,
        mod_bases=all_mod_bases,
        mod_long_names=all_mod_long_names,
        motifs=motifs,
    )
    for input_dataset, num_chunks in datasets:
        if base_pred:
            label_conv = np.arange(4, dtype=np.long)
        elif input_dataset.mod_bases == output_dataset.mod_bases:
            label_conv = np.arange(
                len(output_dataset.mod_bases) + 1, dtype=np.long
            )
        else:
            label_conv = np.empty(
                len(input_dataset.mod_bases) + 1, dtype=np.long
            )
            label_conv[0] = 0
            for input_lab, mod_base in enumerate(input_dataset.mod_bases):
                label_conv[input_lab + 1] = (
                    output_dataset.mod_bases.find(mod_base) + 1
                )
        added_chunks = 0
        for (
            (b_sig, b_seq, b_ss_map, b_seq_lens),
            b_labels,
            b_rd,
        ) in input_dataset:
            b_labels = label_conv[b_labels]
            if added_chunks + b_labels.size >= num_chunks:
                batch_size = num_chunks - (added_chunks + b_labels.size)
                b_rd = None if b_rd is None else b_rd[:batch_size]
                output_dataset.add_batch(
                    b_sig[:batch_size],
                    b_seq[:batch_size],
                    b_ss_map[:batch_size],
                    b_seq_lens[:batch_size],
                    b_labels[:batch_size],
                    b_rd,
                )
                break
            added_chunks += b_labels.size
            output_dataset.add_batch(
                b_sig, b_seq, b_ss_map, b_seq_lens, b_labels, b_rd
            )
        LOGGER.info(
            f"Copied {num_chunks} chunks. New label distribution: "
            f"{output_dataset.get_label_counts()}"
        )

    output_dataset.clip_chunks()
    if balance:
        balanced_dataset = output_dataset.balance_classes()
        LOGGER.info(
            f"Balanced out to {balanced_dataset.sig_tensor.shape[0]} chunks. New label distribution: "
            f"{balanced_dataset.get_label_counts()}"
        )
        return balanced_dataset

    return output_dataset
