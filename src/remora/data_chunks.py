import gc
import re
import os
from collections import Counter
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm

from remora.refine_signal_map import SigMapRefiner
from remora import constants, log, RemoraError, util, encoded_kmers

LOGGER = log.get_logger()

DATASET_VERSION = 2
MISMATCH_ARRS = {
    0: np.array([1, 2, 3]),
    1: np.array([0, 2, 3]),
    2: np.array([0, 1, 3]),
    3: np.array([0, 1, 2]),
}
# CIGAR operations which correspond to query and reference sequence
MATCH_OPS = np.array(
    [True, False, False, False, False, False, False, True, True]
)
QUERY_OPS = np.array([True, True, False, False, True, False, False, True, True])
REF_OPS = np.array([True, False, True, True, False, False, False, True, True])
CIGAR_CODES = ["M", "I", "D", "N", "S", "H", "P", "=", "X"]
CODE_TO_OP = {
    "M": 0,
    "I": 1,
    "D": 2,
    "N": 3,
    "S": 4,
    "H": 5,
    "P": 6,
    "=": 7,
    "X": 8,
}
CIGAR_STRING_PATTERN = re.compile(r"(\d+)" + f"([{''.join(CIGAR_CODES)}])")


def cigartuples_from_string(cigarstring):
    """
    Returns pysam-style list of (op, count) tuples from a cigarstring.
    """
    return [
        (CODE_TO_OP[m.group(2)], int(m.group(1)))
        for m in re.finditer(CIGAR_STRING_PATTERN, cigarstring)
    ]


def map_ref_to_signal(*, query_to_signal, ref_to_query_knots):
    """Compute interpolated mapping from reference, through query alignment to
    signal coordinates

    Args:
        query_to_signal (np.array): Query to signal coordinate mapping
        ref_to_query_knots (np.array): Reference to query coordinate mapping
    """
    return np.floor(
        np.interp(
            ref_to_query_knots,
            np.arange(query_to_signal.size),
            query_to_signal,
        )
    ).astype(int)


def make_sequence_coordinate_mapping(cigar):
    """Maps an element in reference to every element in basecalls using
    alignment in `cigar`.

    Args:
        cigar (list): "cigartuples" representing alignment

    Returns:
        array shape (ref_len,). [x_0, x_1, ..., x_(ref_len)]
            such that read_seq[x_i] <> ref_seq[i]. Note that ref_len is derived
            from the cigar input.
    """
    ops, lens = map(np.array, zip(*cigar))
    assert ops.min() >= 0 and ops.max() <= 8, "invalid cigar op(s)"
    assert lens.min() >= 0, "cigar lengths may not be negative"

    is_match = MATCH_OPS[ops]
    match_counts = lens[is_match]
    offsets = np.array([match_counts, np.ones_like(match_counts)])

    # TODO remove knots around ambiguous indels (e.g. left justified HPs)
    # note this requires the ref and query sequences
    ref_knots = np.cumsum(np.where(REF_OPS[ops], lens, 0))
    ref_knots = np.concatenate(
        [[0], (ref_knots[is_match] - offsets).T.flatten(), [ref_knots[-1]]]
    )
    query_knots = np.cumsum(np.where(QUERY_OPS[ops], lens, 0))
    query_knots = np.concatenate(
        [[0], (query_knots[is_match] - offsets).T.flatten(), [query_knots[-1]]]
    )
    knots = np.interp(np.arange(ref_knots[-1] + 1), ref_knots, query_knots)

    return knots


def compute_ref_to_signal(query_to_signal, cigar):
    ref_to_read_knots = make_sequence_coordinate_mapping(cigar)
    assert query_to_signal.size == ref_to_read_knots[-1].astype(int) + 1, (
        "discordant implied basecall lengths: move_table:"
        f"{query_to_signal.size} cigar:{ref_to_read_knots[-1]}"
    )
    return map_ref_to_signal(
        query_to_signal=query_to_signal, ref_to_query_knots=ref_to_read_knots
    )


@dataclass
class RemoraRead:
    """Object to hold information about a read relevant to Remora training and
    inference.

    Args:
        dacs (np.ndarray): Unnormalized DAC signal. `dacs` should be reversed
            already for reverse_signal data types.
        shift (float): Shift from dac to normalized signal. via formula:
            norm = (dac - shift) / scale
        scale (float): Scale from dac to normalized signal
        seq_to_sig_map (np.ndarray): Position within signal array assigned to
            each base in seq
        int_seq (np.ndarray): Encoded sequence for training/validation.
            See remora.util.seq_to_int
        str_seq (str): String sequence for training/validation. Ignored if
            int_seq is provided.
        read_id (str): Read identifier
        labels (np.ndarray): Output label for each base in read
        focus_bases (np.ndarray): Sites from read to produce calls
        batches (list): List of batches from RemoraDataset

    Note: Must provide either int_seq or str_seq. If str_seq is provided
    int_seq will be derived on init.
    """

    dacs: np.ndarray
    shift: float
    scale: float
    seq_to_sig_map: np.ndarray
    int_seq: np.ndarray = None
    str_seq: str = None
    read_id: str = None
    labels: np.ndarray = None
    focus_bases: np.ndarray = None
    batches: list = None

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
        self._sig = None
        self._dwells = None
        self._sig_cumsum = None
        self._base_levels = None

    @classmethod
    def test_read(cls, nbases=20, signal_per_base=10):
        """Spoofed read for testing"""
        return cls(
            np.zeros(nbases * signal_per_base),
            0.0,
            1.0,
            np.arange(nbases * signal_per_base + 1, step=signal_per_base),
            np.arange(nbases) % 4,
            "test_read",
            np.zeros(nbases, dtype=np.int64),
        )

    @property
    def sig(self):
        if self._sig is None:
            self._sig = (self.dacs - self.shift) / self.scale
        return self._sig

    @property
    def sig_cumsum(self):
        if self._sig_cumsum is None:
            self._sig_cumsum = np.empty(self.sig.size + 1)
            self._sig_cumsum[0] = 0
            self._sig_cumsum[1:] = np.cumsum(self.sig)
        return self._sig_cumsum

    @property
    def dwells(self):
        if self._dwells is None:
            self._dwells = np.diff(self.seq_to_sig_map)
        return self._dwells

    @property
    def base_levels(self):
        if self._base_levels is None:
            with np.errstate(invalid="ignore"):
                self._base_levels = (
                    np.diff(self.sig_cumsum[self.seq_to_sig_map]) / self.dwells
                )
        return self._base_levels

    def check(self):
        if self.seq_to_sig_map.size != self.int_seq.size + 1:
            LOGGER.debug(
                "Invalid read: seq and mapping mismatch "
                f"{self.seq_to_sig_map.size} != {self.int_seq.size + 1}"
            )
            raise RemoraError(
                f"Invalid read: seq ({self.int_seq.size}) and mapping "
                f"({self.seq_to_sig_map.size}) sizes incompatible"
            )
        if self.seq_to_sig_map[0] != 0:
            LOGGER.debug(
                f"Invalid read {self.read_id} : invalid mapping start "
                f"{self.seq_to_sig_map[0]} != 0"
            )
            raise RemoraError("Invalid read: mapping start")
        if self.seq_to_sig_map[-1] != self.sig.size:
            LOGGER.debug(
                f"Invalid read {self.read_id} : invalid mapping end "
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

    def copy(self):
        return RemoraRead(
            dacs=self.dacs.copy(),
            shift=self.shift,
            scale=self.scale,
            seq_to_sig_map=self.seq_to_sig_map,
            int_seq=None if self.int_seq is None else self.int_seq.copy(),
            str_seq=self.str_seq,
            read_id=self.read_id,
            labels=None if self.labels is None else self.labels.copy(),
            focus_bases=None
            if self.focus_bases is None
            else self.focus_bases.copy(),
        )

    def refine_signal_mapping(self, sig_map_refiner, check_read=False):
        if not sig_map_refiner.is_loaded:
            LOGGER.debug("no signal map refiner loaded skipping..")
            return
        if sig_map_refiner.do_rough_rescale:
            prev_shift, prev_scale = self.shift, self.scale
            self.shift, self.scale = sig_map_refiner.rough_rescale(
                self.shift,
                self.scale,
                self.seq_to_sig_map,
                self.int_seq,
                self.dacs,
            )
            self._sig = None
            self._sig_cumsum = None
            self._base_levels = None
        if sig_map_refiner.scale_iters >= 0:
            prev_shift, prev_scale = self.shift, self.scale
            prev_seq_to_sig_map = self.seq_to_sig_map.copy()
            try:
                (
                    self.seq_to_sig_map,
                    self.shift,
                    self.scale,
                ) = sig_map_refiner.refine_sig_map(
                    self.shift,
                    self.scale,
                    self.seq_to_sig_map,
                    self.int_seq,
                    self.dacs,
                )
            except IndexError as e:
                LOGGER.debug(f"refine_error {self.read_id} {e}")
            # reset computed values after refinement
            self._sig = None
            self._dwells = None
            self._sig_cumsum = None
            self._base_levels = None
            LOGGER.debug(f"refine_mapping_shift: {prev_shift} {self.shift}")
            LOGGER.debug(f"refine_mapping_scale: {prev_scale} {self.scale}")
            sig_map_diffs = self.seq_to_sig_map - prev_seq_to_sig_map
            LOGGER.debug(
                f"refine_mapping_median_adjust: {np.median(sig_map_diffs)} "
                f"{self.read_id}"
            )
        if check_read:
            self.check()

    def set_motif_focus_bases(self, motifs):
        """
        Mutates self. Sets self.focus_bases to all hits within self.int_seq.
        :param motifs: Iterable of util.Motifs
        """
        self.focus_bases = util.find_focus_bases_in_int_sequence(
            self.int_seq, motifs
        )

    def downsample_focus_bases(self, max_sites):
        if self.focus_bases is not None and self.focus_bases.size > max_sites:
            LOGGER.debug(
                f"selected {max_sites} focus bases from "
                f"{self.focus_bases.size} in read {self.read_id}"
            )
            self.focus_bases = np.random.choice(
                self.focus_bases,
                size=max_sites,
                replace=False,
            )

    def extract_chunk(
        self,
        focus_sig_idx,
        chunk_context,
        kmer_context_bases,
        label=-1,
        base_pred=False,
        read_focus_base=-1,
        check_chunk=False,
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

        seq_start = (
            np.searchsorted(self.seq_to_sig_map, sig_start, side="right") - 1
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
            signal=chunk_sig,
            seq_w_context=chunk_seq,
            seq_to_sig_map=chunk_seq_to_sig,
            kmer_context_bases=kmer_context_bases,
            chunk_sig_focus_idx=focus_sig_idx - sig_start,
            chunk_focus_base=read_focus_base - seq_start,
            read_focus_base=read_focus_base,
            read_id=self.read_id,
            label=label,
        )
        if check_chunk:
            chunk.check()
        if base_pred:
            chunk.mask_focus_base()
        return chunk

    def iter_chunks(
        self,
        chunk_context,
        kmer_context_bases,
        base_pred=False,
        base_start_justify=False,
        offset=0,
        check_chunks=False,
    ):
        for focus_base in self.focus_bases:
            label = -1 if self.labels is None else self.labels[focus_base]
            # add offset and ensure not out of bounds
            focus_base = max(
                min(focus_base + offset, self.seq_to_sig_map.size - 2), 0
            )
            if base_start_justify:
                focus_sig_idx = self.seq_to_sig_map[focus_base]
            else:
                # compute position at center of central base
                focus_sig_idx = (
                    self.seq_to_sig_map[focus_base]
                    + self.seq_to_sig_map[focus_base + 1]
                ) // 2
            try:
                yield self.extract_chunk(
                    focus_sig_idx,
                    chunk_context,
                    kmer_context_bases,
                    label=label,
                    base_pred=base_pred,
                    read_focus_base=focus_base,
                    check_chunk=check_chunks,
                )
            except RemoraError as e:
                LOGGER.debug(f"FAILED_CHUNK_CHECK {e}")
            except Exception as e:
                LOGGER.debug(f"FAILED_CHUNK_EXTRACT {e}")

    def prepare_batches(self, model_metadata, batch_size):
        """Prepare batches containing chunks from this read

        Args:
            model_metadata: Inference model metadata
            batch_size (int): Number of chunks to call per-batch
        """
        self.batches = []
        self.refine_signal_mapping(model_metadata["sig_map_refiner"])
        chunks = list(
            self.iter_chunks(
                model_metadata["chunk_context"],
                model_metadata["kmer_context_bases"],
                model_metadata["base_pred"],
                model_metadata["base_start_justify"],
                model_metadata["offset"],
            )
        )
        if len(chunks) == 0:
            return
        dataset = RemoraDataset.allocate_empty_chunks(
            num_chunks=len(chunks),
            chunk_context=model_metadata["chunk_context"],
            max_seq_len=max(c.seq_len for c in chunks),
            kmer_context_bases=model_metadata["kmer_context_bases"],
            base_pred=model_metadata["base_pred"],
            mod_bases=model_metadata["mod_bases"],
            mod_long_names=model_metadata["mod_long_names"],
            batch_size=batch_size,
            shuffle_on_iter=False,
            drop_last=False,
        )
        for chunk in chunks:
            dataset.add_chunk(chunk)
        dataset.set_nbatches()

        bb, ab = model_metadata["kmer_context_bases"]
        for (sigs, seqs, seq_maps, seq_lens), labels, (_, read_pos) in dataset:
            enc_kmers = encoded_kmers.compute_encoded_kmer_batch(
                bb, ab, seqs, seq_maps, seq_lens
            )
            self.batches.append((sigs, enc_kmers, labels, read_pos))

    def run_model(self, model):
        """Call modified bases on a read.

        Args:
            model: Compiled inference model (see remora.model_util.load_model)

        Returns:
            3-tuple containing:
              1. Modified base predictions (dim: num_calls, num_mods + 1)
              2. Labels for each base (-1 if labels not provided)
              3. List of positions within the read
        """
        device = next(model.parameters()).device
        read_outputs, read_poss, read_labels = [], [], []
        for sigs, enc_kmers, labels, read_pos in self.batches:
            sigs = torch.from_numpy(sigs).to(device)
            enc_kmers = torch.from_numpy(enc_kmers).to(device)
            output = model(sigs, enc_kmers).detach().cpu().numpy()
            read_outputs.append(output)
            read_labels.append(labels)
            read_poss.append(read_pos)
        read_outputs = np.concatenate(read_outputs, axis=0)
        read_labels = np.concatenate(read_labels)
        read_poss = np.concatenate(read_poss)
        return read_outputs, read_labels, read_poss


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
        chunk_sig_focus_idx (int): Index within signal array on which the chunk
            is focuesed for prediction. May be used in model architecture in the
            future.
        chunk_focus_base (int): Index within chunk sequence (without context
            bases) on which the chunk is focuesed for prediction.
        read_focus_base (int): Position within full read for validation purposes
        read_id (str): Read ID
        label (int): Integer label for training/validation.
    """

    signal: np.ndarray
    seq_w_context: np.ndarray
    seq_to_sig_map: np.ndarray
    kmer_context_bases: tuple
    chunk_sig_focus_idx: int
    chunk_focus_base: int
    read_focus_base: int
    read_id: str = None
    label: int = None
    _base_sig_lens: np.ndarray = None

    def mask_focus_base(self):
        self.seq_w_context[
            self.chunk_focus_base + self.kmer_context_bases[0]
        ] = -1

    def check(self):
        if self.signal.size <= 0:
            LOGGER.debug(
                f"FAILED_CHUNK: no_sig {self.read_id} {self.read_focus_base}"
            )
            raise RemoraError("No signal for chunk")
        if (
            self.seq_w_context.size - sum(self.kmer_context_bases)
            != self.seq_to_sig_map.size - 1
        ):
            LOGGER.debug(
                f"FAILED_CHUNK: map_len {self.read_id} {self.read_focus_base}"
            )
            raise RemoraError("Invalid sig to seq map length")
        if not np.all(np.diff(self.seq_to_sig_map) >= 0):
            LOGGER.debug(
                f"FAILED_CHUNK: not monotonic {self.read_id} "
                f"{self.seq_to_sig_map}"
            )
        if self.seq_to_sig_map[0] < 0:
            LOGGER.debug(
                f"FAILED_CHUNK: start<0 {self.read_id} {self.seq_to_sig_map[0]}"
            )
            raise RemoraError("Seq to sig map starts before 0")
        if self.seq_to_sig_map[-1] > self.signal.size:
            LOGGER.debug(
                f"FAILED_CHUNK: end>sig_len {self.read_id} "
                f"{self.seq_to_sig_map[-1]}"
            )
            raise RemoraError("Seq to sig map ends after signal")
        # TODO add more logic to these checks

    @property
    def kmer_len(self):
        return sum(self.kmer_context_bases) + 1

    @property
    def seq_len(self):
        return self.seq_w_context.size - sum(self.kmer_context_bases)

    @property
    def seq(self):
        return self.seq_w_context[
            self.kmer_context_bases[0] : self.kmer_context_bases[0]
            + self.seq_len
        ]

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
        labels: (np.array): Chunk labels
            dtype : np.int64
            dims  : nchunks
        read_ids (np.array): Read IDs
            dtype : "U36"
            dims  : nchunks
        read_focus_bases (np.array): Base position within the read. This may be
            the basecalled sequence or the read-oriented reference sequence
            position depending on the extraction method.
            dtype : "U36"
            dims  : nchunks
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
        motifs (list): Tuples containing sequence motifs and relative position.
            E.g. ("N", 0) for all-contexts or ("CG", 0) for CG-contexts.
        reverse_signal (bool): Is nanopore signal 3' to 5' orientation?
            Primarily for directRNA
        batch_size (int): Size of batches to be produced
        shuffle_on_iter (bool): Shuffle data before each iteration over batches
        drop_last (bool): Drop the last batch of each iteration
        batch_label_props (np.array): Select proportion of each label
        sig_map_refiner (remora.refine_signal_map.SigMapRefiner): Signal
            mapping refiner
        base_start_justify (bool): Extract chunk centered on start of base
        offset (int): Extract chunk centered on base offset from base of
            interest
        drop_read_attrs (bool): Drop read id and read focus positions

    Yields:
        Batches of data for training or inference
    """

    # data attributes
    sig_tensor: np.ndarray
    seq_array: np.ndarray
    seq_mappings: np.ndarray
    seq_lens: np.ndarray
    labels: np.ndarray
    read_ids: np.ndarray
    read_focus_bases: np.ndarray
    nchunks: int = None

    # scalar metadata attributes
    chunk_context: tuple = constants.DEFAULT_CHUNK_CONTEXT
    max_seq_len: int = None
    kmer_context_bases: tuple = constants.DEFAULT_KMER_CONTEXT_BASES
    base_pred: bool = False
    mod_bases: str = ""
    mod_long_names: list = None
    motifs: list = None
    reverse_signal: bool = False

    # batch attributes (defaults set for training)
    batch_size: int = constants.DEFAULT_BATCH_SIZE
    shuffle_on_iter: bool = True
    drop_last: bool = True
    batch_label_props: np.ndarray = None

    # signal mapping refinement params
    sig_map_refiner: SigMapRefiner = None

    # chunk extraction attributes
    base_start_justify: bool = False
    offset: int = 0

    # extra attributes
    drop_read_attrs: bool = False

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
        if self.motifs is not None:
            self.motifs = sorted(set(self.motifs))
        self.set_nbatches()
        self.shuffled = False
        if self.drop_read_attrs:
            self.read_ids = None
            self.read_focus_bases = None
            gc.collect()
        self._class_ordered_indices = None

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
        if not self.drop_read_attrs:
            self.read_ids[self.nchunks] = chunk.read_id
            self.read_focus_bases[self.nchunks] = chunk.read_focus_base
        self.nchunks += 1

    def add_batch(
        self, b_sig, b_seq, b_ss_map, b_seq_lens, b_labels, b_rids, b_rfbs
    ):
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
        if not self.drop_read_attrs:
            self.read_ids[b_st:b_en] = b_rids
            self.read_focus_bases[b_st:b_en] = b_rfbs
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
        if not self.drop_read_attrs:
            self.read_ids = self.read_ids[: self.nchunks]
            self.read_focus_bases = self.read_focus_bases[: self.nchunks]
        # reset nbatches after clipping dataset
        self.set_nbatches()

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

    def get_label_nchunks(self, target_nchunks):
        """Get the number of chunks for each label most closely matching
        specified batch_label_props,
        """
        if self.batch_label_props is None:
            raise RemoraError("Must set batch_label_props to get label nchunks")
        lab_nchunks = np.floor(target_nchunks * self.batch_label_props).astype(
            int
        )
        while lab_nchunks.sum() < target_nchunks:
            lab_nchunks[
                np.argmin(
                    (lab_nchunks / lab_nchunks.sum()) - self.batch_label_props
                )
            ] += 1
        return lab_nchunks

    def reorder_chunks(self, indices):
        """Re-order chunks via specified indices"""
        for attr in (
            "sig_tensor",
            "seq_array",
            "seq_mappings",
            "seq_lens",
            "labels",
        ):
            setattr(self, attr, getattr(self, attr)[indices])
            gc.collect()
        if not self.drop_read_attrs:
            self.read_ids = self.read_ids[indices]
            self.read_focus_bases = self.read_focus_bases[indices]
        gc.collect()

    def order_chunks_by_batch_labels(self):
        """Order chunks such that each slice of size self.batch_size includes
        the specified proportion of each label. Also store the indices
        cooresponding to each label to facilitate shuffling within each label
        while maintaining specified batch balance.

        Note that "extra" chunks are stored at the end of the dataset and will
        be shuffled into the used portion of the dataset at each call to
        shuffle_by_batch_labels.
        """
        class_indices = [
            np.where(self.labels == class_label)[0]
            for class_label in range(self.num_labels)
        ]
        batch_nchunks = self.get_label_nchunks(self.batch_size)
        class_nbatches = [
            ci.size // bnc for ci, bnc in zip(class_indices, batch_nchunks)
        ]
        # set to number of batches given smallest class size
        self.nbatches = min(class_nbatches)
        if self.nbatches < 1:
            raise RemoraError("Ordering by batch labels produced no batches")
        # set indices for each label within each batch
        idx_starts = np.concatenate([[0], np.cumsum(batch_nchunks)])
        class_extras = [
            ci.size - (bnc * self.nbatches)
            for ci, bnc in zip(class_indices, batch_nchunks)
        ]
        LOGGER.debug("Computing class order indices")
        # store indcies for each class to order all chunks and save to shuffle
        # within class each epoch
        self._class_ordered_indices = []
        extras_index = self.batch_size * self.nbatches
        for lab in range(self.num_labels):
            lab_b_rng = np.arange(idx_starts[lab], idx_starts[lab + 1])
            self._class_ordered_indices.append(
                np.concatenate(
                    [
                        lab_b_rng + (bn * self.batch_size)
                        for bn in range(self.nbatches)
                    ]
                    + [
                        np.arange(
                            extras_index, extras_index + class_extras[lab]
                        )
                    ]
                )
            )
            extras_index += class_extras[lab]
        global_ordered_indices = np.empty(self.nchunks, dtype=int)
        for ci, coi in zip(class_indices, self._class_ordered_indices):
            global_ordered_indices[coi] = ci
        gc.collect()
        LOGGER.debug("Reordering chunks")
        # order each tensor to batch ordering
        self.reorder_chunks(global_ordered_indices)

    def shuffle_by_batch_labels(self):
        if self._class_ordered_indices is None:
            raise RemoraError(
                "Must run order_chunks_by_batch_labels before "
                "shuffle_by_batch_labels"
            )
        global_ordered_indices = np.empty(self.nchunks, dtype=int)
        for coi in self._class_ordered_indices:
            global_ordered_indices[coi] = np.random.permutation(coi)
        self.reorder_chunks(global_ordered_indices)
        self.shuffled = True

    def shuffle(self):
        if not self.is_clipped:
            raise RemoraError("Cannot shuffle an unclipped dataset.")
        if self.batch_label_props is not None:
            raise RemoraError(
                "Cannot shuffle dataset with batch_label_props. "
                "Use shuffle_by_batch_labels"
            )
        shuf_idx = np.random.permutation(self.nchunks)
        self.reorder_chunks(shuf_idx)
        self.shuffled = True

    def clone_metadata(self):
        return {
            "chunk_context": self.chunk_context,
            "max_seq_len": self.max_seq_len,
            "kmer_context_bases": self.kmer_context_bases,
            "base_pred": self.base_pred,
            "mod_bases": self.mod_bases,
            "mod_long_names": self.mod_long_names,
            "motifs": self.motifs,
            "reverse_signal": self.reverse_signal,
            "batch_size": self.batch_size,
            "sig_map_refiner": self.sig_map_refiner,
            "offset": self.offset,
            "base_start_justify": self.base_start_justify,
        }

    def head(
        self,
        nchunks=None,
        prop=0.01,
        shuffle_on_iter=False,
        drop_last=False,
        stratified=False,
    ):
        if nchunks is None:
            nchunks = int(prop * self.nchunks)
        read_ids = read_focus_bases = None
        if stratified:
            LOGGER.debug("Performing stratified head")
            head_indices = []
            for class_label in range(self.num_labels):
                class_indices = np.where(self.labels == class_label)[0]
                if class_indices.size == 0:
                    continue
                cls_val_idx = int(class_indices.size * nchunks / self.nchunks)
                head_indices.append(class_indices[:cls_val_idx])

            head_indices = np.concatenate(head_indices)
            if not self.drop_read_attrs:
                read_ids = self.read_ids[head_indices].copy()
                read_focus_bases = self.read_focus_bases[head_indices].copy()
            return RemoraDataset(
                self.sig_tensor[head_indices].copy(),
                self.seq_array[head_indices].copy(),
                self.seq_mappings[head_indices].copy(),
                self.seq_lens[head_indices].copy(),
                self.labels[head_indices].copy(),
                read_ids,
                read_focus_bases,
                shuffle_on_iter=shuffle_on_iter,
                drop_last=drop_last,
                drop_read_attrs=self.drop_read_attrs,
                **self.clone_metadata(),
            )

        if not self.drop_read_attrs:
            read_ids = self.read_ids[:nchunks].copy()
            read_focus_bases = self.read_focus_bases[:nchunks].copy()
        return RemoraDataset(
            self.sig_tensor[:nchunks].copy(),
            self.seq_array[:nchunks].copy(),
            self.seq_mappings[:nchunks].copy(),
            self.seq_lens[:nchunks].copy(),
            self.labels[:nchunks].copy(),
            read_ids,
            read_focus_bases,
            shuffle_on_iter=shuffle_on_iter,
            drop_last=drop_last,
            drop_read_attrs=self.drop_read_attrs,
            **self.clone_metadata(),
        )

    def copy(self):
        read_ids = read_focus_bases = None
        if not self.drop_read_attrs:
            read_ids = self.read_ids.copy()
            read_focus_bases = self.read_focus_bases.copy()
        return RemoraDataset(
            self.sig_tensor.copy(),
            self.seq_array.copy(),
            self.seq_mappings.copy(),
            self.seq_lens.copy(),
            self.labels.copy(),
            read_ids,
            read_focus_bases,
            shuffle_on_iter=self.shuffle_on_iter,
            drop_last=self.drop_last,
            **self.clone_metadata(),
        )

    def __iter__(self):
        if self.batch_label_props is not None:
            self.shuffle_by_batch_labels()
        elif self.shuffle_on_iter:
            self.shuffle()
        self._batch_i = 0
        return self

    def __next__(self):
        if self._batch_i >= self.nbatches:
            raise StopIteration
        self._batch_i += 1
        b_st = (self._batch_i - 1) * self.batch_size
        b_en = b_st + self.batch_size
        read_ids = read_focus_bases = None
        if not self.drop_read_attrs:
            read_ids = self.read_ids[b_st:b_en]
            read_focus_bases = self.read_focus_bases[b_st:b_en]
        return (
            (
                self.sig_tensor[b_st:b_en],
                self.seq_array[b_st:b_en],
                self.seq_mappings[b_st:b_en],
                self.seq_lens[b_st:b_en],
            ),
            self.labels[b_st:b_en],
            (
                read_ids,
                read_focus_bases,
            ),
        )

    def __len__(self):
        return self.nbatches

    def get_label_counts(self):
        return Counter(self.labels[: self.nchunks])

    def split_data(
        self,
        *,
        val_prop=None,
        val_num=None,
        stratified=True,
    ):
        if val_prop is None and val_num is None:
            raise RemoraError(
                "Must supply either val_prop or val_num to split dataset"
            )
        if val_prop is not None and val_num is not None:
            LOGGER.warning(
                "val_prop and val_num supplied to split. Using val_num"
            )
            val_prop = None
        if not self.is_clipped:
            raise RemoraError("Cannot split an unclipped dataset.")
        if val_prop is not None:
            if val_prop > 0.5:
                raise RemoraError("val_prop > 0.5")
            if val_prop < 0:
                raise RemoraError("val_prop < 0")
            val_num = int(self.nchunks * val_prop)
        if val_num > self.nchunks:
            raise RemoraError("Too many validation chunks for split")
        if val_num > self.nchunks // 2:
            LOGGER.warning("Val split contains more than half of chunks")

        if not self.shuffled:
            self.shuffle()

        if stratified:
            LOGGER.debug("Performing stratified split")
            val_indices = []
            trn_indices = []
            for class_label in range(self.num_labels):
                class_indices = np.where(self.labels == class_label)[0]
                if class_indices.size == 0:
                    continue
                np.random.shuffle(class_indices)
                cls_val_idx = int(class_indices.size * val_num / self.nchunks)
                val_indices.append(class_indices[:cls_val_idx])
                trn_indices.append(class_indices[cls_val_idx:])

            val_indices = np.concatenate(val_indices)
            trn_indices = np.concatenate(trn_indices)
        else:
            val_indices = np.arange(0, val_num)
            trn_indices = np.arange(val_num, self.sig_tensor.shape[0])

        LOGGER.debug("Extracting validation dataset")
        val_read_ids = val_read_focus_bases = None
        if not self.drop_read_attrs:
            val_read_ids = self.read_ids[val_indices]
            val_read_focus_bases = self.read_focus_bases[val_indices]
        val_ds = RemoraDataset(
            self.sig_tensor[val_indices],
            self.seq_array[val_indices],
            self.seq_mappings[val_indices],
            self.seq_lens[val_indices],
            self.labels[val_indices],
            val_read_ids,
            val_read_focus_bases,
            shuffle_on_iter=False,
            drop_last=False,
            drop_read_attrs=self.drop_read_attrs,
            **self.clone_metadata(),
        )
        LOGGER.debug("Extracting train dataset")
        trn_read_ids = trn_read_focus_bases = None
        if not self.drop_read_attrs:
            trn_read_ids = self.read_ids[trn_indices]
            trn_read_focus_bases = self.read_focus_bases[trn_indices]
        trn_ds = RemoraDataset(
            self.sig_tensor[trn_indices],
            self.seq_array[trn_indices],
            self.seq_mappings[trn_indices],
            self.seq_lens[trn_indices],
            self.labels[trn_indices],
            trn_read_ids,
            trn_read_focus_bases,
            shuffle_on_iter=True,
            drop_last=False,
            drop_read_attrs=self.drop_read_attrs,
            batch_label_props=self.batch_label_props,
            **self.clone_metadata(),
        )
        return trn_ds, val_ds

    def split_by_label(self):
        if self.base_pred:
            labels = "ACGT"
        else:
            labels = ["control"] + self.mod_long_names
        label_datasets = []
        for int_label, label in enumerate(labels):
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
                        None
                        if self.drop_read_attrs
                        else self.read_ids[label_indices],
                        None
                        if self.drop_read_attrs
                        else self.read_focus_bases[label_indices],
                        shuffle_on_iter=False,
                        drop_last=False,
                        **self.clone_metadata(),
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

        return RemoraDataset(
            sig_tensor=self.sig_tensor[choices],
            seq_array=self.seq_array[choices],
            seq_mappings=self.seq_mappings[choices],
            seq_lens=self.seq_lens[choices],
            labels=self.labels[choices],
            read_ids=None if self.drop_read_attrs else self.read_ids[choices],
            read_focus_bases=None
            if self.drop_read_attrs
            else self.read_focus_bases[choices],
            nchunks=outs * min_class_len,
            **self.clone_metadata(),
        )

    def filter(self, indices):
        if len(indices) > self.sig_tensor.shape[0]:
            raise RemoraError(
                "Filter indices cannot be longer than dataset size"
            )
        return RemoraDataset(
            self.sig_tensor[indices],
            self.seq_array[indices],
            self.seq_mappings[indices],
            self.seq_lens[indices],
            self.labels[indices],
            None if self.drop_read_attrs else self.read_ids[indices],
            None if self.drop_read_attrs else self.read_focus_bases[indices],
            shuffle_on_iter=False,
            drop_last=False,
            **self.clone_metadata(),
        )

    def add_fake_base(self, new_mod_long_names, new_mod_bases):
        if not set(self.mod_long_names).issubset(new_mod_long_names):
            raise RemoraError(
                "There are no mods in common between the model being trained "
                f"({new_mod_long_names}) and the external validation set "
                f"({self.mod_long_names})."
            )
        for mod in self.mod_long_names:
            new_index = new_mod_long_names.index(mod) + 1
            old_index = self.mod_long_names.index(mod) + 1
            self.labels[np.where(self.labels == old_index)] = new_index
        self.mod_long_names = new_mod_long_names
        self.mod_bases = new_mod_bases

    def save(self, filename):
        self.clip_chunks()
        np.savez(
            filename,
            sig_tensor=self.sig_tensor,
            seq_array=self.seq_array,
            seq_mappings=self.seq_mappings,
            seq_lens=self.seq_lens,
            labels=self.labels,
            read_ids=None if self.drop_read_attrs else self.read_ids,
            read_focus_bases=None
            if self.drop_read_attrs
            else self.read_focus_bases,
            chunk_context=self.chunk_context,
            kmer_context_bases=self.kmer_context_bases,
            base_pred=self.base_pred,
            mod_bases=self.mod_bases,
            mod_long_names=self.mod_long_names,
            motifs=[mot[0] for mot in self.motifs],
            motif_offset=[mot[1] for mot in self.motifs],
            reverse_signal=int(self.reverse_signal),
            base_start_justify=int(self.base_start_justify),
            offset=self.offset,
            version=DATASET_VERSION,
            **self.sig_map_refiner.get_save_kwargs(),
        )

    @classmethod
    def load_from_file(cls, filename, *args, **kwargs):
        # use allow_pickle=True to allow None type in read_data
        data = np.load(filename, allow_pickle=True)
        try:
            version = int(data["version"].item())
        except KeyError:
            version = None
        if version is None or version != DATASET_VERSION:
            raise RemoraError(
                f"Remora dataset version ({version}) does "
                f"not match current distribution ({DATASET_VERSION})"
            )
        chunk_context = tuple(data["chunk_context"].tolist())
        kmer_context_bases = tuple(data["kmer_context_bases"].tolist())
        base_pred = data["base_pred"].item()
        try:
            mod_bases = data["mod_bases"].item()
        except ValueError:
            mod_bases = ""
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
        sig_map_refiner = SigMapRefiner.load_from_np_savez(data)
        base_start_justify = bool(int(data["base_start_justify"].item()))
        # default to reverse_signal=False for datasets without this attribute
        reverse_signal = bool(int(data.get("reverse_signal", 0)))
        offset = int(data["offset"].item())
        return cls(
            data["sig_tensor"],
            data["seq_array"],
            data["seq_mappings"],
            data["seq_lens"],
            data["labels"],
            data["read_ids"],
            data["read_focus_bases"],
            chunk_context=chunk_context,
            kmer_context_bases=kmer_context_bases,
            base_pred=base_pred,
            mod_bases=mod_bases,
            mod_long_names=mod_long_names,
            motifs=motifs,
            reverse_signal=reverse_signal,
            sig_map_refiner=sig_map_refiner,
            base_start_justify=base_start_justify,
            offset=offset,
            *args,
            **kwargs,
        )

    def perturb_seq_mismatch(self, mm_rate):
        errs_added = 0
        for c_idx, c_num_mm in tqdm(
            enumerate(np.random.binomial(self.seq_lens, mm_rate)),
            smoothing=0,
            total=self.nchunks,
            desc="Mismatches",
            leave=False,
            disable=os.environ.get("LOG_SAFE", False),
        ):
            if c_num_mm == 0:
                continue
            for seq_pos in np.random.choice(
                self.seq_lens[c_idx], c_num_mm, replace=False
            ):
                # convert from position in chunk sequence to position in
                # sequence array (including k-mer context bases).
                seq_arr_pos = seq_pos + self.kmer_context_bases[0]
                if self.seq_array[c_idx, seq_arr_pos] == -1:
                    continue
                self.seq_array[c_idx, seq_arr_pos] = np.random.choice(
                    MISMATCH_ARRS[self.seq_array[c_idx, seq_arr_pos]]
                )
                errs_added += 1
        LOGGER.info(f"Introduced {errs_added} mismatch errors")

    def perturb_seq_to_sig_map(self, sig_shift):
        for c_idx, c_seq_len in tqdm(
            enumerate(self.seq_lens),
            smoothing=0,
            total=self.nchunks,
            desc="Signal shifts",
            leave=False,
            disable=os.environ.get("LOG_SAFE", False),
        ):
            # shift seq to sig mapping in the middle of chunk
            self.seq_mappings[c_idx, 1:c_seq_len] = np.clip(
                self.seq_mappings[c_idx, 1:c_seq_len] + sig_shift,
                self.seq_mappings[c_idx, 0],
                self.seq_mappings[c_idx, c_seq_len],
            )

    @property
    def can_base(self):
        return self.motifs[0][0][self.motifs[0][1]]

    @property
    def is_clipped(self):
        return self.nchunks == self.sig_tensor.shape[0]

    @property
    def is_multiclass(self):
        return self.base_pred or len(self.mod_bases) > 1

    @property
    def num_motifs(self):
        return len(self.motifs)

    @property
    def num_labels(self):
        if self.base_pred:
            return 4
        return len(self.mod_bases) + 1

    @property
    def summary(self):
        return (
            f"               num chunks : {self.nchunks}\n"
            f"       label distribution : {self.get_label_counts()}\n"
            f"                base_pred : {self.base_pred}\n"
            f"                mod_bases : {self.mod_bases}\n"
            f"           mod_long_names : {self.mod_long_names}\n"
            f"       kmer_context_bases : {self.kmer_context_bases}\n"
            f"            chunk_context : {self.chunk_context}\n"
            f"                   motifs : {self.motifs}\n"
            f"           reverse_signal : {self.reverse_signal}\n"
            f" chunk_extract_base_start : {self.base_start_justify}\n"
            f"     chunk_extract_offset : {self.offset}\n"
            f"          sig_map_refiner : {self.sig_map_refiner}\n"
        )

    def __repr__(self):
        return self.summary

    @classmethod
    def allocate_empty_chunks(
        cls,
        num_chunks,
        chunk_context,
        kmer_context_bases,
        max_seq_len=None,
        min_samps_per_base=None,
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
        labels = np.empty(num_chunks, dtype=np.int64)
        read_ids = np.empty(num_chunks, dtype="U36")
        read_pos = np.empty(num_chunks, dtype=int)
        return cls(
            sig_tensor,
            seq_array,
            seq_mappings,
            seq_lens,
            labels,
            read_ids,
            read_pos,
            nchunks=0,
            chunk_context=chunk_context,
            max_seq_len=max_seq_len,
            kmer_context_bases=kmer_context_bases,
            *args,
            **kwargs,
        )


def merge_datasets(input_datasets, balance=False, quiet=False):
    def load_dataset(ds_path, num_chunks):
        dataset = RemoraDataset.load_from_file(
            ds_path,
            shuffle_on_iter=False,
            drop_last=False,
        )
        if num_chunks is not None and num_chunks < dataset.nchunks:
            dataset.shuffle()
        else:
            num_chunks = dataset.nchunks
        return dataset, num_chunks

    log_fp = LOGGER.debug if quiet else LOGGER.info

    # load first file to determine base_pred or mod_bases
    dataset, num_chunks = load_dataset(*input_datasets[0])
    base_pred = dataset.base_pred
    chunk_context = dataset.chunk_context
    max_seq_len = dataset.max_seq_len
    kmer_context_bases = dataset.kmer_context_bases
    motifs = set(dataset.motifs)
    sig_map_refiner = dataset.sig_map_refiner
    base_start_justify = dataset.base_start_justify
    reverse_signal = dataset.reverse_signal
    offset = dataset.offset
    raw_mod_long_names = []
    raw_mod_bases = []
    if not base_pred:
        for mod_base, mln in zip(dataset.mod_bases, dataset.mod_long_names):
            if mod_base not in raw_mod_bases:
                raw_mod_bases.append(mod_base)
                raw_mod_long_names.append(mln)
    del dataset
    gc.collect()

    all_num_chunks = [num_chunks]
    for ds_path, num_chunks in input_datasets[1:]:
        dataset, num_chunks = load_dataset(ds_path, num_chunks)
        for attr_name, attr_val in (
            ("base_pred", base_pred),
            ("chunk_context", chunk_context),
            ("max_seq_len", max_seq_len),
            ("kmer_context_bases", kmer_context_bases),
            ("base_start_justify", base_start_justify),
            ("offset", offset),
        ):
            if getattr(dataset, attr_name) != attr_val:
                raise RemoraError(
                    f"All datasets must have same {attr_name} "
                    f"{getattr(dataset, attr_name)} != {attr_val}"
                )
        if set(dataset.motifs) != motifs:
            log_fp(
                "WARNING: Datasets have different motifs. Merging motifs "
                f"{motifs} with motifs {set(dataset.motifs)}"
            )
            motifs.update(dataset.motifs)
        if not base_pred:
            for mod_base, mln in zip(dataset.mod_bases, dataset.mod_long_names):
                if mod_base not in raw_mod_bases:
                    raw_mod_bases.append(mod_base)
                    raw_mod_long_names.append(mln)
        all_num_chunks.append(num_chunks)
        # TODO add checks for refine attributes
        del dataset
        gc.collect()

    all_mod_bases = ""
    all_mod_long_names = None
    if not base_pred:
        # sort modified bases alphabetically
        all_mod_long_names = []
        for idx in sorted(
            range(len(raw_mod_bases)), key=raw_mod_bases.__getitem__
        ):
            all_mod_bases += raw_mod_bases[idx]
            all_mod_long_names.append(raw_mod_long_names[idx])

    output_dataset = RemoraDataset.allocate_empty_chunks(
        num_chunks=sum(all_num_chunks),
        chunk_context=chunk_context,
        max_seq_len=max_seq_len,
        kmer_context_bases=kmer_context_bases,
        base_pred=base_pred,
        mod_bases=all_mod_bases,
        mod_long_names=all_mod_long_names,
        motifs=motifs,
        reverse_signal=reverse_signal,
        sig_map_refiner=sig_map_refiner,
        base_start_justify=base_start_justify,
        offset=offset,
    )
    LOGGER.info(f"Output dataset summary:\n{output_dataset.summary}")
    for ds_path, num_chunks in input_datasets:
        input_dataset, num_chunks = load_dataset(ds_path, num_chunks)
        if base_pred:
            label_conv = np.arange(4, dtype=np.int64)
        else:
            label_conv = np.empty(
                len(input_dataset.mod_bases) + 1, dtype=np.int64
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
            (b_rids, b_rfbs),
        ) in input_dataset:
            b_labels = label_conv[b_labels]
            if added_chunks + b_labels.size >= num_chunks:
                batch_size = num_chunks - added_chunks
                output_dataset.add_batch(
                    b_sig[:batch_size],
                    b_seq[:batch_size],
                    b_ss_map[:batch_size],
                    b_seq_lens[:batch_size],
                    b_labels[:batch_size],
                    b_rids[:batch_size],
                    b_rfbs[:batch_size],
                )
                added_chunks += batch_size
                break
            added_chunks += b_labels.size
            output_dataset.add_batch(
                b_sig, b_seq, b_ss_map, b_seq_lens, b_labels, b_rids, b_rfbs
            )
        log_fp(
            f"Copied {added_chunks} chunks. New label distribution: "
            f"{output_dataset.get_label_counts()}"
        )
        del input_dataset
        gc.collect()
    output_dataset.clip_chunks()
    if balance:
        balanced_dataset = output_dataset.balance_classes()
        log_fp(
            f"Balanced out to {balanced_dataset.sig_tensor.shape[0]} chunks. "
            f"New label distribution: {balanced_dataset.get_label_counts()}"
        )
        return balanced_dataset

    return output_dataset
