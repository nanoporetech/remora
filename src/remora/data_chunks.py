import re
import os
import json
import hashlib
import dataclasses
from glob import glob
from copy import deepcopy
from itertools import chain

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import IterableDataset

from remora.refine_signal_map import SigMapRefiner
from remora.data_chunks_core import trim_sb_chunk_context_core
from remora import constants, log, RemoraError, util, encoded_kmers

LOGGER = log.get_logger()

DATASET_VERSION = 3
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
MATCH_OPS_SET = set(np.where(MATCH_OPS)[0])
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
    while len(cigar) > 0 and cigar[-1][0] not in MATCH_OPS_SET:
        cigar = cigar[:-1]
    if len(cigar) == 0:
        raise RemoraError("No match operations found in alignment cigar")
    ops, lens = map(np.array, zip(*cigar))
    if ops.min() < 0 or ops.max() > 8:
        raise RemoraError("Invalid cigar op(s)")
    if lens.min() < 0:
        raise RemoraError("Cigar lengths may not be negative")

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
    return map_ref_to_signal(
        query_to_signal=query_to_signal, ref_to_query_knots=ref_to_read_knots
    )


@dataclasses.dataclass
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
            self._sig = ((self.dacs - self.shift) / self.scale).astype(
                np.float32
            )
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
        return chunk

    def iter_chunks(
        self,
        chunk_context,
        kmer_context_bases,
        base_start_justify=False,
        offset=0,
        check_chunks=False,
        motifs=None,
    ):
        for focus_base in self.focus_bases:
            if motifs is not None:
                if not any(
                    motif.match(self.int_seq, focus_base) for motif in motifs
                ):
                    LOGGER.debug("FAILED_MOTIF_CHECK")
                    continue
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
                model_metadata["base_start_justify"],
                model_metadata["offset"],
            )
        )
        if len(chunks) == 0:
            return
        motif_seqs, motif_offsets = zip(*model_metadata["motifs"])
        # prepare in memory dataset to perform chunk extraction
        dataset = CoreRemoraDataset(
            mode="w",
            metadata=DatasetMetadata(
                allocate_size=len(chunks),
                max_seq_len=max(c.seq_len for c in chunks),
                mod_bases=model_metadata["mod_bases"],
                mod_long_names=model_metadata["mod_long_names"],
                motif_sequences=motif_seqs,
                motif_offsets=motif_offsets,
                chunk_context=model_metadata["chunk_context"],
                kmer_context_bases=model_metadata["kmer_context_bases"],
                extra_arrays={"read_focus_bases": ("int64", "")},
            ),
            infinite_iter=False,
        )
        for chunk in chunks:
            dataset.write_chunk(chunk)
        for batch in dataset:
            self.batches.append(
                (
                    batch["signal"],
                    batch["enc_kmers"],
                    batch["labels"],
                    batch["read_focus_bases"],
                )
            )

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


@dataclasses.dataclass
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
        if np.any(np.isnan(self.signal)):
            LOGGER.debug(
                f"FAILED_CHUNK: NaN signal {self.read_id} "
                f"{self.read_focus_base}"
            )
            raise RemoraError("Signal contains NaN")
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


@dataclasses.dataclass
class DatasetMetadata:
    """DatasetMetadata contains metadata related to a RemoraDataset or
    CoreRemoraDataset. This data is transferred to a Remora model at training
    time to ensure that chunk extraction is performed the same at training data
    preparation as inference.

    Args:
        allocate_size (int): Size (number of chunks) allocated for dataset
        max_seq_len (int): Maximum sequence length of a chunk (used to set
            dimension for seq arrays.
        mod_bases (list): Modified base short names represented by labels
            (single letter or ChEBI codes)
        mod_long_names (list): Modified base long names represented by labels
        motifs_sequences (list): Sequences at which model trained from chunks
            is applicable
        motifs_offsets (list): Offsets within motifs_sequences are applicable
        dataset_start (int): Index of first chunk to use when reading/iterating
            over the dataset
        dataset_end (int): Index one beyond the  last chunk to use when
            reading/iterating over the dataset
        version (int): Dataset version
        modified_base_labels (bool): Are labels modified bases? Non-modified
            base dataset will generally require custom scripts for inference.
        extra_arrays (dict): Extra arrays to store information about chunks.
            Dict keys define the name of the extra arrays and values contain
            the string dtype of the array and a description of the data to self
            document the dataset.
        chunk_context (tuple): 2-tuple containing the number of signal points
            before and after the central position.
        base_start_justify (bool): Extract chunk centered on start of base
        offset (int): Extract chunk centered on base offset from base of
            interest
        kmer_context_bases (tuple): 2-tuple containing the bases to include in
            the encoded k-mer presented as input.
        reverse_signal (bool): Is nanopore signal 3' to 5' orientation?
            Primarily for directRNA
        sig_map_refiner (remora.refine_signal_map.SigMapRefiner): Signal
            mapping refiner
    """

    # dataset attributes
    allocate_size: int
    max_seq_len: int
    # labels
    mod_bases: list
    mod_long_names: list
    # chunk extract
    motif_sequences: list
    motif_offsets: list

    dataset_start: int = 0
    dataset_end: int = 0
    version: int = DATASET_VERSION
    modified_base_labels: bool = True
    # extra arrays
    extra_arrays: dict = None
    # chunk extract
    chunk_context: tuple = constants.DEFAULT_CHUNK_CONTEXT
    base_start_justify: bool = False
    offset: int = 0
    kmer_context_bases: tuple = constants.DEFAULT_KMER_CONTEXT_BASES
    reverse_signal: bool = False
    # signal refinement
    sig_map_refiner: SigMapRefiner = None
    rough_rescale_method: str = constants.DEFAULT_ROUGH_RESCALE_METHOD

    _stored_kmer_context_bases: tuple = None
    _stored_chunk_context: tuple = None

    @property
    def chunk_width(self):
        return sum(self.chunk_context)

    @property
    def stored_chunk_context(self):
        if self._stored_chunk_context is None:
            return self.chunk_context
        return self._stored_chunk_context

    @property
    def stored_chunk_width(self):
        return sum(self.stored_chunk_context)

    @property
    def chunk_context_adjusted(self):
        return self.stored_chunk_context != self.chunk_context

    @property
    def kmer_len(self):
        return sum(self.kmer_context_bases) + 1

    @property
    def stored_kmer_context_bases(self):
        if self._stored_kmer_context_bases is None:
            return self.kmer_context_bases
        return self._stored_kmer_context_bases

    @property
    def kmer_context_bases_adjusted(self):
        return self.stored_kmer_context_bases != self.kmer_context_bases

    @property
    def size(self):
        return self.dataset_end - self.dataset_start

    @property
    def labels(self):
        return ["control"] + self.mod_long_names

    @property
    def num_labels(self):
        return len(self.mod_long_names) + 1

    @property
    def motifs(self):
        return list(zip(self.motif_sequences, self.motif_offsets))

    @property
    def num_motifs(self):
        return len(self.motif_sequences)

    @property
    def extra_array_names(self):
        return (
            [] if self.extra_arrays is None else list(self.extra_arrays.keys())
        )

    @property
    def extra_array_dtypes_and_shapes(self):
        return (
            []
            if self.extra_arrays is None
            else [
                (arr_name, arr_dtype, self.extras_shape)
                for arr_name, (arr_dtype, _) in self.extra_arrays.items()
            ]
        )

    @property
    def signal_shape(self):
        return self.allocate_size, 1, self.stored_chunk_width

    @property
    def sequence_width(self):
        return self.max_seq_len + sum(self.stored_kmer_context_bases)

    @property
    def sequence_shape(self):
        return self.allocate_size, self.sequence_width

    @property
    def sequence_to_signal_mapping_width(self):
        return self.max_seq_len + 1

    @property
    def sequence_to_signal_mapping_shape(self):
        return self.allocate_size, self.sequence_to_signal_mapping_width

    @property
    def sequence_lengths_shape(self):
        return tuple((self.allocate_size,))

    @property
    def labels_shape(self):
        return tuple((self.allocate_size,))

    @property
    def extras_shape(self):
        return tuple((self.allocate_size,))

    def check_motifs(self):
        motifs = [util.Motif(*motif) for motif in self.motifs]
        ambig_focus_motifs = [
            motif for motif in motifs if motif.focus_base not in "ACGT"
        ]
        if len(ambig_focus_motifs) > 0:
            raise RemoraError(
                f"Cannot create dataset at motifs with ambiguous bases "
                f"{ambig_focus_motifs}"
            )
        focus_bases = set(motif.focus_base for motif in motifs)
        if len(focus_bases) > 1:
            raise RemoraError(
                f"Cannot create dataset with multiple motif focus bases: "
                f"{focus_bases}"
            )

    def __post_init__(self):
        # Support original single letter codes or new list short names
        # (including ChEBI codes)
        if isinstance(self.mod_bases, str):
            self.mod_bases = list(self.mod_bases)
        self.mod_bases = list(map(str, self.mod_bases))
        assert len(self.mod_bases) == len(self.mod_long_names), (
            f"mod_bases ({self.mod_bases}) must be the same length as "
            f"mod_long_names ({self.mod_long_names})"
        )
        self.chunk_context = tuple(self.chunk_context)
        self.kmer_context_bases = tuple(self.kmer_context_bases)
        if self._stored_chunk_context is not None:
            self._stored_chunk_context = tuple(self._stored_chunk_context)
        if self._stored_kmer_context_bases is not None:
            self._stored_kmer_context_bases = tuple(
                self._stored_kmer_context_bases
            )
        self.check_motifs()

    def asdict(self):
        r_dict = dataclasses.asdict(self)
        del r_dict["sig_map_refiner"]
        if self.sig_map_refiner is not None:
            r_dict.update(self.sig_map_refiner.asdict())
        return r_dict

    def copy(self):
        return deepcopy(self)

    def write(self, metadata_path, kmer_table_path=None):
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        self_dict = self.asdict()
        if self_dict.get("refine_kmer_levels") is not None:
            if kmer_table_path is not None:
                np.save(
                    kmer_table_path,
                    self_dict["refine_kmer_levels"],
                    allow_pickle=False,
                )
            del self_dict["refine_kmer_levels"]
        with open(metadata_path, "w") as metadata_fh:
            json.dump(self_dict, metadata_fh, cls=NpEncoder)


def check_super_batch(super_batch, chunk_width):
    if not np.all(super_batch["sequence_lengths"]) > 0:
        raise RemoraError("Sequence lengths must all be positive.")
    # check that seq to sig mapping extends from chunk start to end
    sm_r = np.arange(super_batch["sequence_to_signal_mapping"].shape[1])
    sm_mask = sm_r < (super_batch["sequence_lengths"][:, None] + 1)
    sm_m = super_batch["sequence_to_signal_mapping"][sm_mask]
    if sm_m.max() > chunk_width:
        raise RemoraError("Signal mapping extend beyond chunk width")
    if sm_m.min() < 0:
        raise RemoraError("Signal mapping cannot contain negative values")
    chunks_r = np.arange(super_batch["sequence_lengths"].size)
    if not np.all(
        super_batch["sequence_to_signal_mapping"][
            chunks_r, super_batch["sequence_lengths"]
        ]
        == chunk_width
    ):
        raise RemoraError("Chunk does not end at chunk_width")
    seqlen_cs = np.cumsum(super_batch["sequence_lengths"])
    sm_diff_mask = np.ones(sm_m.size - 1, dtype=bool)
    sm_diff_mask[seqlen_cs[:-1] + np.arange(seqlen_cs.size)[:-1]] = 0
    if np.diff(sm_m)[sm_diff_mask].min() < 0:
        raise RemoraError("Sequence to signal mappings are not monotonic")
    # check that sequence values are valid
    seq_r = np.arange(super_batch["sequence"].shape[1])
    seq_mask = seq_r < super_batch["sequence_lengths"][:, None]
    seq_m = super_batch["sequence"][seq_mask]
    if seq_m.max() > 3:
        raise RemoraError("Sequence max must be less than 4")
    if seq_m.min() < -1:
        raise RemoraError("Sequence min must greater tha -2")


@dataclasses.dataclass
class CoreRemoraDataset:
    """CoreRemoraDataset manages the storage and access to a single file of
    training data.
    """

    data_path: str = None
    mode: str = "r"
    metadata: DatasetMetadata = None
    override_metadata: dict = None
    batch_size: int = constants.DEFAULT_BATCH_SIZE
    super_batch_size: int = constants.DEFAULT_SUPER_BATCH_SIZE
    super_batch_sample_frac: float = None
    super_batch_offset: int = 0
    infinite_iter: bool = True
    do_check_super_batches: bool = False

    _core_dtypes = {
        "signal": np.float32,
        "sequence": np.int8,
        "sequence_to_signal_mapping": np.int16,
        "sequence_lengths": np.int16,
        "labels": np.int64,
    }
    _core_arrays = list(_core_dtypes.keys())

    @staticmethod
    def dataset_paths(data_path):
        data_path = util.resolve_path(data_path)
        paths = [
            os.path.join(data_path, arr_path)
            for arr_path in ["metadata.jsn"]
            + [
                f"{array_name}.npy"
                for array_name in CoreRemoraDataset._core_arrays
            ]
        ]
        paths.extend(glob(os.path.join(data_path, "extra_*.npy")))
        if os.path.isfile(os.path.join(data_path, "kmer_table.npy")):
            paths.append(os.path.join(data_path, "kmer_table.npy"))
        return paths

    @staticmethod
    def check_dataset_dir(data_path):
        return all(
            [
                os.path.isfile(arr_path)
                for arr_path in CoreRemoraDataset.dataset_paths(data_path)
            ]
        )

    @staticmethod
    def hash(data_path):
        def file_digest(fh, _bufsize=2**18, num_buf=8):
            # copy bits from hashlib file_digest to port back to python<3.11
            # https://github.com/python/cpython/blob/3.11/Lib/hashlib.py#L292
            digest = hashlib.sha256()
            buf = bytearray(_bufsize)
            view = memoryview(buf)
            file_size = fh.seek(0, os.SEEK_END)
            if file_size < _bufsize * num_buf:
                # if file is smaller than _bufsize * num_buf digest entire file
                fh.seek(0)
                while True:
                    size = fh.readinto(buf)
                    if size == 0:
                        break
                    digest.update(view[:size])
            else:
                # else digest num_buf evenly spaced chunks of the file
                for f_pos in np.floor(
                    np.linspace(0, file_size - _bufsize, num_buf)
                ).astype(int):
                    fh.seek(f_pos)
                    fh.readinto(buf)
                    digest.update(view)
            return digest.hexdigest()

        LOGGER.debug(f"Computing hash for dataset at {data_path}")
        files_hash = ""
        for arr_path in CoreRemoraDataset.dataset_paths(data_path):
            with open(arr_path, "rb") as fh:
                files_hash += file_digest(fh)
        return hashlib.sha256(files_hash.encode("utf-8")).hexdigest()

    @property
    def metadata_path(self):
        if self.data_path is None:
            raise RemoraError("No path available for in-memory dataset")
        return os.path.join(self.data_path, "metadata.jsn")

    @property
    def kmer_table_path(self):
        if self.data_path is None:
            raise RemoraError("No path available for in-memory dataset")
        return os.path.join(self.data_path, "kmer_table.npy")

    @property
    def size(self):
        return self.metadata.dataset_end - self.metadata.dataset_start

    @property
    def array_names(self):
        return self._core_arrays + self.metadata.extra_array_names

    @property
    def arrays(self):
        """Generator of chunk arrys in dataset. Arrays will be sliced to current
        dataset size not allocated arrays. Note that this will load each array
        from disk.
        """
        for array_name in self.array_names:
            yield getattr(self, array_name)[
                self.metadata.dataset_start : self.metadata.dataset_end
            ]

    @property
    def arrays_info(self):
        return list(
            chain(
                (
                    (name, dtype, getattr(self.metadata, f"{name}_shape"))
                    for name, dtype in self._core_dtypes.items()
                ),
                self.metadata.extra_array_dtypes_and_shapes,
            )
        )

    @property
    def summary(self):
        return (
            f"                data_path : {self.data_path}\n"
            f"                     size : {self.size:,}\n"
            f"            dataset_start : {self.metadata.dataset_start:,}\n"
            f"              dataset_end : {self.metadata.dataset_end:,}\n"
            f"       label distribution : {self.label_summary}\n"
            "     modified_base_labels : "
            f"{self.metadata.modified_base_labels}\n"
            f"                mod_bases : {self.metadata.mod_bases}\n"
            f"           mod_long_names : {self.metadata.mod_long_names}\n"
            f"       kmer_context_bases : {self.metadata.kmer_context_bases}\n"
            f"            chunk_context : {self.metadata.chunk_context}\n"
            f"                   motifs : {self.metadata.motifs}\n"
            f"           reverse_signal : {self.metadata.reverse_signal}\n"
            f" chunk_extract_base_start : {self.metadata.base_start_justify}\n"
            f"     chunk_extract_offset : {self.metadata.offset}\n"
            f"          sig_map_refiner : {self.metadata.sig_map_refiner}\n"
        )

    def get_label_counts(self):
        ds_labels = self.labels[
            self.metadata.dataset_start : self.metadata.dataset_end
        ]
        if self.label_conv is None:
            lab_counts = np.bincount(ds_labels)
        else:
            lab_counts = np.bincount(self.label_conv[ds_labels])
        return lab_counts

    @property
    def label_summary(self):
        return "; ".join(
            f"{self.metadata.labels[lab_idx]}:{count:,}"
            for lab_idx, count in enumerate(self.get_label_counts())
        )

    def load_metadata(self):
        """Load metadata from file and apply override_metadata attributes if
        possible.

        Attributes allowed to be overridden are:
          - dataset_start
            - Allows slicing of accessed elements
          - dataset_end
            - Allows slicing of accessed elements
          - mod_bases
            - Allow expansion of labels represented
          - mod_long_names
            - Allow expansion of labels represented
          - extra_arrays
            - Must be equal or subset of stored extra arrays
          - kmer_context_bases
            - Both values must be smaller than stored values
          - chunk_context
            - Both values must be smaller than stored values
        """
        with open(self.metadata_path) as metadata_fh:
            loaded_metadata = json.load(metadata_fh)
        if loaded_metadata.get("version") != DATASET_VERSION:
            raise RemoraError(
                f"Remora dataset version ({loaded_metadata.get('version')}) "
                f"does not match current distribution ({DATASET_VERSION})"
            )
        # load signal map refiner if supplied
        if os.path.exists(self.kmer_table_path):
            loaded_metadata["refine_kmer_levels"] = np.load(
                self.kmer_table_path
            )
        loaded_metadata["refine_sd_arr"] = np.asarray(
            loaded_metadata["refine_sd_arr"], np.float32
        )
        loaded_metadata["sig_map_refiner"] = SigMapRefiner.load_from_metadata(
            loaded_metadata
        )
        for ra in [k for k in loaded_metadata if k.startswith("refine_")]:
            del loaded_metadata[ra]
        if self.override_metadata is None:
            self.metadata = DatasetMetadata(**loaded_metadata)
            return

        # process metadata to override loaded metadata
        invalid_keys = []
        for md_key, md_val in self.override_metadata.items():
            if md_key == "dataset_start":
                if md_val < 0:
                    raise RemoraError("Dataset start must be positive")
            elif md_key == "dataset_end":
                if md_val > loaded_metadata["dataset_end"]:
                    raise RemoraError("Cannot set dataset end past loaded end")
            elif md_key == "mod_bases":
                assert "mod_long_names" in self.override_metadata
                assert len(self.override_metadata["mod_long_names"]) == len(
                    md_val
                )
                # TODO remove this and have additional local labels added to
                # the end. Need to consider actual use cases
                assert all(
                    mb in md_val for mb in self.metadata.mod_bases
                ), "Cannot remove modified base"
                if (
                    self.metadata.mod_bases
                    != md_val[: len(self.metadata.mod_bases)]
                ):
                    self.label_conv = np.empty(
                        self.metadata.num_labels, dtype=np.int64
                    )
                    self.label_conv[0] = 0
                    for in_lab, mod_base in enumerate(self.metadata.mod_bases):
                        # apply at super chunks and label access
                        self.label_conv[in_lab + 1] = next(
                            idx + 1
                            for idx, mb in enumerate(md_val)
                            if mb == mod_base
                        )
                    LOGGER.debug(
                        f"Setting label conversion: {self.label_conv} "
                        f"{self.data_path}"
                    )
            elif md_key == "mod_long_names":
                assert "mod_bases" in self.override_metadata
            elif md_key == "extra_arrays":
                missing_arrays = set(md_val).difference(
                    loaded_metadata["extra_arrays"]
                )
                if len(missing_arrays) > 0:
                    raise RemoraError(
                        "Cannot load missing arrays: "
                        f"{', '.join(missing_arrays)}\nAvailable extra arrays: "
                        f"{', '.join(loaded_metadata['extra_arrays'].keys())}"
                    )
                md_val = dict(
                    (k, loaded_metadata["extra_arrays"][k]) for k in md_val
                )
            elif md_key == "chunk_context":
                md_val = tuple(md_val)
                scc = loaded_metadata["chunk_context"] = tuple(
                    loaded_metadata["chunk_context"]
                )
                if md_val[0] > scc[0] or md_val[1] > scc[1]:
                    raise RemoraError(
                        f"Cannot expand chunk context (stored:{scc} ; "
                        f"requested:{md_val})"
                    )
                loaded_metadata["_stored_chunk_context"] = scc
            elif md_key == "kmer_context_bases":
                md_val = tuple(md_val)
                skcb = loaded_metadata["kmer_context_bases"] = tuple(
                    loaded_metadata["kmer_context_bases"]
                )
                if md_val[0] > skcb[0] or md_val[1] > skcb[1]:
                    raise RemoraError(
                        f"Cannot expand kmer context (stored:{skcb} ; "
                        f"requested:{md_val})"
                    )
                loaded_metadata["_stored_kmer_context_bases"] = skcb
            else:
                invalid_keys.append(md_key)
                continue
            # if no error is raised, set metadata value
            if loaded_metadata[md_key] != md_val:
                LOGGER.debug(
                    f"Overriding {md_key} from value "
                    f"'{loaded_metadata[md_key]}' to '{md_val}'"
                )
                loaded_metadata[md_key] = md_val
        if loaded_metadata["dataset_start"] >= loaded_metadata["dataset_end"]:
            raise RemoraError("Loaded dataset is empty")
        if len(invalid_keys) > 0:
            raise RemoraError(
                f"Cannot change metadata values: {', '.join(invalid_keys)}"
            )
        self.metadata = DatasetMetadata(**loaded_metadata)

    def update_metadata(self, other):
        """Update metadata to match attributes from another dataset"""
        # TODO add a dry run option to check compatibility of merging metadata
        # would allow datasets to remain unchanged until all datasets have been
        # checked. Note this would require refactoring checks out of
        # load_metadata to use here as well
        md = dict(
            (
                (md_key, getattr(other.metadata, md_key))
                for md_key in (
                    "mod_bases",
                    "mod_long_names",
                    "extra_arrays",
                    "kmer_context_bases",
                    "chunk_context",
                )
            )
        )
        if len(md) > 0:
            # keep start and end at set values
            md.update(
                {
                    "dataset_start": self.metadata.dataset_start,
                    "dataset_end": self.metadata.dataset_end,
                }
            )
            self.override_metadata = md
            # load metadata instead of setting values directly to set
            # associated attributes (label_conv etc)
            self.load_metadata()

    def get_array_path(self, array_name):
        if self.data_path is None:
            raise RemoraError("No path available for in-memory dataset")
        if array_name in self._core_arrays:
            return os.path.join(self.data_path, f"{array_name}.npy")
        elif array_name in self.metadata.extra_arrays:
            return os.path.join(self.data_path, f"extra_{array_name}.npy")
        raise RemoraError(f"Invalid extra array name: {array_name}")

    def allocate_arrays(self):
        if self.mode != "w":
            raise RemoraError("Cannot write when mode is not 'w'")
        if self.data_path is None:
            # load in memory numpy arrays
            for arr_name, arr_dtype, arr_shape in self.arrays_info:
                setattr(
                    self,
                    arr_name,
                    np.empty(dtype=arr_dtype, shape=arr_shape),
                )
            return
        for arr_name, arr_dtype, arr_shape in self.arrays_info:
            # Open with write mode only in this method
            setattr(
                self,
                arr_name,
                np.memmap(
                    self.get_array_path(arr_name),
                    arr_dtype,
                    mode="w+",
                    shape=arr_shape,
                ),
            )

    def refresh_memmaps(self):
        # in-memory dataset does not touch memmaps
        if self.data_path is None:
            return
        mode = "r" if self.mode == "r" else "r+"
        for arr_name, arr_dtype, arr_shape in self.arrays_info:
            # close prev memmap to avoid mem leaks
            if hasattr(self, arr_name):
                delattr(self, arr_name)
            setattr(
                self,
                arr_name,
                np.memmap(
                    self.get_array_path(arr_name),
                    arr_dtype,
                    mode=mode,
                    shape=arr_shape,
                ),
            )

    def close_memmaps(self):
        # in-memory dataset does not touch memmaps
        if self.data_path is None:
            return
        for arr_name in self._core_arrays:
            setattr(self, arr_name, None)

    def write_metadata(self):
        self.metadata.write(self.metadata_path, self.kmer_table_path)

    def __post_init__(self):
        self.label_conv = None
        assert self.mode in "rw", "mode must be 'r' or 'w'"
        if self.data_path is None:
            assert self.mode == "w", "In-memory dataset must have mode='w'"
            assert isinstance(
                self.metadata, DatasetMetadata
            ), "Must provide metadata for in-memory dataset"
            self.allocate_arrays()
        elif self.mode == "r":
            self.data_path = util.resolve_path(self.data_path)
            self.load_metadata()
        else:
            assert isinstance(
                self.metadata, DatasetMetadata
            ), "Must provide metadata for new dataset"
            self.data_path = util.resolve_path(self.data_path)
            self.allocate_arrays()
            self.write_metadata()
        self.refresh_memmaps()
        self._iter = None

    def write_batch(self, arrays):
        if self.mode != "w":
            raise RemoraError("Cannot write when mode is not 'w'")
        batch_size = next(iter(arrays.values())).shape[0]
        if any(arr.shape[0] != batch_size for arr in arrays.values()):
            raise RemoraError("All arrays in a batch must be the same size")
        if self.metadata.dataset_end + batch_size > self.metadata.allocate_size:
            # write metadata before raise to update size on disk
            self.write_metadata()
            raise RemoraError("Batch write greater than allocated memory")
        missing_arrs = set(self.array_names).difference(arrays.keys())
        if len(missing_arrs) > 0:
            raise RemoraError(
                "Batch write must include all arrays. Missing: "
                f"{', '.join(missing_arrs)}"
            )
        unspec_arrs = set(arrays.keys()).difference(self.array_names)
        if len(unspec_arrs) > 0:
            raise RemoraError(
                "Batch write must only include spcified arrays. Found: "
                f"{', '.join(unspec_arrs)}"
            )
        for arr_name, in_array in arrays.items():
            out_array = getattr(self, arr_name)
            out_array[
                self.metadata.dataset_end : self.metadata.dataset_end
                + batch_size
            ] = in_array
        # update size
        self.metadata.dataset_end = self.metadata.dataset_end + batch_size

    def write_chunk(self, chunk):
        if self.mode != "w":
            raise RemoraError("Cannot write when mode is not 'w'")
        seq_arr = np.empty(
            (1, self.metadata.sequence_width),
            dtype=self._core_dtypes["sequence"],
        )
        seq_arr[0, : chunk.seq_w_context.size] = chunk.seq_w_context
        ssm_arr = np.empty(
            (1, self.metadata.sequence_to_signal_mapping_width),
            dtype=self._core_dtypes["sequence_to_signal_mapping"],
        )
        ssm_arr[0, : chunk.seq_to_sig_map.size] = chunk.seq_to_sig_map
        chunk_dict = {
            "signal": np.expand_dims(chunk.signal, axis=0).astype(
                self._core_dtypes["signal"]
            ),
            "sequence": seq_arr,
            "sequence_to_signal_mapping": ssm_arr,
            "sequence_lengths": np.array(
                [chunk.seq_len], dtype=self._core_dtypes["sequence_lengths"]
            ),
            "labels": np.array(
                [chunk.label], dtype=self._core_dtypes["labels"]
            ),
        }
        if (
            self.metadata.extra_arrays is not None
            and "read_ids" in self.metadata.extra_arrays
        ):
            chunk_dict["read_ids"] = np.array(
                [chunk.read_id],
                dtype=self.metadata.extra_arrays["read_ids"][0],
            )
        if (
            self.metadata.extra_arrays is not None
            and "read_focus_bases" in self.metadata.extra_arrays
        ):
            chunk_dict["read_focus_bases"] = np.array(
                [chunk.read_focus_base],
                dtype=self.metadata.extra_arrays["read_focus_bases"][0],
            )
        self.write_batch(chunk_dict)

    def shuffle(self, batch_size=100_000, show_prog=False):
        # TODO add option to perform pseudo-shuffle without reading full
        # core arrays into memory.
        if self.mode != "w":
            raise RemoraError("Cannot write when mode is not 'w'")
        b_ranges = list(
            zip(
                range(0, self.size, batch_size),
                range(batch_size, self.size + batch_size, batch_size),
            )
        )
        shuf_indices = np.random.permutation(self.size)
        if show_prog:
            arr_pb = tqdm(
                total=len(self.array_names),
                smoothing=0,
                position=0,
                desc="Arrays",
            )
        for array_name in self.array_names:
            if show_prog:
                b_pb = tqdm(
                    total=len(b_ranges),
                    smoothing=0,
                    leave=False,
                    position=1,
                    desc="Batches",
                )
            LOGGER.debug(f"Shuffling {array_name} array")
            # note that memmap array slice remains a reference to the memmap
            # array so writes still apply here.
            array = getattr(self, array_name)[
                self.metadata.dataset_start : self.metadata.dataset_end
            ]
            arr_copy = array.copy()
            for b_idx, (b_st, b_en) in enumerate(b_ranges):
                array[b_st : min(b_en, self.size)] = arr_copy[
                    shuf_indices[b_st:b_en]
                ]
                array.flush()
                LOGGER.debug(f"{b_idx + 1}/{len(b_ranges)} batches complete")
                if show_prog:
                    b_pb.update()
            if show_prog:
                b_pb.close()
                arr_pb.update()

    def adjust_batch_params(self):
        """Adjust super-batch parameters to be valid values. Including setting
        super batch size to no larger than the dataset and
        """
        if self.super_batch_size > self.size:
            self.super_batch_size = self.size
        if self.super_batch_sample_frac is None:
            sb_select_num_chunks = None
            chunks_per_sb = self.super_batch_size
        else:
            prev_batch_size = self.batch_size
            prev_sb_size = self.super_batch_size
            # round up to next number batch size and adjust other batch attrs
            # accordingly
            sb_select_num_chunks = int(
                np.ceil(
                    self.super_batch_size
                    * self.super_batch_sample_frac
                    / self.batch_size
                )
                * self.batch_size
            )
            if sb_select_num_chunks > self.super_batch_size:
                sb_select_num_chunks -= self.batch_size
            if sb_select_num_chunks == 0:
                self.batch_size = int(
                    self.super_batch_size * self.super_batch_sample_frac
                )
                sb_select_num_chunks = self.batch_size
            if self.super_batch_sample_frac == 1.0:
                # allow ragged batch from finite iterator if frac is 1.0
                self.super_batch_size = sb_select_num_chunks
            chunks_per_sb = sb_select_num_chunks
            LOGGER.debug(
                f"Adjusted values for super_batch_sample_frac: "
                f"{self.super_batch_sample_frac}\tbatch_size: "
                f"{prev_batch_size}->{self.batch_size}\tsuper_batch_size: "
                f"{prev_sb_size}->{self.super_batch_size}"
            )
        return chunks_per_sb, sb_select_num_chunks

    def trim_sb_kmer_context_bases(self, super_batch):
        """Trim super-batch sequence array to achieve loaded k-mer context
        bases. Note that the end trimming is applied at the encoded k-mer
        computation via the compute_encoded_kmer_batch call with
        load_kmer_context_bases.
        """
        if not self.metadata.kmer_context_bases_adjusted:
            return super_batch
        seq_diff = (
            self.metadata.stored_kmer_context_bases[0]
            - self.metadata.kmer_context_bases[0]
        )
        if seq_diff > 0:
            try:
                super_batch["sequence"][:, :-seq_diff] = super_batch[
                    "sequence"
                ][:, seq_diff:]
            except ValueError:
                super_batch["sequence"] = super_batch["sequence"].copy()
                super_batch["sequence"][:, :-seq_diff] = super_batch[
                    "sequence"
                ][:, seq_diff:]
        return super_batch

    def trim_sb_chunk_context(self, super_batch):
        """Trim super-batch sequence and seq_to_sig_map for new chunk context.
        This requires additional compute for loading each super batch and may
        slow processing.

        Note if applying both dynamic chunk_context and kmer_context_bases, the
        trim_sb_kmer_context_bases function should be run first.
        """
        if not self.metadata.chunk_context_adjusted:
            return super_batch
        # simple signal array trimming
        st_diff = (
            self.metadata.stored_chunk_context[0]
            - self.metadata.chunk_context[0]
        )
        new_en = (
            self.metadata.stored_chunk_context[0]
            + self.metadata.chunk_context[1]
        )
        super_batch["signal"] = super_batch["signal"][:, :, st_diff:new_en]
        super_batch["signal"] = np.ascontiguousarray(super_batch["signal"])

        try:
            super_batch["sequence_to_signal_mapping"] -= st_diff
        except ValueError:
            super_batch["sequence_to_signal_mapping"] = (
                super_batch["sequence_to_signal_mapping"].copy() - st_diff
            )
            super_batch["sequence"] = super_batch["sequence"].copy()
            super_batch["sequence_lengths"] = super_batch[
                "sequence_lengths"
            ].copy()
        trim_sb_chunk_context_core(
            *self.metadata.stored_chunk_context,
            *self.metadata.chunk_context,
            sum(self.metadata.kmer_context_bases),
            super_batch["sequence"],
            super_batch["sequence_to_signal_mapping"],
            super_batch["sequence_lengths"],
        )
        return super_batch

    def load_super_batch(self, offset=0, size=None, select_num_chunks=None):
        super_batch = {}
        if self.infinite_iter:
            offset %= self.size
        else:
            if offset >= self.size:
                return
        sb_arr_st = self.metadata.dataset_start + offset
        # load full dataset if size is None
        if size is None:
            if self.infinite_iter:
                raise RemoraError(
                    "Must specify size of super batch for infinite iter dataset"
                )
            size = self.metadata.dataset_end - sb_arr_st
        if size > self.size:
            raise RemoraError("Super batch larger than dataset requested")
        sb_arr_en = sb_arr_st + size
        if sb_arr_en <= self.metadata.dataset_end:
            for arr_name in self.array_names:
                super_batch[arr_name] = getattr(self, arr_name)[
                    sb_arr_st:sb_arr_en
                ].copy()
        elif self.infinite_iter:
            # wrap super batch around end of dataset
            wrap_en = sb_arr_en - self.size
            for arr_name in self.array_names:
                super_batch[arr_name] = np.concatenate(
                    [
                        getattr(self, arr_name)[
                            sb_arr_st : self.metadata.dataset_end
                        ],
                        getattr(self, arr_name)[
                            self.metadata.dataset_start : wrap_en
                        ],
                    ]
                )
        else:
            # return last batch with smaller batch dim
            for arr_name in self.array_names:
                super_batch[arr_name] = getattr(self, arr_name)[
                    sb_arr_st : self.metadata.dataset_end
                ]
        if select_num_chunks is not None:
            selected_indices = np.random.choice(
                super_batch["labels"].size,
                min(select_num_chunks, super_batch["labels"].size),
                replace=False,
            )
            for arr_name in self.array_names:
                super_batch[arr_name] = super_batch[arr_name][selected_indices]
        if self.label_conv is not None:
            super_batch["labels"] = self.label_conv[super_batch["labels"]]
        super_batch = self.trim_sb_kmer_context_bases(super_batch)
        super_batch = self.trim_sb_chunk_context(super_batch)
        return super_batch

    def iter_super_batches(self, select_num_chunks=None):
        super_batch_num = 0
        while True:
            self.refresh_memmaps()
            super_batch = self.load_super_batch(
                self.super_batch_offset
                + (super_batch_num * self.super_batch_size),
                self.super_batch_size,
                select_num_chunks=select_num_chunks,
            )
            if super_batch is None:
                break
            if self.do_check_super_batches:
                check_super_batch(super_batch, self.metadata.chunk_width)
            super_batch_num += 1
            yield super_batch

    def extract_batch(self, super_batch, batch_st):
        batch_en = (
            super_batch["sequence"].shape[0]
            if batch_st + self.batch_size > super_batch["sequence"].shape[0]
            else batch_st + self.batch_size
        )
        batch = {
            "enc_kmers": encoded_kmers.compute_encoded_kmer_batch(
                *self.metadata.kmer_context_bases,
                super_batch["sequence"][batch_st:batch_en],
                super_batch["sequence_to_signal_mapping"][batch_st:batch_en],
                super_batch["sequence_lengths"][batch_st:batch_en],
            )
        }
        batch.update(
            dict(
                (
                    arr_name,
                    super_batch[arr_name][batch_st:batch_en],
                )
                for arr_name in ["signal", "labels"]
                + self.metadata.extra_array_names
            )
        )
        return batch

    def iter_batches(self, max_batches=None):
        chunks_per_sb, sb_select_num_chunks = self.adjust_batch_params()
        super_batches = self.iter_super_batches(sb_select_num_chunks)
        batch_num = 0
        for super_batch in super_batches:
            for batch_st in range(0, chunks_per_sb, self.batch_size):
                yield self.extract_batch(super_batch, batch_st)
                batch_num += 1
                if max_batches is not None and batch_num >= max_batches:
                    return

    def __iter__(self):
        if self._iter is None or not self.infinite_iter:
            self._iter = self.iter_batches()
        return self._iter

    def __next__(self):
        return next(self._iter)

    def flush(self):
        if self.data_path is None:
            return
        for arr_name in self.array_names:
            getattr(self, arr_name).flush()
        self.refresh_memmaps()


def parse_dataset_config(config_path, used_configs=None):
    paths, weights, hashes = [], [], []
    config_path = util.resolve_path(config_path)
    if used_configs is None:
        used_configs = {config_path: config_path}
    with open(config_path) as config_fh:
        for ds_info in json.load(config_fh):
            if len(ds_info) == 2:
                ds_path, weight = ds_info
                ds_hash = None
            elif len(ds_info) == 3:
                ds_path, weight, ds_hash = ds_info
            assert weight > 0, "dataset config weight must be positive"
            ds_path = util.resolve_path(ds_path)
            if not os.path.exists(ds_path):
                raise RemoraError(
                    f"Core dataset path does not exist. {ds_path}"
                )
            if os.path.isdir(ds_path):
                computed_hash = CoreRemoraDataset.hash(ds_path)
                if ds_hash is None:
                    ds_hash = computed_hash
                elif ds_hash != computed_hash:
                    raise RemoraError(
                        "Dataset hash does not match value from config "
                        f"for dataset at {ds_path}"
                    )
                paths.append(ds_path)
                weights.append(weight)
                hashes.append(ds_hash)
            else:
                if ds_path in used_configs:
                    raise RemoraError(
                        "Circular or repeated dataset config refrence. "
                        f"{ds_path} found in {config_path} and previously "
                        f"found in {used_configs[ds_path]}"
                    )
                used_configs[ds_path] = config_path
                sub_paths, sub_weights, sub_hashs = parse_dataset_config(
                    ds_path, used_configs=used_configs
                )
                paths.extend(sub_paths)
                weights.extend(sub_weights * weight)
                hashes.extend(sub_hashs)
    if len(paths) != len(set(paths)):
        LOGGER.warning("Core datasets loaded multiple times")
    # normalize weights and return
    weights = np.array(weights)
    props = weights / weights.sum()
    return paths, props, hashes


def load_dataset(ds_path):
    """Parse either core dataset or dataset config"""
    ds_path = util.resolve_path(ds_path)
    if not os.path.exists(ds_path):
        raise RemoraError(f"Dataset path does not exist. {ds_path}")
    if os.path.isdir(ds_path):
        return [ds_path], np.ones(1, dtype=float), None
    return parse_dataset_config(ds_path)


def compute_best_split(total_size, props):
    """Compute best split of total size into len(props) integers where each
    value is >=1 and approxiamtely closeset to props.
    """
    if total_size < len(props):
        raise RemoraError(
            f"total_size ({total_size}) smaller than number of proportions "
            f"{len(props)}"
        )
    sizes = np.floor(total_size * props).astype(int)
    # if any sizes are empty add at least 1 chunk
    sizes[sizes == 0] = 1
    # if adding 1 to empty sizes exceeds total_size subtract from largest
    while sizes.sum() > total_size:
        sizes[np.argmax(sizes)] -= 1
    # until total size is reached add chunks to value farthest below prop
    while sizes.sum() < total_size:
        sizes[np.argmin((sizes / sizes.sum()) - props)] += 1
    return sizes


def dataloader_worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return

    ds = worker_info.dataset
    if ds.seed is not None:
        np.random.seed(worker_info.dataset.seed + worker_info.id)
    ds.super_batch_offsets = [
        np.random.randint(0, sub_ds.size) for sub_ds in ds.datasets
    ]
    # TODO jitter super batch sizes to reduce simultaneous reads
    LOGGER.debug(
        f"Dataset worker {worker_info.id} using super batch offsets "
        f"{', '.join(map(str, ds.super_batch_offsets))}"
    )


class RemoraDataset(IterableDataset):
    """Remora dataset composed of one or more CoreRemoraDatasets. Core datasets
    will be combined at fixed ratios in the batches supplied.
    """

    @property
    def num_datasets(self):
        return len(self.datasets)

    @property
    def paths(self):
        return [ds.data_path for ds in self.datasets]

    @property
    def size(self):
        return sum(ds.size for ds in self.datasets)

    @property
    def hashes(self):
        if self._hashes is None or any(
            ds_hash is None for ds_hash in self._hashes
        ):
            LOGGER.debug("Computing dataset hashes")
            self._hashes = [ds.hash(ds.data_path) for ds in self.datasets]
        return self._hashes

    @property
    def summary(self):
        return (
            f"                     size : {self.size:,}\n"
            "     modified_base_labels : "
            f"{self.metadata.modified_base_labels}\n"
            f"                mod_bases : {self.metadata.mod_bases}\n"
            f"           mod_long_names : {self.metadata.mod_long_names}\n"
            f"       kmer_context_bases : {self.metadata.kmer_context_bases}\n"
            f"            chunk_context : {self.metadata.chunk_context}\n"
            f"                   motifs : {self.metadata.motifs}\n"
            f"           reverse_signal : {self.metadata.reverse_signal}\n"
            f" chunk_extract_base_start : {self.metadata.base_start_justify}\n"
            f"     chunk_extract_offset : {self.metadata.offset}\n"
            f"          sig_map_refiner : {self.metadata.sig_map_refiner}\n"
        )

    @property
    def init_kwargs(self):
        return {
            "proportions": self.props,
            "hashes": self._hashes,
            "batch_size": self.batch_size,
            "super_batch_size": self.super_batch_size,
            "super_batch_sample_frac": self.super_batch_sample_frac,
            "seed": self.seed,
        }

    def set_global_metadata(self):
        self.metadata = self.datasets[0].metadata.copy()
        # not applicable for super dataset
        for md_name in (
            "allocate_size",
            "max_seq_len",
            "dataset_start",
            "dataset_end",
        ):
            setattr(self.metadata, md_name, None)
        self.metadata.motif_sequences, self.metadata.motif_offsets = zip(
            *[
                motif.to_tuple()
                for motif in util.merge_motifs(self.metadata.motifs)
            ]
        )
        self.metadata.check_motifs()
        for ds in self.datasets[1:]:
            # first check attrs for which exact match is required
            for attr_name in (
                "modified_base_labels",
                "base_start_justify",
                "offset",
                "reverse_signal",
                "sig_map_refiner",
            ):
                if getattr(ds.metadata, attr_name) != getattr(
                    self.metadata, attr_name
                ):
                    raise RemoraError(
                        f"All datasets must have same {attr_name} "
                        f"{getattr(ds.metadata, attr_name)} != "
                        f"{getattr(self.metadata, attr_name)}"
                    )
            if set(ds.metadata.extra_array_names) != set(
                self.metadata.extra_array_names
            ):
                raise RemoraError(
                    f"Extra arrays not equal: {ds.metadata.extra_array_names} "
                    f"!= {self.metadata.extra_array_names}"
                )
            for mb, mln in zip(
                ds.metadata.mod_bases, ds.metadata.mod_long_names
            ):
                if mb in self.metadata.mod_bases:
                    # ensure same mod long name is specified for the short name
                    md_mln = next(
                        md_mln
                        for md_mb, md_mln in zip(
                            self.metadata.mod_bases,
                            self.metadata.mod_long_names,
                        )
                        if mb == md_mb
                    )
                    assert mln == md_mln, (
                        "Mismatched modified bases.\n\tPreviously loaded "
                        f"modified bases: {self.metadata.mod_bases} "
                        f"{self.metadata.mod_long_names}\n\tNew modified "
                        f"bases: {ds.metadata.mod_bases} "
                        f"{ds.metadata.mod_long_names}"
                    )
                else:
                    # add mod base to super dataset metadata
                    self.metadata.mod_bases.append(mb)
                    self.metadata.mod_long_names.append(mln)

            # kmer_context bases can be reduced
            if (
                ds.metadata.kmer_context_bases
                != self.metadata.kmer_context_bases
            ):
                LOGGER.debug(
                    "K-mer context bases not equal. Setting to minimum values. "
                    f"{ds.metadata.kmer_context_bases} != "
                    f"{self.metadata.kmer_context_bases}"
                )
                self.metadata.kmer_context_bases = (
                    min(
                        self.metadata.kmer_context_bases[0],
                        ds.metadata.kmer_context_bases[0],
                    ),
                    min(
                        self.metadata.kmer_context_bases[1],
                        ds.metadata.kmer_context_bases[1],
                    ),
                )
            # separate chunk_context as well
            if ds.metadata.chunk_context != self.metadata.chunk_context:
                LOGGER.debug(
                    "Chunk context not equal. Setting to minimum values. "
                    f"{ds.metadata.chunk_context} != "
                    f"{self.metadata.chunk_context}"
                )
                self.metadata.kmer_context_bases = (
                    min(
                        self.metadata.chunk_context[0],
                        ds.metadata.chunk_context[0],
                    ),
                    min(
                        self.metadata.chunk_context[1],
                        ds.metadata.chunk_context[1],
                    ),
                )
            # merge motifs
            if set(ds.metadata.motifs) != set(self.metadata.motifs):
                LOGGER.debug(
                    f"Motif sets not equal: {set(ds.metadata.motifs)} "
                    f"!= {set(self.metadata.motifs)}. Merging motif sets."
                )
                (
                    self.metadata.motif_sequences,
                    self.metadata.motif_offsets,
                ) = zip(
                    *[
                        motif.to_tuple()
                        for motif in util.merge_motifs(
                            self.metadata.motifs + ds.metadata.motifs
                        )
                    ]
                )
                self.metadata.check_motifs()

        # sort modified bases alphabetically
        mod_bases, mod_long_names = [], []
        for idx in sorted(
            range(len(self.metadata.mod_bases)),
            key=self.metadata.mod_bases.__getitem__,
        ):
            mod_bases.append(self.metadata.mod_bases[idx])
            mod_long_names.append(self.metadata.mod_long_names[idx])
        self.metadata.mod_bases = mod_bases
        self.metadata.mod_long_names = mod_long_names

    def update_metadata(self, other):
        for md_key in (
            "modified_base_labels",
            "offset",
            "reverse_signal",
            "sig_map_refiner",
        ):
            if getattr(self.metadata, md_key) != getattr(
                other.metadata, md_key
            ):
                raise RemoraError(
                    f"Cannot update metadata with mismatching '{md_key}'. "
                    f"({getattr(self.metadata, md_key)} != "
                    f"{getattr(other.metadata, md_key)})"
                )
        for ds in self.datasets:
            ds.update_metadata(other)
        for md_key in (
            "mod_bases",
            "mod_long_names",
            "extra_arrays",
            "kmer_context_bases",
            "chunk_context",
        ):
            setattr(self.metadata, md_key, getattr(other.metadata, md_key))

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.batch_sizes = compute_best_split(self.batch_size, self.props)
        bs_str = "\n".join(
            (
                f"{bs}\t{ds.data_path}"
                for bs, ds in zip(self.batch_sizes, self.datasets)
            )
        )
        LOGGER.debug(f"Dataset batch sizes:\n{bs_str}")

    def __init__(
        self,
        datasets,
        proportions,
        hashes=None,
        batch_size=constants.DEFAULT_BATCH_SIZE,
        super_batch_size=constants.DEFAULT_SUPER_BATCH_SIZE,
        super_batch_sample_frac=None,
        seed=None,
    ):
        super(RemoraDataset).__init__()
        self.datasets = datasets
        self.props = proportions
        if not all(0 <= prop <= 1 for prop in self.props):
            raise RemoraError("Dataset proportions must be between 0 and 1.")
        if len(self.datasets) != len(self.props):
            raise RemoraError("Dataset and proportions must be same length.")
        self._hashes = hashes
        self.set_batch_size(batch_size)
        self.super_batch_size = super_batch_size
        self.super_batch_sample_frac = super_batch_sample_frac
        self.seed = seed

        # RemoraDataset is infinite iter if all core datasets are infinite
        self.infinite_iter = all(ds.infinite_iter for ds in self.datasets)
        self.set_global_metadata()
        # apply applicable global metadata to sub-datasets
        for ds in self.datasets:
            ds.update_metadata(self)
        self.super_batch_offsets = [0 for ds in self.datasets]
        self._ds_iters = None
        self._iter = None
        self._all_batches = None

    @classmethod
    def from_config(
        cls,
        config_path,
        override_metadata=None,
        ds_kwargs=None,
        **kwargs,
    ):
        paths, props, hashes = parse_dataset_config(config_path)
        LOGGER.debug(f"Loaded dataset paths: {', '.join(paths)}")
        LOGGER.debug(
            f"Loaded dataset proportions: {', '.join(map(str, props))}"
        )
        LOGGER.debug(f"Loaded dataset hashes: {', '.join(map(str, hashes))}")
        if override_metadata is None:
            override_metadata = {}
        if ds_kwargs is None:
            ds_kwargs = {}
        datasets = [
            CoreRemoraDataset(
                ds_path,
                override_metadata=override_metadata.copy(),
                **ds_kwargs,
            )
            for ds_path in paths
        ]
        label_summaries = "\n".join(ds.label_summary for ds in datasets)
        LOGGER.debug(f"Loaded dataset label summaries:\n{label_summaries}")
        return cls(datasets, props, hashes, **kwargs)

    def train_test_split(self, num_test_chunks, override_metadata=None):
        test_sizes = compute_best_split(num_test_chunks, self.props)
        if override_metadata is None:
            override_metadata = {}
        train_datasets, test_datasets = [], []
        for ds, test_size in zip(self.datasets, test_sizes):
            if test_size >= ds.size:
                raise RemoraError("Not enough chunks")
            trn_md = override_metadata.copy()
            trn_md["dataset_start"] = ds.metadata.dataset_start + test_size
            LOGGER.debug(f"train split override metadata: {trn_md}")
            train_datasets.append(
                CoreRemoraDataset(ds.data_path, override_metadata=trn_md)
            )
            test_md = override_metadata.copy()
            test_md["dataset_end"] = ds.metadata.dataset_start + test_size
            LOGGER.debug(f"test split override metadata: {test_md}")
            test_datasets.append(
                CoreRemoraDataset(
                    ds.data_path,
                    infinite_iter=False,
                    override_metadata=test_md,
                )
            )
        return RemoraDataset(train_datasets, **self.init_kwargs), RemoraDataset(
            test_datasets, **self.init_kwargs
        )

    def head(self, num_chunks, override_metadata=None):
        ds_sizes = compute_best_split(num_chunks, self.props)
        if override_metadata is None:
            override_metadata = {}
        head_datasets = []
        for ds, ds_size in zip(self.datasets, ds_sizes):
            if ds_size >= ds.size:
                raise RemoraError("Not enough chunks")
            head_md = override_metadata.copy()
            head_md["dataset_start"] = ds.metadata.dataset_start
            head_md["dataset_end"] = ds.metadata.dataset_start + ds_size
            head_datasets.append(
                CoreRemoraDataset(
                    ds.data_path, infinite_iter=False, override_metadata=head_md
                )
            )
        return RemoraDataset(head_datasets, **self.init_kwargs)

    def _set_sub_ds_iters(self):
        for ds, bs, sb_offset in zip(
            self.datasets, self.batch_sizes, self.super_batch_offsets
        ):
            ds.batch_size = bs
            ds.super_batch_offset = sb_offset
            ds.super_batch_size = self.super_batch_size
            ds.super_batch_sample_frac = self.super_batch_sample_frac
        self._ds_iters = [ds.iter_batches() for ds in self.datasets]

    def iter_batches(self, return_arrays=("enc_kmers", "signal", "labels")):
        if self._ds_iters is None:
            self._set_sub_ds_iters()
        while True:
            try:
                ds_arrays = [next(ds) for ds in self._ds_iters]
            except StopIteration:
                break
            yield [
                torch.from_numpy(
                    np.concatenate([arr[arr_name] for arr in ds_arrays])
                )
                for arr_name in return_arrays
            ]

    def load_all_batches(self):
        if self.infinite_iter:
            raise RemoraError("Cannot save all batches for infinite dataset")
        self._set_sub_ds_iters()
        self._all_batches = list(self.iter_batches())
        for ds in self.datasets:
            ds.close_memmaps()

    def __iter__(self):
        if self._all_batches is not None:
            self._iter = iter(self._all_batches)
            return self._iter
        # if first time calling iter or if this is an exhaustible dataset
        # re-initialize the iterator
        if self._iter is None or not self.infinite_iter:
            self._set_sub_ds_iters()
            self._iter = self.iter_batches()
        return self._iter

    def __next__(self):
        return next(self._iter)

    def get_label_counts(self):
        label_counts = np.zeros(self.metadata.num_labels, dtype=int)
        if self._all_batches is not None:
            for _, _, b_labels in self._all_batches:
                for idx, idx_cnt in enumerate(np.bincount(b_labels)):
                    label_counts[idx] += idx_cnt
            return label_counts
        for ds in self.datasets:
            for idx, count in enumerate(ds.get_label_counts()):
                label_counts[idx] += count
        return label_counts

    @property
    def label_summary(self):
        return "; ".join(
            f"{self.metadata.labels[lab_idx]}:{count:,}"
            for lab_idx, count in enumerate(self.get_label_counts())
        )

    def get_config(self):
        return [
            (ds_path, ds_prop)
            if ds_hash is None
            else (ds_path, ds_prop, ds_hash)
            for ds_path, ds_prop, ds_hash in zip(
                self.paths, self.props, self.hashes
            )
        ]

    def epoch_summary(self, batches_per_epoch):
        epoch_chunk_totals = [
            batches_per_epoch * ds_chunks_per_batch
            for ds_chunks_per_batch in self.batch_sizes
        ]
        dss_lab_counts = [
            dict(
                zip(
                    ds.metadata.labels,
                    ds.get_label_counts(),
                )
            )
            for ds in self.datasets
        ]
        dss_lab_props = []
        for ds_lab_counts in dss_lab_counts:
            ds_tot = sum(ds_lab_counts.values())
            dss_lab_props.append(
                dict((lab, cnt / ds_tot) for lab, cnt in ds_lab_counts.items())
            )
        # compute the number of chunks of each label extracted from each dataset
        # each batch
        batch_lab_cols = [
            "\t".join(
                f"{np.ceil(ds_lp.get(lab, 0) * ds_bs).astype(int):,}"
                for lab in self.metadata.labels
            )
            for ds_lp, ds_bs in zip(dss_lab_props, self.batch_sizes)
        ]
        dss_lab_cols = [
            "\t".join(f"{ds_lc.get(lab, 0):,}" for lab in self.metadata.labels)
            for ds_lc in dss_lab_counts
        ]
        summ_strs = [
            f"{ds_chunks_per_epoch/ds.size:10.4%}\t"
            f"{b_lab_cols}\t"
            f"{ds_chunks_per_epoch:,}\t"
            f"{ds.size:,}\t"
            f"{ds_lab_cols}\t"
            f"{ds.data_path}"
            for ds_chunks_per_epoch, b_lab_cols, ds, ds_lab_cols in zip(
                epoch_chunk_totals,
                batch_lab_cols,
                self.datasets,
                dss_lab_cols,
            )
        ]
        b_labels_header = "\t".join(
            (f"batch_{lab}" for lab in self.metadata.labels)
        )
        ds_labels_header = "\t".join(
            (f"dataset_{lab}" for lab in self.metadata.labels)
        )
        return (
            f"percent_of_dataset_per_epoch\t{b_labels_header}\t"
            f"dataset_chunks_per_epoch\tdataset_size\t{ds_labels_header}\t"
            "path\n"
        ) + "\n".join(summ_strs)
