import re
import os
import json
import hashlib
import operator
import dataclasses
from glob import glob
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import IterableDataset

from remora.refine_signal_map import SigMapRefiner
from remora.data_chunks_core import trim_sb_chunk_context_core
from remora import constants, log, RemoraError, util, encoded_kmers

LOGGER = log.get_logger()

DATASET_VERSION = 4
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
        read_metrics (dict): Metrics related to this read. See util.READ_METRICS

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
    read_metrics: dict = None

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
            focus_bases=(
                None if self.focus_bases is None else self.focus_bases.copy()
            ),
        )

    def refine_signal_mapping(self, sig_map_refiner, check_read=False):
        if not sig_map_refiner.is_loaded:
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
            LOGGER.debug(
                f"Refine mapping ::: shift: {prev_shift} -> {self.shift} "
                f"scale: {prev_scale} -> {self.scale}"
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
        signal_padding=False,
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
                fill_en = self.sig.size - sig_start + seq_to_sig_offset
                sig_end = self.sig.size
            chunk_sig[fill_st:fill_en] = self.sig[sig_start:sig_end]
            if signal_padding:
                chunk_sig[:fill_st] = self.sig[
                    sig_start + fill_st : sig_start : -1
                ]
                chunk_sig[fill_en:] = self.sig[
                    sig_end : sig_end - chunk_sig.size + fill_en - 1 : -1
                ]

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
            modbase_label=label,
            read_metrics=self.read_metrics,
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

    def iter_basecall_chunks(
        self,
        chunk_context,
        kmer_context_bases,
        max_chunks_per_read,
        random_offsets=False,
        check_chunks=False,
    ):
        """Iterate over chunks defined by signal position either evenly spaced
        or randomly selected over the read.
        """
        chunk_width = sum(chunk_context)
        num_chunks = min(self.sig.size // chunk_width, max_chunks_per_read)
        if random_offsets:
            chunk_offsets = np.random.randint(
                0, self.sig.size - chunk_width, num_chunks
            )
        else:
            chunk_offsets = np.linspace(
                0, self.sig.size - chunk_width, num_chunks, endpoint=True
            ).astype(int)
        # shift chunk by first offset (this is generally 0 though)
        chunk_offsets += chunk_context[0]
        # TODO add accuracy and other relevant metrics to chunk data
        for chunk_offset in chunk_offsets:
            try:
                yield self.extract_chunk(
                    chunk_offset,
                    chunk_context,
                    kmer_context_bases,
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
            batch_size=batch_size,
            metadata=DatasetMetadata(
                allocate_size=len(chunks),
                max_seq_len=max(c.seq_len for c in chunks),
                mod_bases=model_metadata["mod_bases"],
                mod_long_names=model_metadata["mod_long_names"],
                motif_sequences=motif_seqs,
                motif_offsets=motif_offsets,
                chunk_context=model_metadata["chunk_context"],
                kmer_context_bases=model_metadata["kmer_context_bases"],
                extra_metadata_arrays={
                    "modbase_label": ("int64", "Modified base label"),
                    "read_focus_base": (
                        "int64",
                        "Position within read training sequence",
                    ),
                },
            ),
            return_arrays=[
                "signal",
                "modbase_label",
                "read_focus_base",
                "enc_kmer",
            ],
            infinite_iter=False,
        )
        for chunk in chunks:
            dataset.write_chunk(chunk)
        for batch in dataset:
            self.batches.append(
                (
                    batch["signal"],
                    batch["enc_kmer"],
                    batch["modbase_label"],
                    batch["read_focus_base"],
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
        modbase_label (int): Integer label for training/validation.
        read_metrics (dict): Metrics related to this read. See util.READ_METRICS
    """

    signal: np.ndarray
    seq_w_context: np.ndarray
    seq_to_sig_map: np.ndarray
    kmer_context_bases: tuple
    chunk_sig_focus_idx: int
    chunk_focus_base: int
    read_focus_base: int
    read_id: str = None
    modbase_label: int = None
    read_metrics: dict = None
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
        dataset_type (str): Type of dataset. Currently "modbase" and "sequence"
            are the two accepted values. "modbase" datasets enable certain
            specialized functionalities. For example merging different labeling
            schemas.
        extra_signal_arrays (dict): Extra arrays with the same dimensionality as
            the signal array (chunk_size). For example, to support duplex model
            inputs.
        extra_metadata_arrays (dict): Extra arrays to store metadata information
            about chunks. Dict keys define the name of the extra data and
            values contain the string dtype of the array and a description of
            the data to self document the dataset. Metadata values are
            1-dimensional.
        extra_sequence_arrays (dict): Extra arrays with the same dimensionality
            as the sequence array.
        chunk_context (tuple): 2-tuple containing the number of signal points
            before and after the central position.
        base_start_justify (bool): Extract chunk centered on start of base
        offset (int): Extract chunk centered on base offset from base of
            interest
        kmer_context_bases (tuple): 2-tuple containing the bases to include in
            the encoded k-mer presented as input.
        reverse_signal (bool): Is nanopore signal 3' to 5' orientation?
            Primarily for direct RNA
        pa_scaling (tuple): Zero-centered picoamp scaling factors. These should
            be extracted from a Dorado (v4.3+) basecalling model.
        sig_map_refiner (remora.refine_signal_map.SigMapRefiner): Signal
            mapping refiner
        description (str): Global description of the dataset
    """

    # dataset attributes
    allocate_size: int
    max_seq_len: int

    # modbase attributes
    mod_bases: list = None
    mod_long_names: list = None
    motif_sequences: list = None
    motif_offsets: list = None

    dataset_start: int = 0
    dataset_end: int = 0
    version: int = DATASET_VERSION
    dataset_type: str = constants.DATASET_TYPE_MODBASE
    # extra arrays
    extra_signal_arrays: dict = None
    extra_metadata_arrays: dict = None
    extra_sequence_arrays: dict = None
    # chunk extract
    chunk_context: tuple = constants.DEFAULT_CHUNK_CONTEXT
    base_start_justify: bool = False
    offset: int = 0
    kmer_context_bases: tuple = constants.DEFAULT_KMER_CONTEXT_BASES
    reverse_signal: bool = False
    # signal scaling/refinement
    pa_scaling: tuple = None
    sig_map_refiner: SigMapRefiner = None
    rough_rescale_method: str = constants.DEFAULT_ROUGH_RESCALE_METHOD
    description: str = None

    _stored_kmer_context_bases: tuple = None
    _stored_chunk_context: tuple = None

    @property
    def is_modbase_dataset(self):
        return self.dataset_type == constants.DATASET_TYPE_MODBASE

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
    def modbase_labels(self):
        if self.is_modbase_dataset:
            return ["control"] + self.mod_long_names
        raise RemoraError(
            "Labels attribute not defined for datasets which are not modified "
            "base type."
        )

    @property
    def num_labels(self):
        if self.is_modbase_dataset:
            return len(self.mod_long_names) + 1
        raise RemoraError(
            "Labels attribute not defined for datasets which are not modified "
            "base type."
        )

    @property
    def motifs(self):
        if self.is_modbase_dataset:
            return list(zip(self.motif_sequences, self.motif_offsets))
        raise RemoraError(
            "Motifs attribute not defined for datasets which are not modified "
            "base type."
        )

    @property
    def num_motifs(self):
        if self.is_modbase_dataset:
            return len(self.motif_sequences)
        raise RemoraError(
            "Motifs attribute not defined for datasets which are not modified "
            "base type."
        )

    @property
    def extra_array_names(self):
        arr_names = set()
        if self.extra_signal_arrays is not None:
            arr_names.update(self.extra_signal_arrays.keys())
        if self.extra_metadata_arrays is not None:
            arr_names.update(self.extra_metadata_arrays.keys())
        if self.extra_sequence_arrays is not None:
            arr_names.update(self.extra_sequence_arrays.keys())
        return arr_names

    def extra_arrays_intersection_update(self, other):
        if self.extra_signal_arrays is not None:
            for esak in set(self.extra_signal_arrays).difference(
                other.extra_signal_arrays
            ):
                self.extra_signal_arrays.pop(esak)
        if self.extra_metadata_arrays is not None:
            for emak in set(self.extra_metadata_arrays).difference(
                other.extra_metadata_arrays
            ):
                self.extra_metadata_arrays.pop(emak)
        if self.extra_sequence_arrays is not None:
            for emak in set(self.extra_sequence_arrays).difference(
                other.extra_sequence_arrays
            ):
                self.extra_sequence_arrays.pop(emak)

    @property
    def extra_array_dtypes(self):
        arr_dtypes = {}
        if self.extra_signal_arrays is not None:
            for name, (dtype, _) in self.extra_signal_arrays.items():
                arr_dtypes[name] = dtype
        if self.extra_metadata_arrays is not None:
            for name, (dtype, _) in self.extra_metadata_arrays.items():
                arr_dtypes[name] = dtype
        if self.extra_sequence_arrays is not None:
            for name, (dtype, _) in self.extra_sequence_arrays.items():
                arr_dtypes[name] = dtype
        return arr_dtypes

    def _size(self, mode="r"):
        return self.allocate_size if mode == "w" else self.dataset_end

    def signal_shape(self, mode="r"):
        return self._size(mode), 1, self.stored_chunk_width

    @property
    def sequence_width(self):
        return self.max_seq_len + sum(self.stored_kmer_context_bases)

    def sequence_shape(self, mode="r"):
        return self._size(mode), self.sequence_width

    @property
    def sequence_to_signal_mapping_width(self):
        return self.max_seq_len + 1

    def sequence_to_signal_mapping_shape(self, mode="r"):
        return self._size(mode), self.sequence_to_signal_mapping_width

    def sequence_lengths_shape(self, mode="r"):
        return tuple((self._size(mode),))

    def extras_shape(self, mode="r"):
        return tuple((self._size(mode),))

    def check_motifs(self):
        if not self.is_modbase_dataset:
            return
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
        if self.is_modbase_dataset:
            # Support original single letter codes or new list short names
            # (including ChEBI codes)
            if isinstance(self.mod_bases, str):
                self.mod_bases = list(self.mod_bases)
            self.mod_bases = list(map(str, self.mod_bases))
            assert len(self.mod_bases) == len(self.mod_long_names), (
                f"mod_bases ({self.mod_bases}) must be the same length as "
                f"mod_long_names ({self.mod_long_names})"
            )
            self.check_motifs()
        self.chunk_context = tuple(self.chunk_context)
        self.kmer_context_bases = tuple(self.kmer_context_bases)
        if self._stored_chunk_context is not None:
            self._stored_chunk_context = tuple(self._stored_chunk_context)
        if self._stored_kmer_context_bases is not None:
            self._stored_kmer_context_bases = tuple(
                self._stored_kmer_context_bases
            )

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


class DatasetFilters:
    """Filters to be applied to a CoreRemoraDataset at retrieval time"""

    # derived columns potentially accessing multiple arrays
    _derived_cols = {
        "samples_per_base": lambda sb: sb["signal"].shape[2]
        / sb["sequence_lengths"]
    }

    def __init__(self, filters=None):
        self.filters = filters

    @property
    def filter_columns(self):
        return [
            col for col, _, _ in self.filters if col not in self._derived_cols
        ]

    @property
    def storage_filters(self):
        """Convert operators to string for storage"""
        return [(col, op.__name__, thresh) for col, op, thresh in self.filters]

    @classmethod
    def from_raw_filters(cls, raw_filters, dataset=None):
        if raw_filters is None:
            return cls()
        return cls(DatasetFilters.parse_filters(raw_filters, dataset))

    @staticmethod
    def parse_filters(raw_filters, dataset=None):
        filters = []
        for filt_i in raw_filters:
            if len(filt_i) == 4:
                col, op_str, thresh, is_quantile = filt_i
            elif len(filt_i) == 3:
                col, op_str, thresh = filt_i
                is_quantile = False
            else:
                raise RemoraError(f"Invalid filter length {len(filt_i)}")
            op = getattr(operator, op_str)
            if is_quantile:
                if dataset is None:
                    raise RemoraError(
                        "Dataset must be provided for quantile filter value"
                    )
                try:
                    col_arr = DatasetFilters._derived_cols[col](
                        dataset.arrays_dict
                    )
                except KeyError:
                    try:
                        st = dataset.metadata.dataset_start
                        en = dataset.metadata.dataset_end
                        col_arr = getattr(dataset, col)[st:en]
                    except AttributeError:
                        raise RemoraError(
                            f"Dataset does not contain column: {col}"
                        )
                # TODO potentially perform this operation lazily on first access
                thresh = np.quantile(col_arr, thresh)
                LOGGER.debug(
                    f'Quantile filter set to: "{col}" {op_str} {thresh}'
                )
            elif (
                dataset is not None
                and col not in DatasetFilters._derived_cols
                and col not in dataset
            ):
                raise RemoraError(f"Dataset does not contain column: {col}")
            filters.append((col, op, thresh))
        return filters

    @classmethod
    def from_file(cls, filters_path):
        if filters_path is None:
            return
        with open(filters_path) as filters_fh:
            raw_filters = json.load(filters_fh)
        return DatasetFilters.from_raw_filters(raw_filters)

    @property
    def derived_filters(self):
        if self.filters is None:
            return
        return [filt for filt in self.filters if filt[0] in self._derived_cols]

    @property
    def fixed_filters(self):
        if self.filters is None:
            return
        return [
            filt for filt in self.filters if filt[0] not in self._derived_cols
        ]

    @staticmethod
    def _apply_filter_rows(super_batch, filt_arrs):
        filt_arrs = np.logical_and.reduce(filt_arrs)
        indices = np.nonzero(filt_arrs)[0]
        for col in super_batch.keys():
            super_batch[col] = np.take(super_batch[col], indices, axis=0)

    def get_fixed_filter_rows(self, super_batch):
        filt_arrs = []
        for col, op, thresh in self.fixed_filters:
            try:
                col_arr = super_batch[col]
            except KeyError:
                raise RemoraError(f"Super batch does not contain column: {col}")
            filt_arrs.append(op(col_arr, thresh))
        return filt_arrs

    def apply_fixed_filters(self, super_batch):
        if self.filters is None:
            return
        self._apply_filter_rows(
            super_batch, self.get_fixed_filter_rows(super_batch)
        )

    def get_derived_filter_rows(self, super_batch):
        filt_arrs = []
        for col, op, thresh in self.derived_filters:
            col_arr = self._derived_cols[col](super_batch)
            filt_arrs.append(op(col_arr, thresh))
        return filt_arrs

    def apply_derived_filters(self, super_batch):
        if self.filters is None:
            return
        self._apply_filter_rows(
            super_batch, self.get_derived_filter_rows(super_batch)
        )

    def apply_filters(self, super_batch):
        if self.filters is None:
            return
        self._apply_filter_rows(
            super_batch,
            self.get_fixed_filter_rows(super_batch)
            + self.get_derived_filter_rows(super_batch),
        )


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
        raise RemoraError("Sequence min must greater than -2")


@dataclasses.dataclass
class CoreRemoraDataset:
    """CoreRemoraDataset manages the storage and access to directory of
    training data.

    Args:
        data_path (str): Path to dataset stored on disk
        mode (str): Mode for dataset. "r"ead (default) or "w"rite
        metadata (DatasetMetadata): Metadata associated with this dataset
        override_metadata (dict): Values to override when reading metadata from
            dataset. Only particular override options are valid. See
            CoreRemoraDataset.load_metadata for details.
        batch_size (int): Number of chunks to return from this dataset. This
            can be specified during the CoreRemoraDataset.extract_batch method,
            but when using iterators this object global value will be used.
        super_batch_size (int): Number of chunks to read from disk at one time.
        super_batch_sample_frac (float): Fraction of reads to retain from super
            batch. Lower values allow greater randomization of chunks presented
            within a batch through training, but require more disk IO.
        super_batch_offset (int): Offset within the dataset to start iteration.
            Note that this is relative to metadata.dataset_start for sliced
            datasets. This allows multiple dataloaders to avoid supplying the
            same chunks.
        infinite_iter (bool): Iterate through chunks in an infinite loop via
            wrapping around the end of the dataset. False value will iterate to
            the end of the dataset and stop.
        do_check_super_batches (bool): Check super batches after loading?
        return_arrays (list): Specify arrays to return from batch
            extraction/iteration methods. Set this value with set_return_arrays
            method.
        filters_path (str): Path to a filters file. If not provided the default
            location within the dataset directory will be checked.
        filters (DatasetFilters): Parsed dataset filters object.
    """

    data_path: str = None
    mode: str = "r"
    metadata: DatasetMetadata = None
    override_metadata: dict = None
    batch_size: int = None
    super_batch_size: int = constants.DEFAULT_SUPER_BATCH_SIZE
    super_batch_sample_frac: float = None
    super_batch_offset: int = 0
    infinite_iter: bool = True
    do_check_super_batches: bool = False
    return_arrays: list = None
    filters_path: str = None
    filters: DatasetFilters = None

    # attributes to hold current super batch
    _sb_iter = None
    _curr_sb = None
    _curr_sb_offset = None

    _signal_core_array = "signal"
    _sequence_core_array = "sequence"
    _core_dtypes = {
        _signal_core_array: np.float32,
        _sequence_core_array: np.int8,
        "sequence_to_signal_mapping": np.int16,
        "sequence_lengths": np.int16,
    }
    _core_arrays = list(_core_dtypes.keys())

    _filters_path = "filters.jsn"

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
        # support deprecated modbase labels which used to be a core data type
        deprecated_labels_path = os.path.join(data_path, "labels.npy")
        if os.path.exists(deprecated_labels_path):
            paths.append(deprecated_labels_path)
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
    def filters_path_resolved(self):
        if self.filters_path is not None:
            return self.filters_path
        if self.data_path is None:
            raise RemoraError("No path available for in-memory dataset")
        return os.path.join(self.data_path, self._filters_path)

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
        return self._core_arrays + list(self.metadata.extra_array_names)

    @property
    def extra_sig_return_array_names(self):
        return list(
            set(self.return_arrays).intersection(
                self.metadata.extra_signal_arrays
            )
        )

    @property
    def extra_metadata_return_array_names(self):
        return list(
            set(self.return_arrays).intersection(
                self.metadata.extra_metadata_arrays
            )
        )

    @property
    def extra_seq_return_array_names(self):
        return list(
            set(self.return_arrays).intersection(
                self.metadata.extra_sequence_arrays
            )
        )

    @property
    def output_return_arrays(self):
        """Output return arrays requested from this dataset. This includes
        converting sequence output encoding names.
        """
        out_r_arrs = []
        for arr_name in self.return_arrays:
            try:
                out_r_arrs.extend(constants.DATASET_SEQ_OUTPUTS[arr_name])
            except KeyError:
                out_r_arrs.append(arr_name)
        return out_r_arrs

    @property
    def valid_return_arrays(self):
        """Set of return arrays that can be supplied from this dataset"""
        return set(constants.DATASET_SEQ_OUTPUTS).union(
            self.metadata.extra_array_names
        )

    @property
    def super_batch_load_arrays(self):
        """Convert sequence arrays to specified output names"""
        core_arrays = self._core_arrays.copy() + list(
            constants.DATASET_SEQ_OUTPUTS
        )
        sb_load_arrs = self._core_arrays.copy()
        for arr_name in self.return_arrays:
            if arr_name in core_arrays:
                continue
            sb_load_arrs.append(arr_name)
        # add filter columns to be loaded
        if self.filters is not None:
            for arr_name, _, _ in self.filters.storage_filters:
                if (
                    arr_name in DatasetFilters._derived_cols
                    or arr_name in sb_load_arrs
                ):
                    continue
                sb_load_arrs.append(arr_name)
        return sb_load_arrs

    @property
    def arrays(self):
        """Generator of chunk arrays in dataset. Arrays will be sliced to
        current dataset size not allocated arrays. Memory mapped ararys are
        returned.
        """
        for array_name in self.array_names:
            yield getattr(self, array_name)[
                self.metadata.dataset_start : self.metadata.dataset_end
            ]

    @property
    def arrays_dict(self):
        """Generator of chunk arrays in dataset. Arrays will be sliced to
        current dataset size not allocated arrays. Memory mapped ararys are
        returned.
        """
        arr_dict = {}
        for array_name in self.array_names:
            arr_dict[array_name] = getattr(self, array_name)[
                self.metadata.dataset_start : self.metadata.dataset_end
            ]
        return arr_dict

    @property
    def arrays_info(self):
        arrays_info = []
        for name, dtype in self._core_dtypes.items():
            arrays_info.append(
                (
                    name,
                    dtype,
                    getattr(self.metadata, f"{name}_shape")(self.mode),
                )
            )
        if self.metadata.extra_signal_arrays is not None:
            for name, (dtype, _) in self.metadata.extra_signal_arrays.items():
                arrays_info.append(
                    (name, dtype, self.metadata.signal_shape(self.mode))
                )
        if self.metadata.extra_metadata_arrays is not None:
            for name, (dtype, _) in self.metadata.extra_metadata_arrays.items():
                arrays_info.append(
                    (name, dtype, self.metadata.extras_shape(self.mode))
                )
        if self.metadata.extra_sequence_arrays is not None:
            for name, (dtype, _) in self.metadata.extra_sequence_arrays.items():
                arrays_info.append(
                    (name, dtype, self.metadata.sequence_shape(self.mode))
                )
        return arrays_info

    @property
    def summary(self):
        summ_txt = (
            f"                data_path : {self.data_path}\n"
            f"                     size : {self.size:,}\n"
            f"            dataset start : {self.metadata.dataset_start:,}\n"
            f"              dataset end : {self.metadata.dataset_end:,}\n"
            f"       kmer context bases : {self.metadata.kmer_context_bases}\n"
            f"            chunk context : {self.metadata.chunk_context}\n"
            f"           reverse signal : {self.metadata.reverse_signal}\n"
            f" chunk extract base start : {self.metadata.base_start_justify}\n"
            f"     chunk extract offset : {self.metadata.offset}\n"
            f"          sig map refiner : {self.metadata.sig_map_refiner}\n"
        )
        # add modbase-specific metadata
        if self.metadata.is_modbase_dataset:
            summ_txt += (
                f"                mod_bases : {self.metadata.mod_bases}\n"
                f"           mod long names : {self.metadata.mod_long_names}\n"
                "       is modbase dataset? : "
                f"{self.metadata.is_modbase_dataset}\n"
                f"    mod label distribution : {self.modbase_label_summary}\n"
                f"                   motifs : {self.metadata.motifs}\n"
            )
        return summ_txt

    def load_filters(self):
        try:
            if not os.path.exists(self.filters_path_resolved):
                return
        except RemoraError:
            # in-memory dataset
            return
        self.filters = DatasetFilters.from_file(self.filters_path_resolved)

    def get_extra_counts(self, arr_name="label"):
        """Get bincount of categorical metadata array"""
        ds_labels = getattr(self, arr_name)[
            self.metadata.dataset_start : self.metadata.dataset_end
        ]
        return np.bincount(ds_labels)

    def get_modbase_label_counts(self):
        """Get bincount of modbase labels array, applying label conversion if
        necessary.
        """
        ds_labels = self.modbase_label[
            self.metadata.dataset_start : self.metadata.dataset_end
        ]
        if self.modbase_label_conv is not None:
            ds_labels = self.modbase_label_conv[ds_labels]
        return np.bincount(ds_labels)

    @property
    def modbase_label_summary(self):
        return "; ".join(
            f"{self.metadata.modbase_labels[lab_idx]}:{count:,}"
            for lab_idx, count in enumerate(self.get_modbase_label_counts())
        )

    @property
    def super_batch_sample_num_chunks(self):
        if self.super_batch_sample_frac is None:
            return None
        return np.ceil(self.super_batch_size * self.super_batch_sample_frac)

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
            # support old metadata format
            if loaded_metadata["version"] == 3:
                is_modbase_dataset = loaded_metadata["modified_base_labels"]
                if not is_modbase_dataset:
                    raise RemoraError(
                        "v3 non-modified base datasets not supported"
                    )
                del loaded_metadata["modified_base_labels"]
                loaded_metadata["dataset_type"] = constants.DATASET_TYPE_MODBASE

                loaded_metadata["extra_metadata_arrays"] = loaded_metadata[
                    "extra_arrays"
                ]
                del loaded_metadata["extra_arrays"]
                # add previously core labels array to extras array
                loaded_metadata["extra_metadata_arrays"]["modbase_label"] = (
                    "int64",
                    "Modified base label",
                )

        if loaded_metadata.get("version") != DATASET_VERSION:
            if loaded_metadata.get("version") == 3:
                LOGGER.warning(
                    "Support for v3 Remora datasets will be deprecated in a "
                    "future release."
                )
            else:
                raise RemoraError(
                    f"Remora dataset version ({loaded_metadata.get('version')})"
                    f" does not match current distribution ({DATASET_VERSION})"
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
                if (
                    self.override_metadata["mod_long_names"] is None
                    and md_val is None
                ):
                    continue
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
                    self.modbase_label_conv = np.empty(
                        self.metadata.num_labels, dtype=np.int64
                    )
                    self.modbase_label_conv[0] = 0
                    for in_lab, mod_base in enumerate(self.metadata.mod_bases):
                        # apply at super chunks and label access
                        self.modbase_label_conv[in_lab + 1] = next(
                            idx + 1
                            for idx, mb in enumerate(md_val)
                            if mb == mod_base
                        )
                    LOGGER.debug(
                        f"Setting label conversion: {self.modbase_label_conv} "
                        f"{self.data_path}"
                    )
            elif md_key == "mod_long_names":
                assert "mod_bases" in self.override_metadata
            elif md_key.startswith("extra_"):
                if md_val is not None:
                    missing_arrays = set(md_val).difference(
                        loaded_metadata[md_key]
                    )
                    if len(missing_arrays) > 0:
                        raise RemoraError(
                            "Cannot load missing arrays: "
                            f"{', '.join(missing_arrays)}\n"
                            "Available extra arrays: "
                            f"{', '.join(loaded_metadata[md_key].keys())}"
                        )
                    md_val = dict(
                        (k, loaded_metadata[md_key][k]) for k in md_val
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
            if (
                md_key in ("extra_signal_arrays", "extra_sequence_arrays")
                and loaded_metadata["version"] == 3
            ):
                LOGGER.debug(f"Initializing {md_key} for version 3 dataset")
                loaded_metadata[md_key] = None
            elif loaded_metadata[md_key] != md_val:
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
                    "extra_signal_arrays",
                    "extra_metadata_arrays",
                    "extra_sequence_arrays",
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
            # associated attributes (modbase_label_conv etc)
            self.load_metadata()

    def get_array_path(self, array_name):
        if self.data_path is None:
            raise RemoraError("No path available for in-memory dataset")
        if array_name in self._core_arrays:
            return os.path.join(self.data_path, f"{array_name}.npy")
        elif array_name == "modbase_label":
            # handle old or new format
            deprecated_labels_path = os.path.join(self.data_path, "labels.npy")
            if os.path.exists(deprecated_labels_path):
                return deprecated_labels_path
            return os.path.join(self.data_path, "extra_modbase_label.npy")
        elif array_name in self.metadata.extra_array_names:
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
                old_memmap = getattr(self, arr_name)
                if isinstance(old_memmap, np.memmap):
                    try:
                        old_memmap._mmap.close()
                    except AttributeError:
                        pass
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

    def set_return_arrays(self, return_arrays):
        if return_arrays is None:
            self.return_arrays = None
            return
        # check that return arrays are available
        invalid_return_arrays = set(return_arrays).difference(
            self.valid_return_arrays
        )
        if len(invalid_return_arrays) > 1:
            ira_str = ",".join(invalid_return_arrays)
            raise RemoraError(f"Invalid return array(s) requested: {ira_str}")
        self.return_arrays = return_arrays

    def __post_init__(self):
        self.modbase_label_conv = None
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
        self.load_filters()

    def write_batch(self, arrays):
        # TODO add explicit write buffer to this function
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
        self.metadata.dataset_end += batch_size

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
        # construct core data arrays
        chunk_dict = {
            "signal": np.expand_dims(chunk.signal, axis=0).astype(
                self._core_dtypes["signal"]
            ),
            "sequence": seq_arr,
            "sequence_to_signal_mapping": ssm_arr,
            "sequence_lengths": np.array(
                [chunk.seq_len], dtype=self._core_dtypes["sequence_lengths"]
            ),
        }
        # add all extra attributes
        for arr_name in self.metadata.extra_array_names:
            try:
                # first try direct attributes of chunk
                metric = getattr(chunk, arr_name)
            except AttributeError:
                if chunk.read_metrics is None:
                    raise RemoraError("Requested metric not available.")
                # then try read metrics dict
                metric = chunk.read_metrics.get(
                    arr_name, util.READ_METRICS[arr_name].default
                )
            chunk_dict[arr_name] = np.array(
                [metric],
                dtype=self.metadata.extra_array_dtypes[arr_name],
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
            arr_copy = np.array(array)
            for b_idx, (b_st, b_en) in enumerate(b_ranges):
                array[b_st : min(b_en, self.size)] = arr_copy[
                    shuf_indices[b_st:b_en]
                ]
                array.flush()
                if b_idx % 10 == 9:
                    # update every 10 batches
                    LOGGER.debug(
                        f"{b_idx + 1}/{len(b_ranges)} batches complete"
                    )
                if show_prog:
                    b_pb.update()
            if show_prog:
                b_pb.close()
                arr_pb.update()

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
                if self.metadata.extra_sequence_arrays is not None:
                    for arr_name in self.metadata.extra_sequence_arrays:
                        super_batch[arr_name] = super_batch[arr_name].copy()
                        super_batch[arr_name][:, :-seq_diff] = super_batch[
                            arr_name
                        ][:, seq_diff:]
            except ValueError:
                super_batch["sequence"] = super_batch["sequence"].copy()
                super_batch["sequence"][:, :-seq_diff] = super_batch[
                    "sequence"
                ][:, seq_diff:]
                if self.metadata.extra_sequence_arrays is not None:
                    for arr_name in self.metadata.extra_sequence_arrays:
                        super_batch[arr_name] = super_batch[arr_name].copy()
                        super_batch[arr_name][:, :-seq_diff] = super_batch[
                            arr_name
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
        if self.metadata.extra_signal_arrays is not None:
            for arr_name in self.metadata.extra_signal_arrays:
                super_batch[arr_name] = super_batch[arr_name][
                    :, :, st_diff:new_en
                ]
                super_batch[arr_name] = np.ascontiguousarray(
                    super_batch[arr_name]
                )

        try:
            super_batch["sequence_to_signal_mapping"] -= st_diff
        except ValueError:
            # for read only arrays from memmap make a copy
            super_batch["sequence_to_signal_mapping"] = (
                super_batch["sequence_to_signal_mapping"].copy() - st_diff
            )
            super_batch["sequence"] = super_batch["sequence"].copy()
            super_batch["sequence_lengths"] = super_batch[
                "sequence_lengths"
            ].copy()
            if self.metadata.extra_sequence_arrays is not None:
                for arr_name in self.metadata.extra_sequence_arrays:
                    super_batch[arr_name] = super_batch[arr_name].copy()

        # trimming coordinates for extra sequence arrays
        seq_clip_coords = np.empty_like(super_batch["sequence_lengths"])
        trim_sb_chunk_context_core(
            *self.metadata.stored_chunk_context,
            *self.metadata.chunk_context,
            sum(self.metadata.kmer_context_bases),
            super_batch["sequence"],
            super_batch["sequence_to_signal_mapping"],
            super_batch["sequence_lengths"],
            seq_clip_coords,
        )
        if self.metadata.extra_sequence_arrays is not None:
            for chunk_idx, st_clip in enumerate(seq_clip_coords):
                if st_clip == 0:
                    continue
                if st_clip < 0:
                    raise RemoraError(
                        "Invalid coordinate in chunk context clipping."
                    )
                for arr_name in self.metadata.extra_sequence_arrays:
                    super_batch[arr_name][chunk_idx, :-st_clip] = super_batch[
                        arr_name
                    ][chunk_idx, st_clip:]
        return super_batch

    def load_super_batch(self, offset=0, size=None):
        if self.return_arrays is None:
            raise RemoraError("Must specify return arrays")
        super_batch = {}
        if self.infinite_iter:
            offset %= self.size
        else:
            if offset >= self.size:
                raise RemoraError(
                    "Cannot extract super batch from exhausted finite dataset"
                )
        sb_arr_st = self.metadata.dataset_start + offset
        # load full dataset if size is None
        if size is None:
            if self.infinite_iter:
                raise RemoraError(
                    "Must specify size of super batch for infinite iter dataset"
                )
            size = self.metadata.dataset_end - sb_arr_st
        if size > self.size:
            size = self.size
        sb_arr_en = sb_arr_st + size
        if sb_arr_en <= self.metadata.dataset_end:
            for arr_name in self.super_batch_load_arrays:
                super_batch[arr_name] = np.array(
                    getattr(self, arr_name)[sb_arr_st:sb_arr_en]
                )
        elif self.infinite_iter:
            # wrap super batch around end of dataset
            wrap_en = sb_arr_en - self.size
            for arr_name in self.super_batch_load_arrays:
                super_batch[arr_name] = np.array(
                    np.concatenate(
                        [
                            getattr(self, arr_name)[
                                sb_arr_st : self.metadata.dataset_end
                            ],
                            getattr(self, arr_name)[
                                self.metadata.dataset_start : wrap_en
                            ],
                        ]
                    )
                )
        else:
            # return last batch with smaller batch dim
            for arr_name in self.super_batch_load_arrays:
                super_batch[arr_name] = np.array(
                    getattr(self, arr_name)[
                        sb_arr_st : self.metadata.dataset_end
                    ]
                )
        if self.super_batch_sample_num_chunks is not None:
            selected_indices = np.random.choice(
                super_batch["sequence_lengths"].size,
                min(
                    self.super_batch_sample_num_chunks,
                    super_batch["sequence_lengths"].size,
                ),
                replace=False,
            )
            for arr_name in self.super_batch_load_arrays:
                super_batch[arr_name] = super_batch[arr_name][selected_indices]
        if (
            self.metadata.is_modbase_dataset
            and self.modbase_label_conv is not None
        ):
            super_batch["modbase_label"] = self.modbase_label_conv[
                super_batch["modbase_label"]
            ]
        if constants.DATASET_SEQS_AND_LENS in self.return_arrays:
            # replace padding with -1 required for bonito processing
            for sb_idx, chunk_len in enumerate(
                super_batch["sequence_lengths"]
                + sum(self.metadata.stored_kmer_context_bases)
            ):
                super_batch["sequence"][sb_idx, chunk_len:] = -1
            if self.metadata.reverse_signal:
                # for seq and lens return need to provide signal in 3'->5'.
                # reverse_signal is stored in 5'->3' direction in RemoraDataset
                # (opposite of sequencing time)
                super_batch["signal"] = super_batch["signal"][:, ::-1]
        super_batch = self.trim_sb_kmer_context_bases(super_batch)
        super_batch = self.trim_sb_chunk_context(super_batch)
        if self.filters is not None:
            self.filters.apply_filters(super_batch)
        return super_batch

    def iter_super_batches(self):
        super_batch_num = 0
        while True:
            self.refresh_memmaps()
            super_batch = self.load_super_batch(
                self.super_batch_offset
                + (super_batch_num * self.super_batch_size),
                self.super_batch_size,
            )
            if super_batch is None:
                break
            if self.do_check_super_batches:
                check_super_batch(super_batch, self.metadata.chunk_width)
            super_batch_num += 1
            yield super_batch

    def init_super_batch_iter(self):
        if self._sb_iter is None:
            self._sb_iter = self.iter_super_batches()

    def _load_next_super_batch(self):
        self.init_super_batch_iter()
        try:
            self._curr_sb = next(self._sb_iter, None)
        except RemoraError as e:
            LOGGER.debug(f"Could not load super batch: {e}")
            self._curr_sb = None
        self._curr_sb_offset = 0

    def extract_seq_output(self, seq_out_name, seqs, seq_to_sig_maps, seq_lens):
        if seq_out_name == constants.DATASET_ENC_KMER:
            return [
                (
                    constants.DATASET_ENC_KMER,
                    encoded_kmers.compute_encoded_kmer_batch(
                        *self.metadata.kmer_context_bases,
                        seqs,
                        seq_to_sig_maps,
                        seq_lens,
                    ),
                )
            ]
        elif seq_out_name == constants.DATASET_SEQS_AND_LENS:
            # k-mer context was trimmed off in super batch. Seq lens updated
            # here to be the full sequence length.
            if self.metadata.reverse_signal:
                # TODO this may be a compute bottleneck
                for idx, seq_len in enumerate(seq_lens):
                    seqs[idx, :seq_len] = seqs[idx, :seq_len:-1]
            seq_lens += sum(self.metadata.kmer_context_bases)
            return [
                # convert to bonito alphabet "NACGT"
                ("seq", (seqs + 1).astype(np.int64)),
                ("seq_len", seq_lens.astype(np.int64)),
            ]
        else:
            raise RemoraError(
                f"Sequence output type ({seq_out_name}) not in accepted "
                f"values: {list(constants.DATASET_SEQ_OUTPUTS.keys())}"
            )

    def extract_batch(self, batch_size=None):
        """Extract a batch of training data

        Args:
            batch_size (int): Number of chunks to provide
        """

        def join_arrs(batch):
            j_batch = {}
            for arr_name, arrs in batch.items():
                num_arrs = len(arrs)
                if num_arrs == 0:
                    raise RemoraError("No arrays extracted")
                elif num_arrs == 1:
                    j_batch[arr_name] = arrs[0]
                else:
                    j_batch[arr_name] = np.concatenate(arrs, axis=0)
            return j_batch

        def update_batch(st, en=None):
            for arr_name in self.return_arrays:
                if arr_name in constants.DATASET_SEQ_OUTPUTS:
                    # add sequence output (encoded k-mers or seqs and lens)
                    for arr_name, arr_val in self.extract_seq_output(
                        arr_name,
                        self._curr_sb["sequence"][st:en],
                        self._curr_sb["sequence_to_signal_mapping"][st:en],
                        self._curr_sb["sequence_lengths"][st:en],
                    ):
                        batch[arr_name].append(arr_val)
                else:
                    batch[arr_name].append(self._curr_sb[arr_name][st:en])

        # if super batch is not loaded or is exhausted load a new one
        if (
            self._curr_sb is None
            or self._curr_sb_offset >= self._curr_sb["signal"].shape[0]
        ):
            self._load_next_super_batch()
            if self._curr_sb is None:
                return None
        if batch_size is None:
            if self.batch_size is None:
                raise RemoraError("Must provide batch size")
            batch_size = self.batch_size
        if batch_size <= 0:
            raise RemoraError("Batch size must be positive")
        batch_size = int(batch_size)
        if self.return_arrays is None:
            raise RemoraError("Must specify return arrays")
        batch = dict((arr_name, []) for arr_name in self.output_return_arrays)
        chunks_left_to_add = batch_size
        while (
            self._curr_sb_offset + chunks_left_to_add
            > self._curr_sb["signal"].shape[0]
        ):
            # add data from this super batch and load a new one
            update_batch(self._curr_sb_offset)
            self._load_next_super_batch()
            if self._curr_sb is None:
                return join_arrs(batch)
        if chunks_left_to_add > 0:
            b_st = self._curr_sb_offset
            b_en = self._curr_sb_offset + chunks_left_to_add
            update_batch(b_st, b_en)
            self._curr_sb_offset = b_en
        return join_arrs(batch)

    def iter_batches(self, batch_size=None, max_batches=None):
        """Iterate over batches."""
        batch_num = 0
        while True:
            batch = self.extract_batch(batch_size)
            if batch is None:
                break
            yield batch
            batch_num += 1
            if max_batches is not None and batch_num >= max_batches:
                break

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


def extract_core_dataset_paths(input_path, used_configs=None):
    """Extract the list of core dataset paths given a config or core dataset
    path."""
    paths = []
    input_path = util.resolve_path(input_path)
    if used_configs is None:
        used_configs = {input_path: input_path}
    if os.path.isdir(input_path):
        # return core dataset
        return [input_path]
    with open(input_path) as config_fh:
        for ds_info in json.load(config_fh):
            if len(ds_info) == 2:
                ds_path, _ = ds_info
            elif len(ds_info) == 3:
                ds_path, _, _ = ds_info
            ds_path = util.resolve_path(ds_path)
            if not os.path.exists(ds_path):
                raise RemoraError(
                    f"Core dataset path does not exist. {ds_path}"
                )
            if os.path.isdir(ds_path):
                paths.append(ds_path)
            else:
                if ds_path in used_configs:
                    raise RemoraError(
                        "Circular or repeated dataset config refrence. "
                        f"{ds_path} found in {input_path} and previously "
                        f"found in {used_configs[ds_path]}"
                    )
                used_configs[ds_path] = input_path
                sub_paths = extract_core_dataset_paths(
                    ds_path, used_configs=used_configs
                )
                paths.extend(sub_paths)
    paths = list(set(paths))
    return paths


def parse_dataset_config(input_path, used_configs=None):
    paths, weights, hashes = [], [], []
    input_path = util.resolve_path(input_path)
    if used_configs is None:
        used_configs = {input_path: input_path}
    with open(input_path) as config_fh:
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
                        f"{ds_path} found in {input_path} and previously "
                        f"found in {used_configs[ds_path]}"
                    )
                used_configs[ds_path] = input_path
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


def load_dataset(ds_path, core_ds_kwargs=None, ds_kwargs=None):
    """Parse either core dataset or dataset config"""
    ds_path = util.resolve_path(ds_path)
    if not os.path.exists(ds_path):
        raise RemoraError(f"Dataset path does not exist. {ds_path}")
    if os.path.isdir(ds_path):
        paths, props, hashes = [ds_path], np.ones(1, dtype=float), None
    else:
        paths, props, hashes = parse_dataset_config(ds_path)
    if core_ds_kwargs is None:
        core_ds_kwargs = {}
    if ds_kwargs is None:
        ds_kwargs = {}
    return RemoraDataset(
        [CoreRemoraDataset(path, **core_ds_kwargs) for path in paths],
        props,
        hashes,
        **ds_kwargs,
    )


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


def compute_random_split(total_size, probs):
    class_counts = np.random.multinomial(total_size, probs)

    # If the sum is not exactly total (which can happen due to rounding
    # issues), adjust the sum
    while np.sum(class_counts) != total_size:
        diff = total_size - np.sum(class_counts)
        for i in range(abs(diff)):
            if diff > 0:
                idx = np.random.choice(np.arange(len(probs)), p=probs)
                class_counts[idx] += 1
            elif diff < 0:
                idx = np.random.choice(
                    np.arange(len(probs))[class_counts > 0],
                    p=probs[class_counts > 0],
                )
                class_counts[idx] -= 1
    return class_counts


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

    def __init__(
        self,
        datasets,
        proportions,
        hashes=None,
        batch_size=constants.DEFAULT_BATCH_SIZE,
        super_batch_size=constants.DEFAULT_SUPER_BATCH_SIZE,
        super_batch_sample_frac=None,
        seed=None,
        use_constant_batch_mix=False,
        return_arrays=None,
    ):
        super(RemoraDataset).__init__()
        self.datasets = datasets
        self.props = proportions
        if not all(0 <= prop <= 1 for prop in self.props):
            raise RemoraError("Dataset proportions must be between 0 and 1.")
        if len(self.datasets) != len(self.props):
            raise RemoraError("Dataset and proportions must be same length.")
        self._hashes = hashes
        self.batch_size = batch_size
        self.super_batch_size = super_batch_size
        self.super_batch_sample_frac = super_batch_sample_frac
        self.seed = seed
        self.set_use_constant_batch_mix(use_constant_batch_mix)

        # RemoraDataset is infinite iter if all core datasets are infinite
        self.infinite_iter = all(ds.infinite_iter for ds in self.datasets)
        self.set_global_metadata()
        # apply applicable global metadata to sub-datasets
        for ds in self.datasets:
            ds.update_metadata(self)
        self.super_batch_offsets = [0 for ds in self.datasets]
        self._iter = None
        self._all_batches = None
        self.return_arrays = return_arrays
        if self.return_arrays is None:
            ds_return_arrays = set(ds.return_arrays for ds in self.datasets)
            if len(set(ds_return_arrays)) == 1:
                self.return_arrays = self.datasets[0].return_arrays
            else:
                raise RemoraError("Return arrays not set")
        else:
            for ds in self.datasets:
                ds.set_return_arrays(return_arrays)

    def set_use_constant_batch_mix(self, value):
        self.use_constant_batch_mix = value
        if self.use_constant_batch_mix:
            self._batch_sizes = compute_best_split(self.batch_size, self.props)

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
        label_summaries = "\n".join(ds.modbase_label_summary for ds in datasets)
        LOGGER.debug(f"Loaded dataset label summaries:\n{label_summaries}")
        return cls(datasets, props, hashes, **kwargs)

    @property
    def is_modbase_dataset(self):
        return self.metadata.dataset_type == constants.DATASET_TYPE_MODBASE

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
    def batches_preloaded(self):
        return self._all_batches is not None

    @property
    def valid_hashes(self):
        return self._hashes is not None and all(
            ds_hash is not None for ds_hash in self._hashes
        )

    @property
    def hashes(self):
        if not self.valid_hashes:
            LOGGER.debug("Computing dataset hashes")
            self._hashes = [ds.hash(ds.data_path) for ds in self.datasets]
        return self._hashes

    @property
    def summary(self):
        summ_txt = (
            f"                     size : {self.size:,}\n"
            "        is_modbase_dataset : "
            f"{self.metadata.is_modbase_dataset}\n"
            f"       kmer_context_bases : {self.metadata.kmer_context_bases}\n"
            f"            chunk_context : {self.metadata.chunk_context}\n"
            f"           reverse_signal : {self.metadata.reverse_signal}\n"
            f" chunk_extract_base_start : {self.metadata.base_start_justify}\n"
            f"     chunk_extract_offset : {self.metadata.offset}\n"
            f"               pa_scaling : {self.metadata.pa_scaling}\n"
            f"          sig_map_refiner : {self.metadata.sig_map_refiner}\n"
            f"        batches preloaded : {self.batches_preloaded}\n"
        )
        if self.is_modbase_dataset:
            summ_txt += (
                f"                mod_bases : {self.metadata.mod_bases}\n"
                f"           mod_long_names : {self.metadata.mod_long_names}\n"
                f"                   motifs : {self.metadata.motifs}\n"
            )
        return summ_txt

    @property
    def init_kwargs(self):
        return {
            "proportions": self.props,
            "hashes": self._hashes,
            "batch_size": self.batch_size,
            "super_batch_size": self.super_batch_size,
            "super_batch_sample_frac": self.super_batch_sample_frac,
            "seed": self.seed,
            "use_constant_batch_mix": self.use_constant_batch_mix,
            "return_arrays": self.return_arrays,
        }

    @property
    def output_return_arrays(self):
        """Output return arrays requested from this dataset. This includes
        converting sequence output encoding names.
        """
        out_r_arrs = []
        for arr_name in self.return_arrays:
            try:
                out_r_arrs.extend(constants.DATASET_SEQ_OUTPUTS[arr_name])
            except KeyError:
                out_r_arrs.append(arr_name)
        return out_r_arrs

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
        dataset_types = set(ds.metadata.dataset_type for ds in self.datasets)
        if len(dataset_types) > 1:
            raise RemoraError("Cannot process datasets of different types")
        self.metadata.dataset_type = self.datasets[0].metadata.dataset_type
        if self.metadata.is_modbase_dataset:
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
                "dataset_type",
                "base_start_justify",
                "offset",
                "reverse_signal",
                "pa_scaling",
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
            if ds.metadata.extra_array_names != self.metadata.extra_array_names:
                LOGGER.debug(
                    f"Extra arrays not equal: {ds.metadata.extra_array_names} "
                    f"!= {self.metadata.extra_array_names}; Using intersection."
                )
                self.metadata.extra_arrays_intersection_update(ds.metadata)
            if ds.metadata.is_modbase_dataset:
                for mb, mln in zip(
                    ds.metadata.mod_bases, ds.metadata.mod_long_names
                ):
                    if mb in self.metadata.mod_bases:
                        # ensure same mod short and long names
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

        if self.metadata.is_modbase_dataset:
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
            "dataset_type",
            "offset",
            "reverse_signal",
            "pa_scaling",
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
            "extra_signal_arrays",
            "extra_metadata_arrays",
            "extra_sequence_arrays",
            "kmer_context_bases",
            "chunk_context",
        ):
            setattr(self.metadata, md_key, getattr(other.metadata, md_key))

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
            trn_md["dataset_end"] = ds.metadata.dataset_end
            LOGGER.debug(f"train split override metadata: {trn_md}")
            train_datasets.append(
                CoreRemoraDataset(ds.data_path, override_metadata=trn_md)
            )
            test_md = override_metadata.copy()
            test_md["dataset_start"] = ds.metadata.dataset_start
            test_md["dataset_end"] = ds.metadata.dataset_start + test_size
            LOGGER.debug(f"test split override metadata: {test_md}")
            test_datasets.append(
                CoreRemoraDataset(
                    ds.data_path,
                    infinite_iter=False,
                    override_metadata=test_md,
                )
            )
        trn_ds = RemoraDataset(train_datasets, **self.init_kwargs)
        trn_ds.update_metadata(self)
        test_ds = RemoraDataset(test_datasets, **self.init_kwargs)
        test_ds.update_metadata(self)
        return trn_ds, test_ds

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
        head_ds = RemoraDataset(head_datasets, **self.init_kwargs)
        head_ds.update_metadata(self)
        return head_ds

    def _set_sub_ds_params(self):
        for ds, sb_offset in zip(self.datasets, self.super_batch_offsets):
            ds.super_batch_offset = sb_offset
            ds.super_batch_size = self.super_batch_size
            ds.super_batch_sample_frac = self.super_batch_sample_frac

    def iter_batches(self):
        """Iterate over batches."""
        self._set_sub_ds_params()
        while True:
            ds_batch_sizes = (
                self._batch_sizes
                if self.use_constant_batch_mix
                else compute_random_split(self.batch_size, self.props)
            )
            ds_arrays = [
                ds.extract_batch(bs)
                for ds, bs in zip(self.datasets, ds_batch_sizes)
                if bs > 0
            ]
            ds_arrays = [arr for arr in ds_arrays if arr is not None]
            if len(ds_arrays) == 0:
                break
            r_arrs = {}
            for arr_name in self.output_return_arrays:
                r_arrs[arr_name] = np.concatenate(
                    [arr[arr_name] for arr in ds_arrays],
                    axis=0,
                )
            if "seq" in r_arrs:
                # clip seq array to limit size for GPU transfer
                r_arrs["seq"] = r_arrs["seq"][:, : r_arrs["seq_len"].max()]
            yield [
                torch.from_numpy(r_arrs[arr_name])
                for arr_name in self.output_return_arrays
            ]

    def load_all_batches(self):
        if self.infinite_iter:
            raise RemoraError("Cannot load all batches for infinite dataset")
        self._set_sub_ds_params()
        self._all_batches = list(self.iter_batches())
        for ds in self.datasets:
            ds.close_memmaps()

    def __iter__(self):
        if self.batches_preloaded:
            self._iter = iter(self._all_batches)
            return self._iter
        # if first time calling iter or if this is an exhaustible dataset
        # re-initialize the iterator
        if self._iter is None or not self.infinite_iter:
            self._iter = self.iter_batches()
        return self._iter

    def __next__(self):
        return next(self._iter)

    def get_modbase_label_counts(self):
        label_counts = np.zeros(self.metadata.num_labels, dtype=int)
        if self.batches_preloaded:
            for _, b_labels, _ in self._all_batches:
                for idx, idx_cnt in enumerate(np.bincount(b_labels)):
                    label_counts[idx] += idx_cnt
            return label_counts
        for ds in self.datasets:
            for idx, count in enumerate(ds.get_modbase_label_counts()):
                label_counts[idx] += count
        return label_counts

    @property
    def modbase_label_summary(self):
        return "; ".join(
            f"{self.metadata.modbase_labels[lab_idx]}:{count:,}"
            for lab_idx, count in enumerate(self.get_modbase_label_counts())
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
        # TODO add filters to this summary
        if self.use_constant_batch_mix:
            epoch_chunk_totals = [
                batches_per_epoch * ds_bs for ds_bs in self._batch_sizes
            ]
        else:
            epoch_chunk_totals = [
                batches_per_epoch * self.batch_size * prop
                for prop in self.props
            ]
        if not self.is_modbase_dataset:
            summ_strs = [
                f"{ds_chunks_per_epoch/ds.size:10.4%}\t"
                f"{ds_chunks_per_epoch:,.1f}\t"
                f"{ds.size:,}\t"
                f"{ds.data_path}"
                for ds_chunks_per_epoch, ds in zip(
                    epoch_chunk_totals,
                    self.datasets,
                )
            ]
            return (
                "percent_of_dataset_per_epoch\tdataset_chunks_per_epoch\t"
                "dataset_size\tpath\n"
            ) + "\n".join(summ_strs)

        dss_lab_counts = [
            dict(
                zip(
                    ds.metadata.modbase_labels,
                    ds.get_modbase_label_counts(),
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
        # compute the number of chunks of each label extracted from each
        # dataset each batch
        batch_lab_cols = [
            "\t".join(
                f"{ds_lp.get(lab, 0) * ds_bs:,.1f}"
                for lab in self.metadata.modbase_labels
            )
            for ds_lp, ds_bs in zip(dss_lab_props, self.batch_size * self.props)
        ]
        dss_lab_cols = [
            "\t".join(
                f"{ds_lc.get(lab, 0):,}" for lab in self.metadata.modbase_labels
            )
            for ds_lc in dss_lab_counts
        ]
        summ_strs = [
            f"{ds_chunks_per_epoch/ds.size:10.4%}\t"
            f"{b_lab_cols}\t"
            f"{ds_chunks_per_epoch:,.1f}\t"
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
            (f"batch_{lab}" for lab in self.metadata.modbase_labels)
        )
        ds_labels_header = "\t".join(
            (f"dataset_{lab}" for lab in self.metadata.modbase_labels)
        )
        return (
            f"percent_of_dataset_per_epoch\t{b_labels_header}\t"
            f"dataset_chunks_per_epoch\tdataset_size\t{ds_labels_header}\t"
            "path\n"
        ) + "\n".join(summ_strs)


def load_remora_dataset_for_bonito(
    ds_path,
    n_pre_context_bases,
    n_post_context_bases,
    batch_size,
    super_batch_size=100_000,
    super_batch_sample_frac=None,
    chunks=None,
    valid_chunks=1_000,
    chunk_width=None,
    seed=None,
    prefetch_factor=10_000,
    **kwargs,
):
    override_metadata = {
        "kmer_context_bases": (n_pre_context_bases, n_post_context_bases)
    }
    if chunk_width is not None:
        override_metadata["chunk_context"] = (0, chunk_width)
    dataset = load_dataset(
        ds_path,
        core_ds_kwargs={"override_metadata": override_metadata},
        ds_kwargs={
            "batch_size": batch_size,
            "super_batch_size": super_batch_size,
            "super_batch_sample_frac": super_batch_sample_frac,
            "return_arrays": ["signal", "seq_and_len"],
            "seed": seed,
        },
    )
    trn_ds, val_ds = dataset.train_test_split(valid_chunks)
    val_ds.super_batch_sample_frac = None
    val_ds.do_check_super_batches = True
    val_ds.set_use_constant_batch_mix(True)
    val_ds.load_all_batches()

    train_loader_kwargs = {
        "dataset": trn_ds,
        "shuffle": False,
        "batch_size": None,
        "persistent_workers": True,
        "worker_init_fn": dataloader_worker_init,
        "prefetch_factor": prefetch_factor,
    }
    valid_loader_kwargs = {
        "dataset": val_ds,
        "shuffle": False,
        "batch_size": None,
        "num_workers": 0,
        "pin_memory": False,
    }
    return train_loader_kwargs, valid_loader_kwargs


class RemoraDatasetBonitoLoader:
    def __init__(self, config_path, **kwargs):
        tl_kwargs, vl_kwargs = load_remora_dataset_for_bonito(
            config_path, **kwargs
        )
        self.train_loader_kwargs = lambda **kwargs: tl_kwargs
        self.valid_loader_kwargs = lambda **kwargs: vl_kwargs
        self.worker_init_fn = dataloader_worker_init
