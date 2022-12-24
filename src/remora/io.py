import os
import random
from pathlib import Path
from typing import Callable
from itertools import chain
from copy import copy, deepcopy
from dataclasses import dataclass
from collections import defaultdict
from functools import cached_property

import pod5
import pysam
import numpy as np
from tqdm import tqdm
from pysam import AlignedSegment
from matplotlib import pyplot as plt

from remora.constants import PA_TO_NORM_SCALING_FACTOR
from remora import log, util, data_chunks as DC, duplex_utils as DU, RemoraError

LOGGER = log.get_logger()

_SIG_PROF_FN = os.getenv("REMORA_EXTRACT_SIGNAL_PROFILE_FILE")
_ALIGN_PROF_FN = os.getenv("REMORA_EXTRACT_ALIGN_PROFILE_FILE")

BASE_COLORS = {"A": "#00CC00", "C": "#0000CC", "G": "#FFB300", "T": "#CC0000"}


def parse_bed(bed_path):
    regs = defaultdict(set)
    with open(bed_path) as regs_fh:
        for line in regs_fh:
            fields = line.split()
            ctg, st, en = fields[:3]
            if len(fields) < 6 or fields[5] not in "+-":
                for strand in "+-":
                    regs[(ctg, strand)].update(range(int(st), int(en)))
            else:
                regs[(ctg, fields[5])].update(range(int(st), int(en)))
    return regs


def parse_mods_bed(bed_path):
    regs = defaultdict(dict)
    all_mods = set()
    with open(bed_path) as regs_fh:
        for line in regs_fh:
            fields = line.split()
            ctg, st, en, mod = fields[:4]
            all_mods.update(mod)
            if len(fields) < 6 or fields[5] not in "+-":
                for strand in "+-":
                    for pos in range(int(st), int(en)):
                        regs[(ctg, strand)][pos] = mod
            else:
                for pos in range(int(st), int(en)):
                    regs[(ctg, fields[5])][pos] = mod
    return regs, all_mods


def read_is_primary(read):
    """Helper function to return whether a read is a primary mapping
    (not supplementary or secondary)

    Args:
        read: pysam.AlignedSegment
    """
    return not (read.is_supplementary or read.is_secondary)


def strands_match(strand, bam_read):
    return (
        strand not in "+-"
        or (strand == "+" and bam_read.is_forward)
        or (strand == "-" and bam_read.is_reverse)
    )


@dataclass
class ReadIndexedBam:
    """Index bam file by read id. Note that the BAM file handle is closed after
    initialization. Any other operation (e.g. fetch, get_alignments,
    get_first_alignment) will open the pysam file handle and leave it open.
    This allows easier use with multiprocessing using standard operations.

    Args:
        bam_path (str): Path to BAM file
        skip_non_primary (bool): Should non-primary alignmets be skipped
        req_tags (bool): Skip reads without required tags
        read_id_converter (Callable[[str], str]): Function to convert read ids
            (e.g. for concatenated duplex read ids)
    """

    bam_path: str
    skip_non_primary: bool = True
    req_tags: set = None
    read_id_converter: Callable = None

    def __post_init__(self):
        self.num_reads = None
        self.bam_fh = None
        self._bam_idx = None
        self.compute_read_index()

    def open(self):
        # hide warnings for no index when using unmapped or unsorted files
        self.pysam_save = pysam.set_verbosity(0)
        self.bam_fh = pysam.AlignmentFile(
            self.bam_path, mode="rb", check_sq=False
        )

    def close(self):
        self.bam_fh.close()
        self.bam_fh = None
        pysam.set_verbosity(self.pysam_save)

    def fetch(self, ref_reg):
        if self.bam_fh is None:
            self.open()
        for read in self.bam_fh.fetch(ref_reg.ctg, ref_reg.start, ref_reg.end):
            if strands_match(ref_reg.strand, read):
                yield read

    def compute_read_index(self):
        bam_opened = self.bam_fh is not None
        if not bam_opened:
            self.open()
        self._bam_idx = defaultdict(list)
        pbar = tqdm(smoothing=0, unit=" Reads", desc="Indexing BAM by read id")
        self.num_records = 0
        # iterating over file handle gives incorrect pointers
        while True:
            read_ptr = self.bam_fh.tell()
            try:
                read = next(self.bam_fh)
            except StopIteration:
                break
            pbar.update()
            if self.req_tags is not None:
                tags = set(tg[0] for tg in read.tags)
                if not self.req_tags.issubset(tags):
                    LOGGER.debug(
                        f"{read.query_name} missing tags "
                        f"{self.req_tags.difference(tags)}"
                    )
                    continue
            index_read_id = (
                read.query_name
                if self.read_id_converter is None
                else self.read_id_converter(read.query_name)
            )
            if self.skip_non_primary and (
                not read_is_primary(read) or index_read_id in self._bam_idx
            ):
                LOGGER.debug(f"{read.query_name} not primary")
                continue
            self.num_records += 1
            self._bam_idx[index_read_id].append(read_ptr)
        # close bam if it was closed at start of function call
        if bam_opened:
            self.close()
        pbar.close()
        # convert defaultdict to dict
        self._bam_idx = dict(self._bam_idx)
        self.num_reads = len(self._bam_idx)

    def get_alignments(self, read_id):
        if self._bam_idx is None:
            raise RemoraError("Bam index not yet computed")
        if self.bam_fh is None:
            self.open()
        try:
            read_ptrs = self._bam_idx[read_id]
        except KeyError:
            raise RemoraError(f"Could not find {read_id} in {self.bam_path}")
        for read_ptr in read_ptrs:
            self.bam_fh.seek(read_ptr)
            bam_read = next(self.bam_fh)
            assert bam_read.query_name == read_id, (
                f"Read ID {read_id} does not match extracted BAM record "
                f"{bam_read.query_name} at {read_ptr}. ReadIndexedBam may "
                "be corrupted."
            )
            yield bam_read

    def get_first_alignment(self, read_id):
        return next(self.get_alignments(read_id))

    def __contains__(self, read_id):
        assert isinstance(read_id, str)
        return read_id in self._bam_idx

    def __getitem__(self, read_id):
        return self._bam_idx[read_id]

    def __del__(self):
        if self.bam_fh is not None:
            self.bam_fh.close()

    @cached_property
    def read_ids(self):
        return list(self._bam_idx.keys())


def parse_move_tag(mv_tag, sig_len, seq_len=None, check=True):
    stride = mv_tag[0]
    mv_table = np.array(mv_tag[1:])
    query_to_signal = np.nonzero(mv_table)[0] * stride
    query_to_signal = np.concatenate([query_to_signal, [sig_len]])
    if check and seq_len is not None and query_to_signal.size - 1 != seq_len:
        LOGGER.debug(
            f"Move table (num moves: {query_to_signal.size - 1}) discordant "
            f"with basecalls (seq len: {seq_len})"
        )
        raise RemoraError("Move table discordant with basecalls")
    if check and mv_table.size != sig_len // stride:
        LOGGER.debug(
            f"Move table (len: {mv_table.size}) discordant with "
            f"signal (sig len // stride: {sig_len // stride})"
        )
        raise RemoraError("Move table discordant with signal")
    return query_to_signal, mv_table, stride


def get_ref_seq_and_levels_from_reads(ref_reg, bam_reads, sig_map_refiner):
    # TODO handle missing sig_to_seq_refiner and only return seq
    if ref_reg.strand == "+":
        bb = sig_map_refiner.bases_before
        context_st = ref_reg.start - sig_map_refiner.bases_before
        context_en = ref_reg.end + sig_map_refiner.bases_after
    else:
        bb = sig_map_refiner.bases_after
        context_st = ref_reg.start - sig_map_refiner.bases_after
        context_en = ref_reg.end + sig_map_refiner.bases_before
    context_len = context_en - context_st
    # record forward reference sequence. Will be flipped after for reverse
    # strand reference regions
    context_int_seq = np.full(context_len, -1, np.int32)
    for bam_read in bam_reads:
        read_ref_seq = bam_read.get_reference_sequence().upper()
        context_int_seq[
            max(0, bam_read.reference_start - context_st) : (
                bam_read.reference_end - context_st
            )
        ] = util.seq_to_int(
            read_ref_seq[
                max(0, context_st - bam_read.reference_start) : (
                    context_en - bam_read.reference_start
                )
            ]
        )
        if not np.any(context_int_seq == -1):
            break
    if ref_reg.strand == "+":
        levels = sig_map_refiner.extract_levels(context_int_seq)[
            bb : bb + ref_reg.len
        ]
    else:
        levels = sig_map_refiner.extract_levels(
            util.revcomp_np(context_int_seq)
        )[::-1][bb : bb + ref_reg.len]
    seq = util.int_to_seq(context_int_seq[bb : bb + ref_reg.len])
    return seq, levels


def extract_ref_region_reads(
    bam_fhs,
    pod5_fhs,
    ref_pos,
    sig_map_refiner=None,
    skip_sig_map_refine=False,
    max_reads=50,
):
    # TODO handle both strand plots
    reg_bam_reads = [
        [
            bam_read
            for bam_read in bam_fh.fetch(
                ref_pos.ctg, ref_pos.start, ref_pos.end
            )
            if read_is_primary(bam_read)
            and strands_match(ref_pos.strand, bam_read)
        ]
        for bam_fh in bam_fhs
    ]
    seq, levels = get_ref_seq_and_levels_from_reads(
        ref_pos, chain(*reg_bam_reads), sig_map_refiner
    )
    if max_reads is not None:
        reg_bam_reads = [
            random.sample(samp_reads, max_reads)
            if len(samp_reads) > max_reads
            else samp_reads
            for samp_reads in reg_bam_reads
        ]

    ref_reg_reads = []
    for samp_reads, pod5_fh in zip(reg_bam_reads, pod5_fhs):
        ref_reg_reads.append([])
        for bam_read in samp_reads:
            pod5_read = next(
                pod5_fh.reads(
                    selection=[bam_read.query_name], preload=["samples"]
                )
            )
            io_read = Read.from_pod5_and_alignment(
                pod5_read_record=pod5_read,
                alignment_record=bam_read,
            )
            if sig_map_refiner is not None and not skip_sig_map_refine:
                io_read.set_refine_signal_mapping(
                    sig_map_refiner, ref_mapping=True
                )
            ref_reg_reads[-1].append(
                io_read.extract_reference_region(ref_pos.start, ref_pos.end)
            )
    return ref_reg_reads, seq, levels


def plot_ref_region_reads(
    ref_pos,
    ref_reg_reads,
    seq,
    levels,
    max_reads=50,
    ax=None,
    figsize=(40, 10),
    sig_lw=2,
    sig_cols=["k", "r", "c"],
    levels_lw=6,
    ylim=None,
):
    # start plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if ylim is None:
        sig_min, sig_max = np.percentile(
            np.concatenate([r.norm_signal for r in chain(*ref_reg_reads)]),
            (0, 100),
        )
        sig_diff = sig_max - sig_min
        ylim = (sig_min - sig_diff * 0.01, sig_max + sig_diff * 0.01)
    base_text_loc = ylim[0] + ((ylim[1] - ylim[0]) * 0.02)

    # plot vertical base lines
    for b in np.arange(ref_pos.start, ref_pos.end + 1):
        ax.axvline(x=b, color="k", alpha=0.1, lw=1)
    for samp_col, samp_reads in zip(sig_cols, ref_reg_reads):
        for read_reg in samp_reads:
            # plot read signal
            ax.plot(
                read_reg.sig_x_coords,
                read_reg.norm_signal,
                color=samp_col,
                alpha=0.1,
                lw=sig_lw,
            )
    # plot levels
    if levels is not None:
        for b_num, b_lev in enumerate(levels):
            ax.plot(
                [ref_pos.start + b_num, ref_pos.start + b_num + 1],
                [b_lev] * 2,
                linestyle="-",
                color="y",
                alpha=0.5,
                lw=levels_lw,
            )
    # plot bases
    plot_seq = seq if ref_pos.strand == "+" else util.comp(seq)
    rotation = 0 if ref_pos.strand == "+" else 180
    for b_num, base in enumerate(plot_seq):
        ax.text(
            ref_pos.start + b_num + 0.5,
            base_text_loc,
            base,
            color=BASE_COLORS[base],
            ha="center",
            size=30,
            rotation=rotation,
        )
    ax.set_ylim(*ylim)
    ax.set_ylabel("Normalized Signal", fontsize=45)
    ax.set_xlabel("Reference Position", fontsize=45)
    # TODO shift tick labels left by 0.5 plot units to match genome browsers
    ax.tick_params(labelsize=36)
    return ax


def plot_signal_at_ref_region(
    bam_fhs,
    pod5_fhs,
    ref_pos,
    sig_map_refiner=None,
    skip_sig_map_refine=False,
    max_reads=50,
    ax=None,
    figsize=(40, 10),
    sig_lw=2,
    levels_lw=6,
    ylim=None,
):
    """Plot signal from reads at a reference region.

    Args:
        bam_fhs (pysam.AlignmentFile): Sorted and indexed BAM file handle or
            a list of them
        pod5_fhs (str): POD5 file handles or a list of them
        ref_pos (RefPos): Reference position at which to plot signal
        sig_map_refiner (SigMapRefiner): For signal mapping and level extract
        skip_sig_map_refine (bool): Skip signal mapping refinement
        max_reads (int): Maximum reads to plot (TODO: add overplotting options)
        ax (matplotlib.axes): If None, new figure will be opened
        figsize (tuple): option to pass to plt.subplots if ax is None
        sig_lw (int): Linewidth for signal lines
        levels_lw (int): Linewidth for level lines (if applicable)
        ylim (tuple): 2-tuple with y-axis limits

    Returns:
        matplotlib axis
    """
    if isinstance(bam_fhs, pysam.AlignmentFile):
        bam_fhs = [bam_fhs]
    if isinstance(bam_fhs, pod5.reader.Reader):
        pod5_fhs = [pod5_fhs]
    ref_reg_reads, seq, levels = extract_ref_region_reads(
        bam_fhs,
        pod5_fhs,
        ref_pos,
        sig_map_refiner=sig_map_refiner,
        skip_sig_map_refine=skip_sig_map_refine,
        max_reads=max_reads,
    )
    ax = plot_ref_region_reads(
        ref_pos,
        ref_reg_reads,
        seq,
        levels,
        ax=ax,
        figsize=figsize,
        sig_lw=sig_lw,
        levels_lw=levels_lw,
        ylim=ylim,
    )
    return ax


@dataclass
class RefPos:
    ctg: str
    strand: str
    start: int
    end: int = None

    @property
    def len(self):
        if self.end is None:
            return 1
        return self.end - self.start


@dataclass
class ReadReferenceRegion:
    read_id: str
    norm_signal: np.ndarray
    seq: str
    seq_to_sig_map: np.ndarray
    ref_pos: RefPos

    @property
    def sig_x_coords(self):
        """Compute x-axis coorindates for plotting this read against the
        reference.
        """
        if self.ref_pos.strand == "+":
            return (
                np.interp(
                    np.arange(self.norm_signal.size),
                    self.seq_to_sig_map,
                    np.arange(self.seq_to_sig_map.size),
                )
                + self.ref_pos.start
            )
        return (
            self.ref_pos.start
            + self.seq_to_sig_map.size
            - 1
            - np.interp(
                np.arange(self.norm_signal.size),
                self.seq_to_sig_map,
                np.arange(self.seq_to_sig_map.size),
            )
        )


@dataclass
class ReadBasecallRegion:
    read_id: str
    norm_signal: np.ndarray
    seq: str
    seq_to_sig_map: np.ndarray
    start: int

    @property
    def base_x_coords(self):
        return (
            np.interp(
                np.arange(self.norm_signal.size),
                self.seq_to_sig_map,
                np.arange(self.seq_to_sig_map.size),
            )
            + self.start
        )

    def plot_on_signal_coords(
        self,
        levels=None,
        ax=None,
        figsize=(40, 10),
        sig_lw=8,
        levels_lw=8,
        ylim=None,
    ):
        """Plot signal from reads at a reference region.

        Args:
            levels (np.array): Expected signal levels for bases in region.
            ax (matplotlib.axes): If None, new figure will be opened
            figsize (tuple): option to pass to plt.subplots if ax is None
            sig_lw (int): Linewidth for signal lines
            levels_lw (int): Linewidth for level lines (if applicable)
            ylim (tuple): 2-tuple with y-axis limits

        Returns:
            matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        if ylim is None:
            sig_min, sig_max = np.percentile(self.norm_signal, (0, 100))
            sig_diff = sig_max - sig_min
            ylim = (sig_min - sig_diff * 0.01, sig_max + sig_diff * 0.01)
        base_text_loc = ylim[0] + ((ylim[1] - ylim[0]) * 0.02)

        # plot vertical base lines
        for b in self.seq_to_sig_map:
            ax.axvline(x=b, color="k", alpha=0.1, lw=1)
        # plot read signal
        ax.plot(
            np.arange(self.seq_to_sig_map[0], self.seq_to_sig_map[-1]),
            self.norm_signal,
            color="k",
            alpha=0.5,
            lw=sig_lw,
        )
        # plot levels
        if levels is not None:
            for b_lev, b_st, b_en in zip(
                levels, self.seq_to_sig_map[:-1], self.seq_to_sig_map[1:]
            ):
                ax.plot(
                    [b_st, b_en],
                    [b_lev] * 2,
                    linestyle="-",
                    color="y",
                    alpha=0.5,
                    lw=levels_lw,
                )
        # plot bases
        for b_st, b_en, base in zip(
            self.seq_to_sig_map[:-1], self.seq_to_sig_map[1:], self.seq
        ):
            ax.text(
                (b_en + b_st) / 2,
                base_text_loc,
                base,
                color=BASE_COLORS[base],
                ha="center",
                size=30,
            )
        ax.set_ylim(*ylim)
        ax.set_ylabel("Normalized Signal", fontsize=45)
        ax.set_xlabel("Signal Position", fontsize=45)
        ax.tick_params(labelsize=36)
        return ax

    def plot_on_base_coords(
        self,
        levels=None,
        ax=None,
        figsize=(40, 10),
        sig_lw=8,
        levels_lw=8,
        ylim=None,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        if ylim is None:
            sig_min, sig_max = np.percentile(self.norm_signal, (0, 100))
            sig_diff = sig_max - sig_min
            ylim = (sig_min - sig_diff * 0.01, sig_max + sig_diff * 0.01)
        base_text_loc = ylim[0] + ((ylim[1] - ylim[0]) * 0.02)

        # plot vertical base lines
        for b in np.arange(self.start, self.start + self.seq_to_sig_map.size):
            ax.axvline(x=b, color="k", alpha=0.1, lw=1)
        # plot read signal
        ax.plot(
            self.base_x_coords,
            self.norm_signal,
            color="k",
            alpha=0.5,
            lw=sig_lw,
        )
        # plot levels
        if levels is not None:
            for b_num, b_lev in enumerate(levels):
                ax.plot(
                    [self.start + b_num, self.start + b_num + 1],
                    [b_lev] * 2,
                    linestyle="-",
                    color="y",
                    alpha=0.5,
                    lw=levels_lw,
                )
        # plot bases
        for b_num, base in enumerate(self.seq):
            ax.text(
                self.start + b_num + 0.5,
                base_text_loc,
                base,
                color=BASE_COLORS[base],
                ha="center",
                size=30,
            )
        ax.set_ylim(*ylim)
        ax.set_ylabel("Normalized Signal", fontsize=45)
        ax.set_xlabel("Base Position", fontsize=45)
        ax.tick_params(labelsize=36)
        return ax


@dataclass
class Read:
    read_id: str
    dacs: np.ndarray = None
    seq: str = None
    stride: int = None
    num_trimmed: int = None
    mv_table: np.array = None
    query_to_signal: np.ndarray = None
    shift_dacs_to_pa: float = None
    scale_dacs_to_pa: float = None
    shift_pa_to_norm: float = None
    scale_pa_to_norm: float = None
    shift_dacs_to_norm: float = None
    scale_dacs_to_norm: float = None
    ref_seq: str = None
    ref_pos: RefPos = None
    cigar: list = None
    ref_to_signal: np.ndarray = None
    full_align: str = None

    @property
    def pa_signal(self):
        """Picoampere convereted signal. Shift and scale values are determined
        from the instrument. These values are generally not recommended for any
        type of analysis.
        """
        assert self.scale_dacs_to_pa is not None
        assert self.shift_dacs_to_pa is not None
        return self.scale_dacs_to_pa * (self.dacs + self.shift_dacs_to_pa)

    @property
    def norm_signal(self):
        """Normalized signal, generally aiming to standardize the signal with
        mean=0 and SD=1, though different methods may be applied to perform
        this operation robustly.
        """
        assert self.scale_dacs_to_norm is not None
        assert self.shift_dacs_to_norm is not None
        return (self.dacs - self.shift_dacs_to_norm) / self.scale_dacs_to_norm

    def compute_pa_to_norm_scaling(self, factor=PA_TO_NORM_SCALING_FACTOR):
        self.shift_pa_to_norm = np.median(self.pa_signal)
        self.scale_pa_to_norm = max(
            1.0,
            np.median(np.abs(self.pa_signal - self.shift_pa_to_norm)) * factor,
        )

    @property
    def seq_len(self):
        if self.query_to_signal is None:
            return None
        return self.query_to_signal.size - 1

    @property
    def ref_seq_len(self):
        if self.ref_to_signal is None:
            return None
        return self.ref_to_signal.size - 1

    def with_duplex_alignment(self, duplex_read_alignment, duplex_orientation):
        """Return copy with alignment to duplex sequence

        Args:
            duplex_read_alignment (AlignedSegment): pysam alignment record
                containing duplex sequence
            duplex_orientation (bool): Is duplex mapping in the same
                orientation as this simplex read.
        """
        assert self.query_to_signal is not None, "requires query_to_signal"
        assert (
            duplex_read_alignment.query_sequence is not None
        ), "no duplex base call sequence?"
        assert (
            len(duplex_read_alignment.query_sequence) > 0
        ), "duplex base call sequence is empty string?"

        read = copy(self)

        duplex_read_sequence = (
            duplex_read_alignment.query_sequence
            if duplex_orientation
            else util.revcomp(duplex_read_alignment.query_sequence)
        )

        # we don't have a mapping of each base in the simplex sequence to each
        # base in the duplex sequence, so we infer it by alignment. Using the
        # simplex sequence as the query sequence is somewhat arbitrary, but it
        # makes the downstream coordinate mappings more convenient
        simplex_duplex_mapping = DU.map_simplex_to_duplex(
            simplex_seq=read.seq, duplex_seq=duplex_read_sequence
        )
        # duplex_read_to_signal is a mapping of each position in the duplex
        # sequence to a signal datum from the read
        query_to_signal = read.query_to_signal
        duplex_to_read_signal = DC.map_ref_to_signal(
            query_to_signal=query_to_signal,
            ref_to_query_knots=simplex_duplex_mapping.duplex_to_simplex_mapping,
        )
        read.seq = simplex_duplex_mapping.trimmed_duplex_seq
        read.query_to_signal = duplex_to_read_signal

        read.ref_seq = None
        read.ref_to_signal = None
        read.ref_pos = None
        return read, simplex_duplex_mapping.duplex_offset

    def add_alignment(self, alignment_record, parse_ref_align=True):
        """Add alignment to read object

        Args:
            alignment_record (pysam.AlignedSegment)
            parse_ref_align (bool): Should reference alignment be parsed
        """
        if (
            alignment_record.reference_name is None
            and alignment_record.is_reverse
        ):
            raise RemoraError("Unmapped reads cannot map to reverse strand.")
        if self.dacs is None:
            raise RemoraError("Must add signal to io.Read before alignment.")
        self.full_align = alignment_record.to_dict()

        tags = dict(alignment_record.tags)
        try:
            self.num_trimmed = tags["ts"]
            self.dacs = self.dacs[self.num_trimmed :]
        except KeyError:
            self.num_trimmed = 0

        try:
            self.query_to_signal, self.mv_table, self.stride = parse_move_tag(
                tags["mv"],
                sig_len=self.dacs.size,
                seq_len=len(alignment_record.query_sequence),
            )
        except KeyError:
            self.query_to_signal = self.mv_table = self.stride = None

        try:
            self.shift_pa_to_norm = tags["sm"]
            self.scale_pa_to_norm = tags["sd"]
        except KeyError:
            self.compute_pa_to_norm_scaling()

        self.shift_dacs_to_norm = (
            self.shift_pa_to_norm / self.scale_dacs_to_pa
        ) - self.shift_dacs_to_pa
        self.scale_dacs_to_norm = self.scale_pa_to_norm / self.scale_dacs_to_pa

        self.seq = alignment_record.query_sequence
        if alignment_record.is_reverse:
            self.seq = util.revcomp(self.seq)
        if not parse_ref_align:
            return

        self.ref_pos = RefPos(
            ctg=alignment_record.reference_name,
            strand="-" if alignment_record.is_reverse else "+",
            start=alignment_record.reference_start,
        )
        try:
            self.ref_seq = alignment_record.get_reference_sequence().upper()
        except ValueError:
            self.ref_seq = None
        self.cigar = alignment_record.cigartuples
        if alignment_record.is_reverse:
            self.ref_seq = util.revcomp(self.ref_seq)
            self.cigar = self.cigar[::-1]
        if self.ref_pos.ctg is not None:
            self.ref_to_signal = DC.compute_ref_to_signal(
                query_to_signal=self.query_to_signal,
                cigar=self.cigar,
                query_seq=self.seq,
                ref_seq=self.ref_seq,
            )
            assert self.ref_to_signal.size == len(self.ref_seq) + 1
            self.ref_pos.end = self.ref_pos.start + self.ref_to_signal.size - 1

    @classmethod
    def from_pod5_and_alignment(cls, pod5_read_record, alignment_record):
        """Initialize read from pod5 and pysam records

        Args:
            pod5_read_record (pod5.ReadRecord)
            alignment_record (pysam.AlignedSegment)
        """
        read = Read(
            read_id=str(pod5_read_record.read_id),
            dacs=pod5_read_record.signal,
            shift_dacs_to_pa=pod5_read_record.calibration.offset,
            scale_dacs_to_pa=pod5_read_record.calibration.scale,
        )
        read.add_alignment(alignment_record)
        return read

    def into_remora_read(self, use_reference_anchor):
        """Extract RemoraRead object from io.Read.

        Args:
            use_reference_anchor (bool): Should remora read be reference
                anchored? Or basecall anchored?
        """
        if use_reference_anchor:
            if self.ref_to_signal is None:
                if self.cigar is None or self.ref_seq is None:
                    raise RemoraError("missing reference alignment")
                self.ref_to_signal = DC.compute_ref_to_signal(
                    self.query_to_signal,
                    self.cigar,
                    query_seq=self.seq,
                    ref_seq=self.ref_seq,
                )

            trim_dacs = self.dacs[
                self.ref_to_signal[0] : self.ref_to_signal[-1]
            ]
            shift_seq_to_sig = self.ref_to_signal - self.ref_to_signal[0]
            seq = self.ref_seq
        else:
            trim_dacs = self.dacs[
                self.query_to_signal[0] : self.query_to_signal[-1]
            ]
            shift_seq_to_sig = self.query_to_signal - self.query_to_signal[0]
            seq = self.seq
        remora_read = DC.RemoraRead(
            dacs=trim_dacs,
            shift=self.shift_dacs_to_norm,
            scale=self.scale_dacs_to_norm,
            seq_to_sig_map=shift_seq_to_sig,
            str_seq=seq,
            read_id=self.read_id,
        )
        remora_read.check()
        return remora_read

    def set_refine_signal_mapping(self, sig_map_refiner, ref_mapping=False):
        """Refine signal mapping

        Args:
            sig_map_refiner (refine_signal_map.SigMapRefiner): Signal mapping
                refiner
            ref_mapping (bool): Refine reference mapping? Default: False/refine
                basecall to signal mapping

        Note that shift/scale parameters are overwritten for the reference or
        basecall mapping. To keep both use io.Read.copy().
        """
        if sig_map_refiner is None:
            return
        remora_read = self.into_remora_read(ref_mapping)
        remora_read.refine_signal_mapping(sig_map_refiner)
        if ref_mapping:
            self.ref_to_signal = (
                remora_read.seq_to_sig_map + self.ref_to_signal[0]
            )
        else:
            self.query_to_signal = (
                remora_read.seq_to_sig_map + self.query_to_signal[0]
            )

        self.shift_dacs_to_norm = remora_read.shift
        self.scale_dacs_to_norm = remora_read.scale
        self.shift_pa_to_norm = (
            self.shift_dacs_to_norm + self.shift_dacs_to_pa
        ) * self.scale_dacs_to_pa
        self.scale_pa_to_norm = self.scale_dacs_to_norm * self.scale_dacs_to_pa

    def get_filtered_focus_positions(self, select_focus_positions):
        """
        Args:
            select_focus_positions (dict): lookup table of (contig, strand)
            tuples (both strings) to a set of positions to include.

        Returns:
            np.ndarray of positions covered by the read and within the
            selected focus position
        """
        if self.ref_pos is None or self.ref_seq is None:
            raise RemoraError("Cannot extract focus positions without mapping")
        ref_pos = self.ref_pos
        ref_len = len(self.ref_seq)
        try:
            cs_focus_pos = select_focus_positions[(ref_pos.ctg, ref_pos.strand)]
        except KeyError:
            # no focus positions on contig/strand
            return np.array([], dtype=int)

        read_focus_ref_pos = np.array(
            sorted(
                set(range(ref_pos.start, ref_pos.start + ref_len)).intersection(
                    cs_focus_pos
                )
            ),
            dtype=int,
        )
        return (
            read_focus_ref_pos - ref_pos.start
            if ref_pos.strand == "+"
            else ref_pos.start + ref_len - read_focus_ref_pos[::-1] - 1
        )

    def get_basecall_anchored_focus_bases(
        self, motifs, select_focus_reference_positions
    ):
        """Get basecall anchored focus bases

        Args:
            motifs (list of util.Motif): List of motifs
            select_focus_reference_positions (dict): Reference focus positions
                as returned from io.parse_bed
        """
        if self.cigar is None:
            raise RemoraError("missing alignment")

        basecall_int_seq = util.seq_to_int(self.seq)
        reference_int_seq = util.seq_to_int(self.ref_seq)

        all_base_call_focus_positions = util.find_focus_bases_in_int_sequence(
            int_seq=basecall_int_seq, motifs=motifs
        )
        # mapping of reference sequence positions to base call sequence
        # positions
        mapping = DC.make_sequence_coordinate_mapping(
            cigar=self.cigar, read_seq=self.seq, ref_seq=self.ref_seq
        )

        reference_motif_positions = (
            util.find_focus_bases_in_int_sequence(reference_int_seq, motifs)
            if select_focus_reference_positions is None
            else self.get_filtered_focus_positions(
                select_focus_positions=select_focus_reference_positions,
            )
        )
        reference_supported_focus_bases = mapping[reference_motif_positions]
        base_call_focus_bases = np.array(
            [
                focus_base
                for focus_base in all_base_call_focus_positions
                if focus_base in reference_supported_focus_bases
            ]
        )
        return base_call_focus_bases

    def copy(self):
        return deepcopy(self)

    def extract_basecall_region(self, start_base=None, end_base=None):
        """Extract region of read from basecall coordinates.

        Args:
            start_base (int): Start coordinate for region
            end_base (int): End coordinate for region

        Returns:
            ReadBasecallRegion object
        """
        start_base = start_base or 0
        end_base = end_base or self.seq_len
        reg_seq_to_sig = self.query_to_signal[start_base : end_base + 1].copy()
        reg_sig = self.norm_signal[reg_seq_to_sig[0] : reg_seq_to_sig[-1]]
        reg_seq_to_sig -= reg_seq_to_sig[0]
        return ReadBasecallRegion(
            read_id=self.read_id,
            norm_signal=reg_sig,
            seq=self.seq[start_base:end_base],
            seq_to_sig_map=reg_seq_to_sig,
            start=start_base,
        )

    def extract_reference_region(self, ref_start, ref_end):
        """Extract region of read from reference coordinates.

        Args:
            ref_start (int): Reference start coordinate for region
            ref_end (int): Reference end coordinate for region

        Returns:
            ReadReferenceRegion object
        """
        if self.ref_to_signal is None:
            raise RemoraError(
                "Cannot extract reference region from unmapped read"
            )
        if ref_start >= self.ref_pos.start + self.ref_seq_len:
            raise RemoraError("Reference region starts after read ends")
        if ref_end < self.ref_pos.start:
            raise RemoraError("Reference region ends before read starts")

        if self.ref_pos.strand == "+":
            reg_st_within_read = max(0, ref_start - self.ref_pos.start)
            reg_en_within_read = ref_end - self.ref_pos.start
        else:
            reg_st_within_read = max(0, self.ref_pos.end - ref_end)
            reg_en_within_read = self.ref_pos.end - ref_start
        reg_seq_to_sig = self.ref_to_signal[
            reg_st_within_read : reg_en_within_read + 1
        ].copy()
        reg_sig = self.norm_signal[reg_seq_to_sig[0] : reg_seq_to_sig[-1]]
        reg_seq = self.ref_seq[reg_st_within_read:reg_en_within_read]
        reg_seq_to_sig -= reg_seq_to_sig[0]
        read_reg_ref_st = max(self.ref_pos.start, ref_start)
        return ReadReferenceRegion(
            read_id=self.read_id,
            norm_signal=reg_sig,
            seq=reg_seq,
            seq_to_sig_map=reg_seq_to_sig,
            ref_pos=RefPos(
                self.ref_pos.ctg, self.ref_pos.strand, read_reg_ref_st
            ),
        )


@dataclass
class DuplexRead:
    duplex_read_id: str
    duplex_alignment: dict
    is_reverse_mapped: bool
    template_read: Read
    complement_read: Read
    template_ref_start: int
    complement_ref_start: int

    @classmethod
    def from_reads_and_alignment(
        cls,
        *,
        template_read: Read,
        complement_read: Read,
        duplex_alignment: AlignedSegment,
    ):
        is_reverse_mapped = duplex_alignment.is_reverse
        duplex_direction_read, reverse_complement_read = (
            (template_read, complement_read)
            if not is_reverse_mapped
            else (complement_read, template_read)
        )

        (
            template_read,
            template_ref_start,
        ) = duplex_direction_read.with_duplex_alignment(
            duplex_alignment,
            duplex_orientation=True,
        )
        (
            complement_read,
            complement_ref_start,
        ) = reverse_complement_read.with_duplex_alignment(
            duplex_alignment,
            duplex_orientation=False,
        )

        return DuplexRead(
            duplex_read_id=duplex_alignment.query_name,
            duplex_alignment=duplex_alignment.to_dict(),
            is_reverse_mapped=is_reverse_mapped,
            template_read=template_read,
            complement_read=complement_read,
            template_ref_start=template_ref_start,
            complement_ref_start=complement_ref_start,
        )

    @property
    def duplex_basecalled_sequence(self):
        # n.b. pysam reverse-complements the query sequence on reverse mappings
        # [https://pysam.readthedocs.io/en/latest/api.html#pysam.AlignedSegment.query_sequence]
        return self.duplex_alignment["seq"]


##########################
# Signal then alignments #
##########################


def iter_pod5_reads(pod5_path, num_reads=None, read_ids=None):
    """Iterate over Pod5Read objects

    Args:
        pod5_path (str): Path to POD5 file
        num_reads (int): Maximum number of reads to iterate
        read_ids (iterable): Read IDs to extract

    Yields:
        Requested pod5.reader.ReadRecord objects
    """
    LOGGER.debug(f"Reading from POD5 at {pod5_path}")
    with pod5.Reader(Path(pod5_path)) as pod5_fh:
        for read_num, read in enumerate(
            pod5_fh.reads(selection=read_ids, preload=["samples"])
        ):
            if num_reads is not None and read_num >= num_reads:
                LOGGER.debug(
                    f"Completed pod5 signal worker, reached {num_reads}."
                )
                return

            yield read
    LOGGER.debug("Completed pod5 signal worker")


def iter_signal(pod5_path, num_reads=None, read_ids=None):
    """Iterate io Read objects loaded from Pod5

    Args:
        pod5_path (str): Path to POD5 file
        num_reads (int): Maximum number of reads to iterate
        read_ids (iterable): Read IDs to extract

    Yields:
        2-tuple:
            1. remora.io.Read object
            2. Error text or None if no errors
    """
    for pod5_read in iter_pod5_reads(
        pod5_path=pod5_path, num_reads=num_reads, read_ids=read_ids
    ):
        read = Read(
            read_id=str(pod5_read.read_id),
            dacs=pod5_read.signal,
            shift_dacs_to_pa=pod5_read.calibration.offset,
            scale_dacs_to_pa=pod5_read.calibration.scale,
        )

        yield read, None
    LOGGER.debug("Completed signal worker")


class DuplexPairsBuilder:
    def __init__(self, simplex_index, pod5_path):
        """Duplex Pairs Builder

        Args:
            simplex_index (ReadIndexedBam): Simplex bam index
            pod5_path (str): Path to POD5 file
        """
        self.simplex_index = simplex_index
        self.pod5_path = pod5_path
        self.reader = pod5.Reader(Path(pod5_path))

    def make_read_pair(self, read_id_pair):
        """Make read pair

        Args:
            read_id_pair (tuple): 2-tuple of read ID strings
        """
        try:
            pod5_reads = list(
                self.reader.reads(
                    selection=list(read_id_pair), preload=["samples"]
                )
            )
        except RuntimeError:
            return None, "duplex pair read id(s) missing from pod5"

        if len(pod5_reads) < 2:
            return None, "duplex pair read id(s) missing from pod5"
        if len(pod5_reads) > 2:
            return None, "pod5 has multiple reads with the same id"
        pod5_reads = {str(read.read_id): read for read in pod5_reads}

        temp_read_id, comp_read_id = read_id_pair
        try:
            temp_align = self.simplex_index.get_first_alignment(temp_read_id)
            comp_align = self.simplex_index.get_first_alignment(comp_read_id)
        except RemoraError:
            return None, "failed to find read in simplex bam"
        temp_io_read = Read.from_pod5_and_alignment(
            pod5_read_record=pod5_reads[temp_read_id],
            alignment_record=temp_align,
        )
        comp_io_read = Read.from_pod5_and_alignment(
            pod5_read_record=pod5_reads[comp_read_id],
            alignment_record=comp_align,
        )

        return (temp_io_read, comp_io_read), None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reader.close()
        self.simplex_index.close()


if _SIG_PROF_FN:
    _iter_signal_wrapper = iter_signal

    def iter_signal(*args, **kwargs):
        import cProfile

        sig_prof = cProfile.Profile()
        retval = sig_prof.runcall(_iter_signal_wrapper, *args, **kwargs)
        sig_prof.dump_stats(_SIG_PROF_FN)
        return retval


def extract_alignments(read_err, bam_idx):
    io_read, err = read_err
    if io_read is None:
        return [read_err]
    read_alignments = []
    try:
        for bam_read in bam_idx.get_alignments(io_read.read_id):
            align_read = io_read.copy()
            align_read.add_alignment(bam_read)
            read_alignments.append(tuple((align_read, None)))
    except KeyError:
        return [tuple((None, "Read id not found in BAM file"))]
    return read_alignments


if _ALIGN_PROF_FN:
    _extract_align_wrapper = extract_alignments

    def extract_alignments(*args, **kwargs):
        import cProfile

        align_prof = cProfile.Profile()
        retval = align_prof.runcall(_extract_align_wrapper, *args, **kwargs)
        align_prof.dump_stats(_ALIGN_PROF_FN)
        return retval
