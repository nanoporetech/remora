import os
import re
import random
from pathlib import Path
from typing import Callable
from copy import copy, deepcopy
from dataclasses import dataclass
from collections import defaultdict
from itertools import chain, product
from functools import cached_property

import pod5
import pysam
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pysam import AlignedSegment
from matplotlib import pyplot as plt

from remora.metrics import METRIC_FUNCS
from remora.constants import PA_TO_NORM_SCALING_FACTOR, DEFAULT_PLOT_FIG_SIZE
from remora import log, util, data_chunks as DC, duplex_utils as DU, RemoraError

LOGGER = log.get_logger()

_SIG_PROF_FN = os.getenv("REMORA_EXTRACT_SIGNAL_PROFILE_FILE")
_ALIGN_PROF_FN = os.getenv("REMORA_EXTRACT_ALIGN_PROFILE_FILE")

SIG_PCTL_RANGE = (0.2, 99.8)
SAMPLE_COLORS = ["k", "tab:red", "tab:cyan", "tab:pink", "tab:orange"]
BASE_COLORS = {
    "A": "#00CC00",
    "C": "#0000CC",
    "G": "#FFB300",
    "T": "#CC0000",
    "U": "#CC0000",
    "N": "#FFFFFF",
}


##############
# General IO #
##############


@dataclass
class RefRegion:
    ctg: str
    strand: str
    start: int
    end: int = None

    @property
    def len(self):
        if self.end is None:
            return 1
        return self.end - self.start

    @classmethod
    def parse_ref_region_str(cls, ref_reg_str, req_strand=True):
        mat = re.match(
            r"^(?P<ctg>.+):(?P<st>\d+)-(?P<en>\d+):(?P<strand>[\+\-])$"
            if req_strand
            else r"^(?P<ctg>.+):(?P<st>\d+)-(?P<en>\d+)(:(?P<strand>[\+\-]))?$",
            ref_reg_str,
        )
        if mat is None:
            raise RemoraError(f"Invalid reference region: {ref_reg_str}")
        start = int(mat.group("st")) - 1
        if start < 0:
            raise RemoraError("Invalid reference start coordinate")
        return cls(
            ctg=mat.group("ctg"),
            strand=mat.group("strand"),
            start=start,
            end=int(mat.group("en")),
        )

    @property
    def coord_range(self):
        return range(self.start, self.end)

    def adjust(self, start_adjust=0, end_adjust=0, ref_orient=True):
        """Return a copy of this region adjusted by the specified amounts.

        Args:
            start_adjust (int): Adjustment to start
            end_adjust (int): Adjustment to end
            ref_orient (bool): If True, expand start and end directly. If False
                (read oriented) and region is reverse strand, swap start and
                end adjustments. For read_orient, the region will be expanded
                relative to the reads mapping to the appropriate strand.
        """
        if ref_orient or self.strand == "+":
            end_coord = None if self.end is None else self.end + end_adjust
            return RefRegion(
                self.ctg, self.strand, self.start + start_adjust, end_coord
            )
        else:
            end_coord = None if self.end is None else self.end - start_adjust
            return RefRegion(
                self.ctg, self.strand, self.start - end_adjust, end_coord
            )


def parse_bed_lines(bed_path):
    with open(bed_path) as regs_fh:
        for line in regs_fh:
            fields = line.split()
            ctg, st, en = fields[:3]
            st = int(st)
            en = int(en)
            strand = (
                None if len(fields) < 6 or fields[5] not in "+-" else fields[5]
            )
            yield RefRegion(ctg, strand, st, en)


def parse_bed(bed_path):
    regs = defaultdict(set)
    for ref_reg in parse_bed_lines(bed_path):
        if ref_reg.strand is None:
            for strand in "+-":
                regs[(ref_reg.ctg, strand)].update(ref_reg.coord_range)
        else:
            regs[(ref_reg.ctg, ref_reg.strand)].update(ref_reg.coord_range)
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
        bam_was_closed = self.bam_fh is None
        if bam_was_closed:
            self.open()
        self._bam_idx = defaultdict(list)
        pbar = tqdm(
            smoothing=0,
            unit=" Reads",
            desc="Indexing BAM by read id",
            disable=os.environ.get("LOG_SAFE", False),
        )
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
        if bam_was_closed:
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
            try:
                bam_read = next(self.bam_fh)
            except OSError:
                LOGGER.debug(
                    f"Could not extract {read_id} from {self.bam_path} "
                    f"at {read_ptr}"
                )
                raise RemoraError(
                    "Could not extract BAM read. Ensure BAM file object was "
                    "closed before spawning process."
                )
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


def parse_move_tag(
    mv_tag, sig_len, seq_len=None, check=True, reverse_signal=False
):
    stride = mv_tag[0]
    mv_table = np.array(mv_tag[1:])
    query_to_signal = np.nonzero(mv_table)[0] * stride
    query_to_signal = np.concatenate([query_to_signal, [sig_len]])
    if reverse_signal:
        query_to_signal = sig_len - query_to_signal[::-1]
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


#######################
# POD5/BAM Extraction #
#######################


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


def iter_signal(pod5_path, num_reads=None, read_ids=None, rev_sig=False):
    """Iterate io Read objects loaded from Pod5

    Args:
        pod5_path (str): Path to POD5 file
        num_reads (int): Maximum number of reads to iterate
        read_ids (iterable): Read IDs to extract
        rev_sig (bool): Should signal be reversed on reading

    Yields:
        2-tuple:
            1. remora.io.Read object
            2. Error text or None if no errors
    """
    for pod5_read in iter_pod5_reads(
        pod5_path=pod5_path, num_reads=num_reads, read_ids=read_ids
    ):
        dacs = pod5_read.signal[::-1] if rev_sig else pod5_read.signal
        read = Read(
            read_id=str(pod5_read.read_id),
            dacs=dacs,
            shift_dacs_to_pa=pod5_read.calibration.offset,
            scale_dacs_to_pa=pod5_read.calibration.scale,
        )

        yield read, None
    LOGGER.debug("Completed signal worker")


if _SIG_PROF_FN:
    _iter_signal_wrapper = iter_signal

    def iter_signal(*args, **kwargs):
        import cProfile

        sig_prof = cProfile.Profile()
        retval = sig_prof.runcall(_iter_signal_wrapper, *args, **kwargs)
        sig_prof.dump_stats(_SIG_PROF_FN)
        return retval


def extract_alignments(read_err, bam_idx, rev_sig=False):
    io_read, err = read_err
    if io_read is None:
        return [read_err]
    read_alignments = []
    try:
        for bam_read in bam_idx.get_alignments(io_read.read_id):
            align_read = io_read.copy()
            align_read.add_alignment(bam_read, reverse_signal=rev_sig)
            read_alignments.append(tuple((align_read, None)))
    except KeyError:
        return [tuple((None, "Read id not found in BAM file"))]
    except RemoraError as e:
        return [tuple((None, str(e)))]
    return read_alignments


if _ALIGN_PROF_FN:
    _extract_align_wrapper = extract_alignments

    def extract_alignments(*args, **kwargs):
        import cProfile

        align_prof = cProfile.Profile()
        retval = align_prof.runcall(_extract_align_wrapper, *args, **kwargs)
        align_prof.dump_stats(_ALIGN_PROF_FN)
        return retval


def iter_regions(bam_fh, reg_len=100_000):
    for ctg, ctg_len in zip(bam_fh.header.references, bam_fh.header.lengths):
        for st in range((ctg_len // reg_len) + 1):
            yield RefRegion(
                ctg=ctg,
                strand="+",
                start=st * reg_len,
                end=(st + 1) * reg_len,
            )
            yield RefRegion(
                ctg=ctg,
                strand="-",
                start=st * reg_len,
                end=(st + 1) * reg_len,
            )


def get_reg_bam_reads(ref_reg, bam_fh):
    return [
        bam_read
        for bam_read in bam_fh.fetch(ref_reg.ctg, ref_reg.start, ref_reg.end)
        if read_is_primary(bam_read) and strands_match(ref_reg.strand, bam_read)
    ]


def iter_covered_regions(
    bam_path, chunk_len=1_000, max_chunk_cov=None, pickle_safe=False
):
    with pysam.AlignmentFile(bam_path) as bam_fh:
        for reg in iter_regions(bam_fh, chunk_len):
            bam_reads = get_reg_bam_reads(reg, bam_fh)
            if len(bam_reads) == 0:
                continue
            if max_chunk_cov is not None:
                target_bases = chunk_len * max_chunk_cov
                total_bases = 0
                random.shuffle(bam_reads)
                sampled_bam_reads = []
                for bam_read in bam_reads:
                    sampled_bam_reads.append(bam_read)
                    total_bases += min(bam_read.reference_end, reg.end) - max(
                        bam_read.reference_start, reg.start
                    )
                    if total_bases >= target_bases:
                        break
                bam_reads = sampled_bam_reads
            if pickle_safe:
                yield reg, [br.to_dict() for br in bam_reads]
                continue
            yield reg, bam_reads


def compute_base_space_sig_coords(seq_to_sig_map):
    """Compute coordinates for signal points, interpolating signal assigned to
    each base linearly through the span of each covered base.
    """
    return np.interp(
        np.arange(seq_to_sig_map[-1] - seq_to_sig_map[0]),
        seq_to_sig_map,
        np.arange(seq_to_sig_map.size),
    )


@dataclass
class ReadRefReg:
    read_id: str
    norm_signal: np.ndarray
    seq: str
    seq_to_sig_map: np.ndarray
    ref_reg: RefRegion

    @property
    def ref_sig_coords(self):
        """Compute signal coorindates for plotting this read against the
        mapped stretch of reference.
        """
        return (
            compute_base_space_sig_coords(self.seq_to_sig_map)
            + self.ref_reg.start
        )

    def plot_on_signal_coords(self, **kwargs):
        """Plot signal on signal coordinates. See global plot_on_signal_coords
        function for kwargs.
        """
        return plot_on_signal_coords(
            self.seq,
            self.norm_signal,
            self.seq_to_sig_map,
            rev_strand=self.ref_reg.strand == "-",
            **kwargs,
        )

    def plot_on_base_coords(self, **kwargs):
        """Plot signal on base/sequence coordinates. See global
        plot_on_base_coords function for kwargs.
        """
        return plot_on_base_coords(
            self.ref_reg.start,
            self.seq,
            self.norm_signal,
            self.seq_to_sig_map,
            rev_strand=self.ref_reg.strand == "-",
            **kwargs,
        )


@dataclass
class ReadBasecallRegion:
    read_id: str
    norm_signal: np.ndarray
    seq: str
    seq_to_sig_map: np.ndarray
    start: int

    def plot_on_signal_coords(self, **kwargs):
        """Plot signal on signal coordinates. See global plot_on_signal_coords
        function for kwargs.
        """
        return plot_on_signal_coords(
            self.seq, self.norm_signal, self.seq_to_sig_map, **kwargs
        )

    def plot_on_base_coords(self, **kwargs):
        """Plot signal on base/sequence coordinates. See global
        plot_on_base_coords function for kwargs.
        """
        return plot_on_base_coords(
            self.start,
            self.seq,
            self.norm_signal,
            self.seq_to_sig_map,
            **kwargs,
        )


def get_ref_int_seq_from_reads(ref_reg, bam_reads, ref_orient=True):
    # fill with -2 since N is represented by -1
    int_seq = np.full(ref_reg.len, -2, np.int32)
    # extract forward reference sequence.
    for bam_read in bam_reads:
        read_ref_seq = bam_read.get_reference_sequence().upper()
        int_seq[
            max(0, bam_read.reference_start - ref_reg.start) : (
                bam_read.reference_end - ref_reg.start
            )
        ] = util.seq_to_int(
            read_ref_seq[
                max(0, ref_reg.start - bam_read.reference_start) : (
                    ref_reg.end - bam_read.reference_start
                )
            ]
        )
        if not np.any(int_seq == -2):
            break
    if ref_reg.strand == "-":
        if ref_orient:
            return util.comp_np(int_seq)
        else:
            return util.revcomp_np(int_seq)
    return int_seq


def get_ref_seq_from_reads(ref_reg, bam_reads, ref_orient=True):
    int_seq = get_ref_int_seq_from_reads(
        ref_reg, bam_reads, ref_orient=ref_orient
    )
    int_seq[np.equal(int_seq, -2)] = -1
    return util.int_to_seq(int_seq)


def get_ref_seq_and_levels_from_reads(
    ref_reg,
    bam_reads,
    sig_map_refiner,
    ref_orient=True,
):
    """Extract sequence and levels from BAM reads covering a region.

    Args:
        ref_reg (RefRegion): Reference region
        bam_reads (iterable): pysam.AlignedSegments covering ref_reg
        sig_map_refiner (SigMapRefiner): For level extraction
        ref_orient (bool): Should returned sequence and levels be reference
            oriented? This only effects the output for reverse strand ref_reg.
            Reference orientation will return the reference sequence and levels
            in the forward reference direction. Read orient (ref_orient=False)
            will return sequence and levels in 5' to 3' read direction on the
            reverse strand.
    """
    # extract read oriented context seq
    context_int_seq = get_ref_int_seq_from_reads(
        ref_reg.adjust(
            -sig_map_refiner.bases_before,
            sig_map_refiner.bases_after,
            ref_orient=False,
        ),
        bam_reads,
        ref_orient=False,
    )
    # levels are read oriented
    levels = sig_map_refiner.extract_levels(context_int_seq)
    # convert sequence positions not in reads to nan and seq to N
    levels[np.equal(context_int_seq, -2)] = np.NAN
    context_int_seq[np.equal(context_int_seq, -2)] = -1
    seq = util.int_to_seq(context_int_seq)
    # trim seq and levels
    seq = seq[
        sig_map_refiner.bases_before : sig_map_refiner.bases_before
        + ref_reg.len
    ]
    levels = levels[
        sig_map_refiner.bases_before : sig_map_refiner.bases_before
        + ref_reg.len
    ]

    if ref_reg.strand == "-" and ref_orient:
        seq = seq[::-1]
        levels = levels[::-1]
    return seq, levels


def get_pod5_reads(pod5_fh, read_ids):
    return dict(
        (str(pod5_read.read_id), pod5_read)
        for pod5_read in pod5_fh.reads(selection=read_ids, preload=["samples"])
    )


def get_io_reads(bam_reads, pod5_fh, reverse_signal=False, missing_ok=False):
    pod5_reads = get_pod5_reads(
        pod5_fh, [bam_read.query_name for bam_read in bam_reads]
    )
    io_reads = []
    for bam_read in bam_reads:
        try:
            io_read = Read.from_pod5_and_alignment(
                pod5_read_record=pod5_reads[bam_read.query_name],
                alignment_record=bam_read,
                reverse_signal=reverse_signal,
            )
        except Exception:
            if missing_ok:
                continue
            else:
                raise RemoraError("BAM record not found in POD5")
        io_reads.append(io_read)
    return io_reads


def get_reads_reference_regions(
    ref_reg,
    pod5_and_bam_fhs,
    sig_map_refiner=None,
    skip_sig_map_refine=False,
    max_reads=50,
    reverse_signal=False,
):
    all_bam_reads = []
    samples_read_ref_regs = []
    for pod5_fh, bam_fh in pod5_and_bam_fhs:
        sample_bam_reads = get_reg_bam_reads(ref_reg, bam_fh)
        if len(sample_bam_reads) == 0:
            raise RemoraError("No reads covering region")
        if max_reads is not None and len(sample_bam_reads) > max_reads:
            sample_bam_reads = random.sample(sample_bam_reads, max_reads)
        all_bam_reads.append(sample_bam_reads)
        io_reads = get_io_reads(sample_bam_reads, pod5_fh, reverse_signal)
        if sig_map_refiner is not None and not skip_sig_map_refine:
            for io_read in io_reads:
                io_read.set_refine_signal_mapping(
                    sig_map_refiner, ref_mapping=True
                )
        samples_read_ref_regs.append(
            [io_read.extract_ref_reg(ref_reg) for io_read in io_reads]
        )
    return samples_read_ref_regs, all_bam_reads


def get_ref_reg_sample_metrics(
    ref_reg,
    pod5_fh,
    bam_reads,
    metric,
    sig_map_refiner,
    skip_sig_map_refine=False,
    reverse_signal=False,
    ref_orient=True,
    **kwargs,
):
    io_reads = get_io_reads(bam_reads, pod5_fh, reverse_signal)
    if sig_map_refiner is not None and not skip_sig_map_refine:
        for io_read in io_reads:
            io_read.set_refine_signal_mapping(sig_map_refiner, ref_mapping=True)
    sample_metrics = [
        io_read.compute_per_base_metric(metric, region=ref_reg, **kwargs)
        for io_read in io_reads
    ]
    if len(sample_metrics) > 0:
        reg_metrics = dict(
            (
                metric_name,
                np.stack([mv[metric_name] for mv in sample_metrics]),
            )
            for metric_name in sample_metrics[0].keys()
        )
        # ref_anchored read metrics are read oriented. Thus if ref_orient=True,
        # need to flip metrics
        if ref_orient and ref_reg.strand == "-":
            return dict(
                (metric_name, vals[:, ::-1])
                for metric_name, vals in reg_metrics.items()
            )
        return reg_metrics


def get_ref_reg_samples_metrics(
    ref_reg,
    pod5_and_bam_fhs,
    sig_map_refiner=None,
    skip_sig_map_refine=False,
    max_reads=None,
    reverse_signal=False,
    metric="dwell_trimmean",
    **kwargs,
):
    all_bam_reads = []
    samples_metrics = []
    for pod5_fh, bam_fh in pod5_and_bam_fhs:
        sample_bam_reads = get_reg_bam_reads(ref_reg, bam_fh)
        if len(sample_bam_reads) == 0:
            raise RemoraError("No reads covering region")
        if max_reads is not None and len(sample_bam_reads) > max_reads:
            sample_bam_reads = random.sample(sample_bam_reads, max_reads)
        all_bam_reads.append(sample_bam_reads)
        sample_metrics = get_ref_reg_sample_metrics(
            ref_reg,
            pod5_fh,
            sample_bam_reads,
            metric,
            sig_map_refiner,
            skip_sig_map_refine,
            reverse_signal,
            **kwargs,
        )
        if sample_metrics is not None:
            samples_metrics.append(sample_metrics)

    return samples_metrics, all_bam_reads


###################
# K-mer functions #
###################


def get_region_kmers(
    reg_and_bam_reads,
    pod5_fh,
    sig_map_refiner,
    kmer_context_bases,
    min_cov=10,
    start_trim=2,
    end_trim=2,
    dict_bam_reads=False,
    bam_header=None,
    reverse_signal=False,
):
    reg, bam_reads = reg_and_bam_reads
    if dict_bam_reads:
        bam_reads = [
            pysam.AlignedSegment.from_dict(br, bam_header) for br in bam_reads
        ]
    reg_metrics = get_ref_reg_sample_metrics(
        reg,
        pod5_fh,
        bam_reads,
        "dwell_trimmean",
        sig_map_refiner,
        start_trim=start_trim,
        end_trim=end_trim,
        ref_orient=False,
        reverse_signal=reverse_signal,
    )
    seq = get_ref_seq_from_reads(
        reg.adjust(
            -kmer_context_bases[0], kmer_context_bases[1], ref_orient=False
        ),
        bam_reads,
        ref_orient=False,
    )
    kmer_len = sum(kmer_context_bases) + 1
    reg_kmer_levels = dict(
        ("".join(bs), []) for bs in product("ACGT", repeat=kmer_len)
    )
    for offset in range(reg.len):
        kmer = seq[offset : offset + kmer_len]
        try:
            offset_kmer_levels = reg_kmer_levels[kmer]
        except KeyError:
            continue
        site_read_levels = reg_metrics["trimmean"][:, offset]
        site_read_levels = site_read_levels[np.isfinite(site_read_levels)]
        if site_read_levels.size < min_cov:
            continue
        offset_kmer_levels.append(np.median(site_read_levels))
    return dict(
        (kmer, np.array(levels)) for kmer, levels in reg_kmer_levels.items()
    )


def prep_region_kmers(*args, **kwargs):
    args = list(args)
    args[0] = pod5.Reader(args[0])
    return tuple(args), kwargs


def get_site_kmer_levels(
    pod5_path,
    bam_path,
    sig_map_refiner,
    kmer_context_bases,
    min_cov=10,
    chunk_len=1_000,
    max_chunk_cov=100,
    start_trim=1,
    end_trim=1,
    num_workers=1,
    reverse_signal=False,
):
    regs_bam_reads = util.BackgroundIter(
        iter_covered_regions,
        args=(bam_path, chunk_len, max_chunk_cov),
        kwargs={"pickle_safe": True},
    )
    with pysam.AlignmentFile(bam_path) as bam_fh:
        bam_header = bam_fh.header
    regs_kmer_levels = util.MultitaskMap(
        get_region_kmers,
        regs_bam_reads,
        prep_func=prep_region_kmers,
        num_workers=num_workers,
        use_process=True,
        args=(
            pod5_path,
            sig_map_refiner,
            kmer_context_bases,
        ),
        kwargs={
            "min_cov": min_cov,
            "start_trim": start_trim,
            "end_trim": start_trim,
            "dict_bam_reads": True,
            "bam_header": bam_header,
            "reverse_signal": reverse_signal,
        },
        name="GetKmers",
    )

    # enumerate kmers for dict
    kmer_len = sum(kmer_context_bases) + 1
    all_kmer_levels = dict(
        ("".join(bs), []) for bs in product("ACGT", repeat=kmer_len)
    )
    for reg_kmer_levels in regs_kmer_levels:
        for kmer, levels in reg_kmer_levels.items():
            all_kmer_levels[kmer].append(levels)
    return dict(
        (kmer, np.concatenate(levels) if len(levels) > 0 else np.array([]))
        for kmer, levels in all_kmer_levels.items()
    )


######################
# Plotting functions #
######################


def plot_on_signal_coords(
    seq,
    sig,
    seq_to_sig_map,
    rev_strand=False,
    levels=None,
    fig_ax=None,
    figsize=DEFAULT_PLOT_FIG_SIZE,
    sig_lw=8,
    levels_lw=8,
    ylim=None,
    t_as_u=False,
):
    """Plot a single read on signal coordinates.

    Args:
        seq (str): Sequence to plot
        sig (np.array): Signal to plot
        seq_to_sig_map (np.array): Mapping from sequence to signal coordinates
        rev_strand (bool): Plot seq with 180 rotation
        levels (np.array): Expected signal levels for bases in region
        fig_ax (tuple): If None, new figure/axes will be opened
        figsize (tuple): option to pass to plt.subplots if ax is None
        sig_lw (int): Linewidth for signal lines
        levels_lw (int): Linewidth for level lines (if applicable)
        ylim (tuple): 2-tuple with y-axis limits
        t_as_u (bool): Plot T bases as U (RNA)

    Returns:
        matplotlib axis
    """
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_ax
    if ylim is None:
        sig_min, sig_max = np.percentile(sig, SIG_PCTL_RANGE)
        sig_diff = sig_max - sig_min
        ylim = (sig_min - sig_diff * 0.01, sig_max + sig_diff * 0.01)
    base_text_loc = ylim[0] + ((ylim[1] - ylim[0]) * 0.02)

    # plot vertical base lines
    for b in seq_to_sig_map:
        ax.axvline(x=b, color="k", alpha=0.1, lw=1)
    # plot read signal
    ax.plot(
        np.arange(seq_to_sig_map[0], seq_to_sig_map[-1]),
        sig,
        color="k",
        alpha=0.5,
        lw=sig_lw,
    )
    # plot levels
    if levels is not None:
        for b_lev, b_st, b_en in zip(
            levels, seq_to_sig_map[:-1], seq_to_sig_map[1:]
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
    if t_as_u:
        seq = util.t_to_u(seq)
    for b_st, b_en, base in zip(seq_to_sig_map[:-1], seq_to_sig_map[1:], seq):
        ax.text(
            (b_en + b_st) / 2,
            base_text_loc,
            base,
            color=BASE_COLORS[base],
            ha="center",
            size=30,
            rotation=180 if rev_strand else 0,
        )
    ax.set_ylim(*ylim)
    ax.set_ylabel("Normalized Signal", fontsize=45)
    ax.set_xlabel("Signal Position", fontsize=45)
    ax.tick_params(labelsize=36)
    return fig, ax


def plot_on_base_coords(
    start_base,
    seq,
    sig,
    seq_to_sig_map,
    levels=None,
    rev_strand=False,
    fig_ax=None,
    figsize=DEFAULT_PLOT_FIG_SIZE,
    sig_lw=8,
    levels_lw=8,
    ylim=None,
    t_as_u=False,
):
    """Plot a single read on base/sequence coordinates.

    Args:
        start_base (int): Coordinate of first base in seq
        seq (str): Sequence to plot
        sig (np.array): Signal to plot
        seq_to_sig_map (np.array): Mapping from sequence to signal coordinates
        levels (np.array): Expected signal levels for bases in region
        rev_strand (bool): Plot bases upside down
        fig_ax (tuple): If None, new figure/axes will be opened
        figsize (tuple): option to pass to plt.subplots if ax is None
        sig_lw (int): Linewidth for signal lines
        levels_lw (int): Linewidth for level lines (if applicable)
        ylim (tuple): 2-tuple with y-axis limits
        t_as_u (bool): Plot T bases as U (RNA)

    Returns:
        matplotlib axis
    """
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_ax
    if ylim is None:
        sig_min, sig_max = np.percentile(sig, SIG_PCTL_RANGE)
        sig_diff = sig_max - sig_min
        ylim = (sig_min - sig_diff * 0.01, sig_max + sig_diff * 0.01)
    base_text_loc = ylim[0] + ((ylim[1] - ylim[0]) * 0.02)

    # plot vertical base lines
    for b in np.arange(start_base, start_base + seq_to_sig_map.size):
        ax.axvline(x=b, color="k", alpha=0.1, lw=1)
    # plot read signal
    ax.plot(
        compute_base_space_sig_coords(seq_to_sig_map) + start_base,
        sig,
        color="k",
        alpha=0.5,
        lw=sig_lw,
    )
    # plot levels
    if levels is not None:
        for b_num, b_lev in enumerate(levels):
            ax.plot(
                [start_base + b_num, start_base + b_num + 1],
                [b_lev] * 2,
                linestyle="-",
                color="y",
                alpha=0.5,
                lw=levels_lw,
            )
    # plot bases
    if t_as_u:
        seq = util.t_to_u(seq)
    for b_num, base in enumerate(seq):
        ax.text(
            start_base + b_num + 0.5,
            base_text_loc,
            base,
            color=BASE_COLORS[base],
            ha="center",
            size=30,
            rotation=180 if rev_strand else 0,
        )
    ax.set_ylim(*ylim)
    ax.set_ylabel("Normalized Signal", fontsize=45)
    ax.set_xlabel("Base Position", fontsize=45)
    ax.tick_params(labelsize=36)
    return fig, ax


def plot_ref_region_reads(
    ref_reg,
    ref_reg_reads,
    seq,
    levels,
    max_reads=50,
    fig_ax=None,
    figsize=DEFAULT_PLOT_FIG_SIZE,
    sig_lw=2,
    sig_cols=SAMPLE_COLORS,
    levels_lw=6,
    ylim=None,
    highlight_ranges=None,
    t_as_u=False,
):
    # start plotting
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_ax

    # plot highlight regions
    if highlight_ranges is not None:
        for st, en, col in highlight_ranges:
            ax.axvspan(st, en, facecolor=col, alpha=0.4)
    # plot vertical base lines
    for b in np.arange(ref_reg.start, ref_reg.end + 1):
        ax.axvline(x=b, color="k", alpha=0.1, lw=1)
    # plot read signal
    for sample_col, sample_reads in zip(sig_cols, ref_reg_reads):
        for read_reg in sample_reads:
            ax.plot(
                read_reg.ref_sig_coords,
                read_reg.norm_signal,
                color=sample_col,
                alpha=0.1,
                lw=sig_lw,
            )
    # plot levels
    if levels is not None:
        for b_num, b_lev in enumerate(levels):
            ax.plot(
                [ref_reg.start + b_num, ref_reg.start + b_num + 1],
                [b_lev] * 2,
                linestyle="-",
                color="y",
                alpha=0.5,
                lw=levels_lw,
            )
    # plot bases
    if ylim is None:
        sig_min, sig_max = np.percentile(
            np.concatenate([r.norm_signal for r in chain(*ref_reg_reads)]),
            SIG_PCTL_RANGE,
        )
        sig_diff = sig_max - sig_min
        ylim = (sig_min - sig_diff * 0.01, sig_max + sig_diff * 0.01)
    base_text_loc = ylim[0] + ((ylim[1] - ylim[0]) * 0.02)
    rotation = 0 if ref_reg.strand == "+" else 180
    if t_as_u:
        seq = util.t_to_u(seq)
    for b_num, base in enumerate(seq):
        ax.text(
            ref_reg.start + b_num + 0.5,
            base_text_loc,
            base,
            color=BASE_COLORS[base],
            ha="center",
            size=30,
            rotation=rotation,
        )
    ax.set_ylim(*ylim)
    ax.set_xlim(ref_reg.start, ref_reg.end)
    ax.set_title(ref_reg.ctg, fontsize=50)
    ax.set_ylabel("Normalized Signal", fontsize=45)
    ax.set_xlabel("Reference Position", fontsize=45)
    ax.tick_params(labelsize=36)
    # shift tick labels left by 0.5 plot units to match genome browsers
    xticks = np.array(
        [int(xt) for xt in ax.get_xticks() if ref_reg.start < xt < ref_reg.end]
    )
    ax.set_xticks(xticks - 0.5)
    ax.set_xticklabels(xticks)
    return fig, ax


def plot_signal_at_ref_region(
    ref_reg,
    pod5_and_bam_fhs,
    sig_map_refiner=None,
    skip_sig_map_refine=False,
    max_reads=50,
    reverse_signal=False,
    t_as_u=False,
    fig_ax=None,
    figsize=DEFAULT_PLOT_FIG_SIZE,
    sig_lw=2,
    levels_lw=6,
    ylim=None,
    highlight_ranges=None,
):
    """Plot signal from reads at a reference region.

    Args:
        bam_fhs (pysam.AlignmentFile): Sorted and indexed BAM file handle or
            a list of them
        pod5_fhs (str): POD5 file handles or a list of them
        ref_reg (RefRegion): Reference position at which to plot signal
        sig_map_refiner (SigMapRefiner): For signal mapping and level extract
        skip_sig_map_refine (bool): Skip signal mapping refinement
        max_reads (int): Maximum reads to plot (TODO: add overplotting options)
        reverse_signal (bool): Is nanopore signal 3'>5' orientation?
        t_as_u (bool): Plot T bases as U (RNA)
        fig_ax (tuple): If None, new figure/axes will be opened
        figsize (tuple): option to pass to plt.subplots if ax is None
        sig_lw (int): Linewidth for signal lines
        levels_lw (int): Linewidth for level lines (if applicable)
        ylim (tuple): 2-tuple with y-axis limits
        highlight_ranges (iterable): Elements should be 3-tuples containing
            reference start, end and color values.

    Returns:
        matplotlib axis
    """
    read_ref_regs, reg_bam_reads = get_reads_reference_regions(
        ref_reg,
        pod5_and_bam_fhs,
        sig_map_refiner=sig_map_refiner,
        skip_sig_map_refine=skip_sig_map_refine,
        max_reads=max_reads,
        reverse_signal=reverse_signal,
    )
    seq, levels = get_ref_seq_and_levels_from_reads(
        ref_reg, chain(*reg_bam_reads), sig_map_refiner
    )
    fig_ax = plot_ref_region_reads(
        ref_reg,
        read_ref_regs,
        seq,
        levels,
        fig_ax=fig_ax,
        figsize=figsize,
        sig_lw=sig_lw,
        levels_lw=levels_lw,
        ylim=ylim,
        highlight_ranges=highlight_ranges,
        t_as_u=t_as_u,
    )
    return fig_ax


def plot_ref_region_metrics(
    ref_reg,
    samples_metrics,
    fig_axs=None,
    fig_width=DEFAULT_PLOT_FIG_SIZE[0],
    facet_height=DEFAULT_PLOT_FIG_SIZE[1],
    highlight_ranges=None,
    sample_names=None,
    t_as_u=False,
):
    metric_names = list(samples_metrics[0].keys())
    if fig_axs is None:
        fig, axs = plt.subplots(
            len(metric_names),
            figsize=(fig_width, facet_height * len(metric_names)),
            sharex=True,
        )
    else:
        fig, axs = fig_axs
    if sample_names is None:
        sample_names = [f"Sample{idx}" for idx in range(len(samples_metrics))]

    for ax, metric_name in zip(axs, metric_names):
        samples_num_reads = np.array(
            [sm[metric_name].shape[0] for sm in samples_metrics]
        )
        plot_data = pd.DataFrame(
            {
                metric_name: np.concatenate(
                    [sm[metric_name].flatten() for sm in samples_metrics]
                ),
                "Position": np.tile(
                    np.arange(ref_reg.len), [samples_num_reads.sum()]
                ),
                "Sample": np.repeat(
                    sample_names, samples_num_reads * ref_reg.len
                ),
            }
        )
        plot_data = plot_data.loc[~np.isnan(plot_data[metric_name])]

        with np.errstate(invalid="ignore"):
            _ = sns.boxplot(
                data=plot_data,
                x="Position",
                y=metric_name,
                hue="Sample",
                ax=ax,
            )
        ax.set_xlabel("", fontsize=45)
        if metric_name == "dwell":
            ax.set_ylim(2, 80)
            ax.set_yscale("log")
        if metric_name.endswith("mean"):
            ax.set_ylim(-1.5, 2)
        if metric_name.endswith("sd"):
            ax.set_ylim(-0.005, 0.3)
            ax.set_xlabel("Position", fontsize=45)
        ax.set_ylabel(metric_name, fontsize=45)
        ax.tick_params(labelsize=16, axis="x")
        ax.tick_params(labelsize=32, axis="y")
    return fig, axs


def plot_metric_at_ref_region(
    ref_reg,
    pod5_and_bam_fhs,
    metric="dwell_trimmean",
    sig_map_refiner=None,
    skip_sig_map_refine=False,
    max_reads=None,
    reverse_signal=False,
    t_as_u=False,
    fig_axs=None,
    fig_width=DEFAULT_PLOT_FIG_SIZE[0],
    facet_height=DEFAULT_PLOT_FIG_SIZE[1],
    highlight_ranges=None,
    **kwargs,
):
    """Plot signal from reads at a reference region.

    Args:
        ref_reg (RefRegion): Reference position to plot
        pod5_and_bam_fhs (list): List of 2-tuples containing sorted and indexed
            BAM file handles and POD5 file handles
        metric (str): Named metric (e.g. dwell, mean, sd). Should be a key
            in metrics.METRIC_FUNCS
        sig_map_refiner (SigMapRefiner): For signal mapping and level extract
        skip_sig_map_refine (bool): Skip signal mapping refinement
        max_reads (int): Maximum reads to plot
        reverse_signal (bool): Is nanopore signal 3'>5' orientation?
        t_as_u (bool): Plot T bases as U (RNA)
        fig_axs (tuple): If None, new figure/axes will be opened. Should be
            same length as the number of metrics computed.
        fig_width (float): Figure width
        facet_height (float): Height of each facet (one per metric)
        highlight_ranges (iterable): Elements should be 3-tuples containing
            reference start, end and color values.

    Returns:
        matplotlib axis
    """
    samples_metrics, all_bam_reads = get_ref_reg_samples_metrics(
        ref_reg,
        pod5_and_bam_fhs,
        metric=metric,
        sig_map_refiner=sig_map_refiner,
        skip_sig_map_refine=skip_sig_map_refine,
        max_reads=max_reads,
        reverse_signal=reverse_signal,
        **kwargs,
    )
    fig_axs = plot_ref_region_metrics(
        ref_reg,
        samples_metrics,
        fig_axs=fig_axs,
        fig_width=fig_width,
        facet_height=facet_height,
        highlight_ranges=highlight_ranges,
        t_as_u=t_as_u,
    )
    plt.tight_layout()
    return fig_axs


###########
# IO Read #
###########


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
    ref_reg: RefRegion = None
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
        read.ref_reg = None
        return read, simplex_duplex_mapping.duplex_offset

    def add_alignment(
        self, alignment_record, parse_ref_align=True, reverse_signal=False
    ):
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
                reverse_signal=reverse_signal,
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

        self.ref_reg = RefRegion(
            ctg=alignment_record.reference_name,
            strand="-" if alignment_record.is_reverse else "+",
            start=alignment_record.reference_start,
        )
        try:
            self.ref_seq = alignment_record.get_reference_sequence().upper()
        except ValueError:
            LOGGER.debug(
                "Reference sequence requested, but could not be extracted. "
                "Do reads contain MD tags?"
            )
            self.ref_seq = None
        self.cigar = alignment_record.cigartuples
        if alignment_record.is_reverse:
            if self.ref_seq is not None:
                self.ref_seq = util.revcomp(self.ref_seq)
            self.cigar = self.cigar[::-1]
        if self.ref_reg.ctg is not None and self.ref_seq is not None:
            self.ref_to_signal = DC.compute_ref_to_signal(
                query_to_signal=self.query_to_signal,
                cigar=self.cigar,
            )
            # +1 because knots include the end position of the last base
            assert self.ref_to_signal.size == len(self.ref_seq) + 1, (
                "discordant ref seq lengths: move+cigar:"
                f"{self.ref_to_signal.size} ref_seq:{len(self.ref_seq)}"
            )
            self.ref_reg.end = self.ref_reg.start + self.ref_to_signal.size - 1

    @classmethod
    def from_pod5_and_alignment(
        cls, pod5_read_record, alignment_record, reverse_signal=False
    ):
        """Initialize read from pod5 and pysam records

        Args:
            pod5_read_record (pod5.ReadRecord)
            alignment_record (pysam.AlignedSegment)
        """
        dacs = pod5_read_record.signal
        if reverse_signal:
            dacs = dacs[::-1]
        read = Read(
            read_id=str(pod5_read_record.read_id),
            dacs=dacs,
            shift_dacs_to_pa=pod5_read_record.calibration.offset,
            scale_dacs_to_pa=pod5_read_record.calibration.scale,
        )
        read.add_alignment(alignment_record, reverse_signal=reverse_signal)
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
                )
                assert self.ref_to_signal.size == len(self.ref_seq) + 1, (
                    "discordant ref seq lengths: move+cigar:"
                    f"{self.ref_to_signal.size} ref_seq:{len(self.ref_seq)}"
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
        if self.ref_reg is None or self.ref_seq is None:
            raise RemoraError("Cannot extract focus positions without mapping")
        ref_reg = self.ref_reg
        ref_len = len(self.ref_seq)
        try:
            cs_focus_pos = select_focus_positions[(ref_reg.ctg, ref_reg.strand)]
        except KeyError:
            # no focus positions on contig/strand
            return np.array([], dtype=int)

        read_focus_ref_reg = np.array(
            sorted(
                set(range(ref_reg.start, ref_reg.start + ref_len)).intersection(
                    cs_focus_pos
                )
            ),
            dtype=int,
        )
        return (
            read_focus_ref_reg - ref_reg.start
            if ref_reg.strand == "+"
            else ref_reg.start + ref_len - read_focus_ref_reg[::-1] - 1
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
        mapping = DC.make_sequence_coordinate_mapping(self.cigar).astype(int)

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

    def extract_ref_reg(self, ref_reg):
        """Extract region of read from reference coordinates.

        Args:
            ref_reg (RefRegion): Reference region

        Returns:
            ReadRefReg object
        """
        if self.ref_to_signal is None:
            raise RemoraError(
                "Cannot extract reference region from unmapped read"
            )
        if ref_reg.start >= self.ref_reg.start + self.ref_seq_len:
            raise RemoraError("Reference region starts after read ends")
        if ref_reg.end < self.ref_reg.start:
            raise RemoraError("Reference region ends before read starts")

        if self.ref_reg.strand == "+":
            reg_st_within_read = max(0, ref_reg.start - self.ref_reg.start)
            reg_en_within_read = ref_reg.end - self.ref_reg.start
        else:
            reg_st_within_read = max(0, self.ref_reg.end - ref_reg.end)
            reg_en_within_read = self.ref_reg.end - ref_reg.start
        reg_seq_to_sig = self.ref_to_signal[
            reg_st_within_read : reg_en_within_read + 1
        ].copy()
        reg_sig = self.norm_signal[reg_seq_to_sig[0] : reg_seq_to_sig[-1]]
        reg_seq = self.ref_seq[reg_st_within_read:reg_en_within_read]
        reg_seq_to_sig -= reg_seq_to_sig[0]
        read_reg_ref_st = max(self.ref_reg.start, ref_reg.start)
        # orient reverse strand reads on the reference
        if self.ref_reg.strand == "-":
            reg_sig = reg_sig[::-1]
            reg_seq = reg_seq[::-1]
            reg_seq_to_sig = reg_seq_to_sig[-1] - reg_seq_to_sig[::-1]
        return ReadRefReg(
            read_id=self.read_id,
            norm_signal=reg_sig,
            seq=reg_seq,
            seq_to_sig_map=reg_seq_to_sig,
            ref_reg=RefRegion(
                self.ref_reg.ctg,
                self.ref_reg.strand,
                read_reg_ref_st,
                read_reg_ref_st + len(reg_seq),
            ),
        )

    def compute_per_base_metric(
        self,
        metric=None,
        metric_func=None,
        ref_anchored=True,
        region=None,
        signal_type="norm",
        **kwargs,
    ):
        """Compute a per-base metric from a read.

        Args:
            metric (str): Named metric (e.g. dwell, mean, sd). Should be a key
                in metrics.METRIC_FUNCS
            metric_func (Callable): Function taking three arguments sequence,
                signal and a sequence to signal mapping and return a dict of
                metric names to per base metric values.
            ref_anchored (bool): Compute metric against reference bases. If
                False, return basecall anchored metrics.
            region (RefRegion): Reference region from which to extract metrics
                of bases.
            signal_type (str): Type of signal. Should be one of: norm, pa, and
                dac
            **kwargs: Extra args to pass through to metric computations

        Returns:
            Result of metric_func. Preset metrics will return dict with metric
            name keys and a numpy array of per-base metric values.
        """
        if metric is not None:
            metric_func = METRIC_FUNCS[metric]
        assert (
            metric_func is not None
        ), "Must provide either metric or metric_func"
        st_buf = en_buf = 0
        if region is None:
            seq_to_sig = (
                self.ref_to_signal if ref_anchored else self.query_to_signal
            )
        else:
            if ref_anchored:
                if (
                    self.ref_reg.ctg != region.ctg
                    or self.ref_reg.strand != region.strand
                ):
                    raise RemoraError("Region contig/strand do not match read")
                if (
                    region.start >= self.ref_reg.end
                    or self.ref_reg.start >= region.end
                ):
                    raise RemoraError("Region does not overlap read.")
                if self.ref_reg.strand == "+":
                    st_coord = region.start - self.ref_reg.start
                    en_coord = region.end - self.ref_reg.start
                else:
                    st_coord = self.ref_reg.end - region.end
                    en_coord = self.ref_reg.end - region.start
                if st_coord < 0:
                    st_buf = -st_coord
                    st_coord = 0
                if en_coord > self.ref_seq_len:
                    en_buf = en_coord - self.ref_seq_len
                    en_coord = self.ref_seq_len
                seq_to_sig = self.ref_to_signal[st_coord : en_coord + 1]
            else:
                if region.start < 0 or region.start > self.seq_len:
                    raise RemoraError("Region does not overlap read.")
                # TODO deal with partially overlapping region
                st_buf = en_buf = 0
                seq_to_sig = self.query_to_signal[region.start : region.end]
        if signal_type == "norm":
            sig = self.norm_signal
        elif signal_type == "pa":
            sig = self.pa_signal
        elif signal_type == "dac":
            sig = self.dacs
        else:
            assert False, "signal_type must be norm, pa or dac"

        metrics_vals = metric_func(sig, seq_to_sig, **kwargs)
        if max(st_buf, en_buf) > 0:
            tmp_metrics_vals = {}
            for metric_name, metric_vals in metrics_vals.items():
                tmp_metrics_vals[metric_name] = np.full(region.len, np.NAN)
                tmp_metrics_vals[metric_name][
                    st_buf : st_buf + metric_vals.size
                ] = metric_vals
            metrics_vals = tmp_metrics_vals
        return metrics_vals


##########
# Duplex #
##########


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
