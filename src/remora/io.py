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

import pysam
import numpy as np
import polars as pl
import plotnine as p9
from tqdm import tqdm
from pod5 import DatasetReader
from pysam import AlignedSegment

from remora.metrics import METRIC_FUNCS
from remora.constants import PA_TO_NORM_SCALING_FACTOR
from remora import log, util, data_chunks, duplex_utils, RemoraError

LOGGER = log.get_logger()

_SIG_PROF_FN = os.getenv("REMORA_EXTRACT_SIGNAL_PROFILE_FILE")
_ALIGN_PROF_FN = os.getenv("REMORA_EXTRACT_ALIGN_PROFILE_FILE")
_ITER_ALIGN_PROF_FN = os.getenv("REMORA_ITER_ALIGN_PROFILE_FILE")
_ADD_SIG_PROF_FN = os.getenv("REMORA_ADD_SIG_PROFILE_FILE")

METRIC_PCTL_RANGE = (0.2, 99.8)
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
            all_mods.add(mod)
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
    if strand is None:
        return True
    return (
        strand not in "+-"
        or (strand == "+" and bam_read.is_forward)
        or (strand == "-" and bam_read.is_reverse)
    )


def get_bam_filename(bam_fh):
    if bam_fh.reference_filename is not None:
        return util.to_str(bam_fh.reference_filename)
    if bam_fh.filename is not None:
        return util.to_str(bam_fh.filename)


def get_parent_id(bam_read):
    try:
        # if pi tag is present this is a child read
        return bam_read.get_tag("pi")
    except KeyError:
        # else this is the parent read so return query_name
        return bam_read.query_name


@dataclass
class ReadIndexedBam:
    """Index bam file by read id. Note that the BAM file handle is closed after
    initialization. Any other operation (e.g. fetch, get_alignments,
    get_first_alignment) will open the pysam file handle and leave it open.
    This allows easier use with multiprocessing using standard operations.

    For BAM files with split reads, the parent read is indexed. This allows
    access after extracting signal from a POD5 file.

    Args:
        bam_path (str): Path to BAM file
        skip_non_primary (bool): Should non-primary alignmets be skipped
        req_tags (set): Skip reads without required tags
        read_id_converter (Callable[[str], str]): Function to convert read ids
            (e.g. for concatenated duplex read ids).
    """

    bam_path: str
    skip_non_primary: bool = True
    req_tags: set = None
    read_id_converter: Callable = None
    parent_read_id_subset: set = None
    child_read_id_subset: set = None

    @property
    def reference_filename(self):
        """Alias to mimic AlignmentFile attribute"""
        return self.bam_path

    @property
    def filename(self):
        """Alias to mimic AlignmentFile attribute"""
        return self.bam_path

    def has_index(self):
        """Alias to mimic AlignmentFile attribute"""
        if self.bam_fh is None:
            self.open()
        return self.bam_fh.has_index()

    def __post_init__(self):
        self.num_reads = None
        self.bam_fh = None
        self._bam_idx = None
        self._iter = None
        self.compute_read_index()

    def open(self):
        # hide warnings for no index when using unmapped or unsorted files
        self.pysam_save = pysam.set_verbosity(0)
        self.bam_fh = pysam.AlignmentFile(
            self.bam_path, mode="rb", check_sq=False
        )
        return self

    def close(self):
        self.bam_fh.close()
        self.bam_fh = None
        pysam.set_verbosity(self.pysam_save)

    def fetch(self, ctg, start, end, strand=None):
        if self.bam_fh is None:
            self.open()
        if not self.has_index():
            pysam.index(util.to_str(self.bam_path))
            self.close()
            self.open()
        for read in self.bam_fh.fetch(ctg, start, end):
            if strands_match(strand, read):
                yield read

    def compute_read_index(self):
        bam_was_closed = self.bam_fh is None
        if bam_was_closed:
            self.open()
        self._bam_idx = defaultdict(list)
        pbar = tqdm(
            smoothing=0,
            dynamic_ncols=True,
            unit=" Reads",
            desc="Indexing BAM by parent read id",
            disable=os.environ.get("LOG_SAFE", False),
        )
        self.num_records = 0
        self.skip_reasons = defaultdict(int)
        # iterating over file handle gives incorrect pointers
        while True:
            read_ptr = self.bam_fh.tell()
            try:
                read = next(self.bam_fh)
            except StopIteration:
                break
            pbar.update()
            if (
                self.child_read_id_subset is not None
                and read.query_name not in self.child_read_id_subset
            ):
                self.skip_reasons["Child read ID filtered"] += 1
                continue
            index_read_id = get_parent_id(read)
            if (
                self.parent_read_id_subset is not None
                and index_read_id not in self.parent_read_id_subset
            ):
                self.skip_reasons["Parent read ID filtered"] += 1
                continue
            if self.read_id_converter is not None:
                index_read_id = self.read_id_converter(index_read_id)
            if self.req_tags is not None:
                tags = set(tg[0] for tg in read.tags)
                missing_tags = self.req_tags.difference(tags)
                if len(missing_tags) > 0:
                    self.skip_reasons["Missing BAM tags"] += 1
                    continue
            if self.skip_non_primary and not read_is_primary(read):
                self.skip_reasons["Non-primary alignment"] += 1
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
            except OSError as e:
                LOGGER.debug(
                    f"Could not extract {read_id} from {self.bam_path} "
                    f"at {read_ptr}\nFULL_ERROR: {e}"
                )
                raise RemoraError(
                    "Could not extract BAM read. Ensure BAM file object was "
                    "closed before spawning process."
                )
            yield bam_read

    def get_first_alignment(self, read_id):
        return next(self.get_alignments(read_id))

    def __contains__(self, read_id):
        return read_id in self._bam_idx

    def __getitem__(self, read_id):
        return self._bam_idx[read_id]

    def __del__(self):
        if self.bam_fh is not None:
            self.bam_fh.close()

    @cached_property
    def read_ids(self):
        return list(self._bam_idx.keys())

    def __iter__(self):
        self.bam_fh.reset()
        self._iter = iter(self.bam_fh)
        return self._iter

    def __next__(self):
        if self._iter is None:
            self._iter = iter(self.bam_fh)
        return next(self._iter)


def get_read_ids(bam_idx, pod5_dr, num_reads, return_num_bam_reads=False):
    """Get overlapping read ids from bam index and pod5 file

    Args:
        bam_idx (ReadIndexedBam): Read indexed BAM
        pod5_dr (pod5.DatasetReader): POD5 Dataset Reader
        num_reads (int): Maximum number of reads, or None for no max
        return_num_child_reads (bool): Return the number of bam records (child
            reads and multiple mappings) with a parent read ID. When set to
            False the number of parent read IDs is returned.
    """
    LOGGER.info("Extracting read IDs from POD5")
    pod5_read_ids = set(pod5_dr.read_ids)
    both_read_ids = list(pod5_read_ids.intersection(bam_idx.read_ids))
    num_both_read_ids = sum(
        len(bam_idx._bam_idx[parent_read_id])
        for parent_read_id in both_read_ids
    )
    LOGGER.info(
        f"Found {bam_idx.num_records:,} valid BAM records. Found signal "
        f"in POD5 for {num_both_read_ids / bam_idx.num_records:.2%} of BAM "
        "records."
    )
    if not return_num_bam_reads:
        num_both_read_ids = len(both_read_ids)
    if num_reads is None:
        num_reads = num_both_read_ids
    else:
        num_reads = min(num_reads, num_both_read_ids)
    return both_read_ids, num_reads


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
        Requested pod5.ReadRecord objects
    """
    LOGGER.debug(f"Reading from POD5 at {pod5_path}")
    with DatasetReader(Path(pod5_path)) as pod5_dr:
        for read_num, read in enumerate(
            pod5_dr.reads(selection=read_ids, preload=["samples"])
        ):
            if num_reads is not None and read_num >= num_reads:
                LOGGER.debug(
                    f"Completed pod5 signal worker, reached {num_reads}."
                )
                return

            yield read
    LOGGER.debug("Completed pod5 signal worker")


def iter_signal(
    pod5_path, num_reads=None, read_ids=None, rev_sig=False, pa_scaling=None
):
    """Iterate io Read objects loaded from Pod5

    Args:
        pod5_path (str): Path to POD5 file
        num_reads (int): Maximum number of reads to iterate
        read_ids (iterable): Read IDs to extract
        rev_sig (bool): Should signal be reversed on reading
        pa_scaling (tuple): picoamp to zero-centered picoamp shift and scale

    Yields:
        2-tuple:
            1. remora.io.Read object
            2. Error text or None if no errors
    """
    pa_kwargs = {}
    if pa_scaling is not None:
        pa_kwargs["shift_pa_to_zc_pa"] = pa_scaling[0]
        pa_kwargs["scale_pa_to_zc_pa"] = pa_scaling[1]
    for pod5_read in iter_pod5_reads(
        pod5_path=pod5_path, num_reads=num_reads, read_ids=read_ids
    ):
        dacs = pod5_read.signal[::-1] if rev_sig else pod5_read.signal
        read = Read(
            read_id=str(pod5_read.read_id),
            dacs=dacs,
            shift_dacs_to_pa=pod5_read.calibration.offset,
            scale_dacs_to_pa=pod5_read.calibration.scale,
            **pa_kwargs,
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


def iter_alignments(
    bam_path,
    child_read_id_subset=None,
    parent_read_id_subset=None,
    skip_non_primary=True,
    pa_scaling=None,
):
    pa_kwargs = {}
    if pa_scaling is not None:
        pa_kwargs["shift_pa_to_zc_pa"] = pa_scaling[0]
        pa_kwargs["scale_pa_to_zc_pa"] = pa_scaling[1]
    with pysam.AlignmentFile(bam_path, mode="rb", check_sq=False) as bam_fh:
        for bam_read in bam_fh:
            if (
                child_read_id_subset is not None
                and bam_read.query_name not in child_read_id_subset
            ):
                yield None, "Child read ID filtered"
                continue
            parent_read_id = get_parent_id(bam_read)
            if (
                parent_read_id_subset is not None
                and parent_read_id not in parent_read_id_subset
            ):
                yield None, "Parent read ID filtered"
                continue
            if skip_non_primary and not read_is_primary(bam_read):
                yield None, "Non-primary alignment"
                continue
            io_read = Read(
                read_id=parent_read_id,
                **pa_kwargs,
            )
            io_read.add_alignment(bam_read)
            yield io_read, None


if _ITER_ALIGN_PROF_FN:
    _iter_align_wrapper = iter_alignments

    def iter_alignments(*args, **kwargs):
        import cProfile

        iter_align_prof = cProfile.Profile()
        retval = iter_align_prof.runcall(_iter_align_wrapper, *args, **kwargs)
        iter_align_prof.dump_stats(_ITER_ALIGN_PROF_FN)
        return retval


def add_signal(read_err, p5_read, rev_sig=False):
    io_read, err = read_err
    if io_read is None:
        return [read_err]
    try:
        io_read.add_signal(p5_read, reverse_signal=rev_sig)
    except RemoraError as e:
        LOGGER.debug(f"{io_read.read_id} Add signal error: {e}")
        return io_read, str(e)
    return io_read, ""


if _ADD_SIG_PROF_FN:
    _add_sig_wrapper = add_signal

    def add_signal(*args, **kwargs):
        import cProfile

        add_sig_prof = cProfile.Profile()
        retval = add_sig_prof.runcall(_add_sig_wrapper, *args, **kwargs)
        add_sig_prof.dump_stats(_ADD_SIG_PROF_FN)
        return retval


def extract_alignments(read_err, bam_idx, rev_sig=False, pa_scaling=None):
    io_read, err = read_err
    if io_read is None:
        return [read_err]
    read_alignments = []
    try:
        for bam_read in bam_idx.get_alignments(io_read.read_id):
            align_read = io_read.copy()
            try:
                align_read.add_alignment(
                    bam_read, reverse_signal=rev_sig, pa_scaling=pa_scaling
                )
                read_alignments.append((align_read, None))
            except RemoraError as e:
                LOGGER.debug(f"{io_read.read_id} Extract alignment error: {e}")
                read_alignments.append((align_read, str(e)))
    except RemoraError as e:
        LOGGER.debug(f"{io_read.read_id} Extract alignment error: {e}")
        return [(io_read, str(e))]
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
    if not bam_fh.has_index():
        bam_path = get_bam_filename(bam_fh)
        pysam.index(bam_path)
        bam_fh = pysam.AlignmentFile(bam_path)
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
    sig_start: int = 0

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
            sig_start=self.sig_start,
            **kwargs,
        )

    def plot_on_base_coords(self, **kwargs):
        """Plot signal on base/sequence coordinates. See global
        plot_on_base_coords function for kwargs.
        """
        return plot_on_base_coords(
            self.seq,
            self.norm_signal,
            self.seq_to_sig_map,
            start_base=self.ref_reg.start,
            rev_strand=self.ref_reg.strand == "-",
            xlab="Reference Position",
            **kwargs,
        )


@dataclass
class ReadBasecallRegion:
    read_id: str
    norm_signal: np.ndarray
    seq: str
    seq_to_sig_map: np.ndarray
    start: int
    sig_start: int = 0

    def plot_on_signal_coords(self, **kwargs):
        """Plot signal on signal coordinates. See global plot_on_signal_coords
        function for kwargs.
        """
        return plot_on_signal_coords(
            self.seq,
            self.norm_signal,
            self.seq_to_sig_map,
            sig_start=self.sig_start,
            **kwargs,
        )

    def plot_on_base_coords(self, **kwargs):
        """Plot signal on base/sequence coordinates. See global
        plot_on_base_coords function for kwargs.
        """
        return plot_on_base_coords(
            self.seq,
            self.norm_signal,
            self.seq_to_sig_map,
            start_base=self.start,
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
    if sig_map_refiner is None:
        levels = None
        context_int_seq = get_ref_int_seq_from_reads(
            ref_reg,
            bam_reads,
            ref_orient=False,
        )
        context_int_seq[np.equal(context_int_seq, -2)] = -1
        seq = util.int_to_seq(context_int_seq)
    else:
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
        if levels is not None:
            levels = levels[::-1]
    return seq, levels


def get_pod5_reads(pod5_dr, read_ids):
    return dict(
        (str(pod5_read.read_id), pod5_read)
        for pod5_read in pod5_dr.reads(selection=read_ids, preload=["samples"])
    )


def get_io_reads(
    bam_reads, pod5_dr, reverse_signal=False, missing_ok=False, pa_scaling=None
):
    pod5_reads = get_pod5_reads(
        pod5_dr, list(set(get_parent_id(bam_read) for bam_read in bam_reads))
    )
    io_reads = []
    for bam_read in bam_reads:
        parent_rid = get_parent_id(bam_read)
        p5_record = pod5_reads.get(parent_rid, None)
        if p5_record is None:
            if missing_ok:
                continue
            else:
                raise RemoraError(f"BAM record not found in POD5 {parent_rid}")
        io_reads.append(
            Read.from_pod5_and_alignment(
                pod5_read_record=p5_record,
                alignment_record=bam_read,
                reverse_signal=reverse_signal,
                pa_scaling=pa_scaling,
            )
        )
    return io_reads


def get_reads_reference_regions(
    ref_reg,
    pod5_bam_pairs,
    sig_map_refiner=None,
    skip_sig_map_refine=False,
    max_reads=50,
    reverse_signal=False,
    missing_ok=False,
    pa_scaling=None,
    signal_type="norm",
):
    all_bam_reads = []
    samples_read_ref_regs = []
    for pod5_dr, bam_fh in pod5_bam_pairs:
        sample_bam_reads = get_reg_bam_reads(ref_reg, bam_fh)
        if len(sample_bam_reads) == 0:
            raise RemoraError("No reads covering region")
        if max_reads is not None and len(sample_bam_reads) > max_reads:
            sample_bam_reads = random.sample(sample_bam_reads, max_reads)
        all_bam_reads.append(sample_bam_reads)
        io_reads = get_io_reads(
            sample_bam_reads,
            pod5_dr,
            reverse_signal,
            missing_ok=missing_ok,
            pa_scaling=pa_scaling,
        )
        if sig_map_refiner is not None and not skip_sig_map_refine:
            for io_read in io_reads:
                io_read.set_refine_signal_mapping(
                    sig_map_refiner, ref_mapping=True
                )
        samples_read_ref_regs.append(
            [
                io_read.extract_ref_reg(ref_reg, signal_type=signal_type)
                for io_read in io_reads
            ]
        )
    return samples_read_ref_regs, all_bam_reads


def get_ref_reg_sample_metrics(
    ref_reg,
    pod5_dr,
    bam_reads,
    metric,
    sig_map_refiner,
    skip_sig_map_refine=False,
    reverse_signal=False,
    ref_orient=True,
    missing_ok=False,
    pa_scaling=None,
    signal_type="norm",
    **kwargs,
):
    io_reads = get_io_reads(
        bam_reads,
        pod5_dr,
        reverse_signal,
        missing_ok=missing_ok,
        pa_scaling=pa_scaling,
    )
    if sig_map_refiner is not None and not skip_sig_map_refine:
        for io_read in io_reads:
            io_read.set_refine_signal_mapping(sig_map_refiner, ref_mapping=True)
    sample_metrics = [
        io_read.compute_per_base_metric(
            metric, region=ref_reg, signal_type=signal_type, **kwargs
        )
        for io_read in io_reads
    ]
    if len(sample_metrics) <= 0:
        return
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
    pod5_bam_pairs,
    sig_map_refiner=None,
    skip_sig_map_refine=False,
    max_reads=None,
    reverse_signal=False,
    metric="dwell_trimmean",
    missing_ok=False,
    **kwargs,
):
    all_bam_reads = []
    samples_metrics = []
    for pod5_dr, bam_fh in pod5_bam_pairs:
        sample_bam_reads = get_reg_bam_reads(ref_reg, bam_fh)
        if len(sample_bam_reads) == 0:
            raise RemoraError("No reads covering region")
        if max_reads is not None and len(sample_bam_reads) > max_reads:
            sample_bam_reads = random.sample(sample_bam_reads, max_reads)
        all_bam_reads.append(sample_bam_reads)
        sample_metrics = get_ref_reg_sample_metrics(
            ref_reg,
            pod5_dr,
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
    pod5_dr,
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
        pod5_dr,
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
    args[0] = DatasetReader(args[0])
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
    levels=None,
    sig_start=0,
    sig_lw=0.5,
    levels_lw=1,
    ylim=None,
    rev_strand=False,
    t_as_u=False,
    xlab="Signal Position",
    ylab="Normalized Signal",
    base_dividers=False,
    sig_pctl_range=METRIC_PCTL_RANGE,
):
    """Plot a single read on signal coordinates.

    Args:
        seq (str): Sequence to plot
        sig (np.array): Signal to plot
        seq_to_sig_map (np.array): Mapping from sequence to signal coordinates
        levels (np.array): Expected signal levels for bases in region
        sig_start (int): Signal start value
        sig_lw (int): Linewidth for signal lines
        levels_lw (int): Linewidth for level lines (if applicable)
        ylim (tuple): 2-tuple with y-axis limits
        rev_strand (bool): Plot bases upside down
        t_as_u (bool): Plot T bases as U (RNA)
        xlab (str): X-axis Label
        base_dividers (bool): Add vertical lines separating bases

    Returns:
        plotnine plot object. Use print(return_value) to display plot in in
        a notebook or return_value.save("remora_plot.pdf") to save to file.
    """
    if ylim is None:
        sig_min, sig_max = np.percentile(sig, sig_pctl_range)
        sig_diff = sig_max - sig_min
        ylim = (sig_min - sig_diff * 0.01, sig_max + sig_diff * 0.01)

    sig_df = pl.DataFrame(
        [sig, np.arange(sig_start, sig_start + sig.size)],
        schema=["Signal", xlab],
        orient="col",
    )
    if t_as_u:
        seq = util.t_to_u(seq)
    base_coords = pl.DataFrame(
        [
            sig_start + seq_to_sig_map[:-1],
            sig_start + seq_to_sig_map[1:],
            list(seq),
        ],
        schema=["base_st", "base_en", "base"],
        orient="col",
    )
    p = (
        p9.ggplot()
        + p9.geom_rect(
            p9.aes(
                xmin="base_st",
                xmax="base_en",
                fill="base",
                ymin=sig_min,
                ymax=sig_max,
            ),
            data=base_coords,
            alpha=0.1,
            show_legend=False,
        )
        + p9.geom_text(
            p9.aes(x="base_st", label="base", color="base", y=sig_min),
            data=base_coords,
            va="bottom",
            ha="left",
            size=8,
            angle=180 if rev_strand else 0,
            show_legend=False,
        )
        + p9.scale_fill_manual(BASE_COLORS)
        + p9.scale_color_manual(BASE_COLORS)
        + p9.geom_line(p9.aes(x=xlab, y="Signal"), size=sig_lw, data=sig_df)
        + p9.labels.xlab(xlab)
        + p9.labels.ylab(ylab)
        + p9.coords.coord_cartesian(ylim=ylim)
        + p9.theme(
            panel_grid_major_x=p9.element_blank(),
            panel_grid_minor_x=p9.element_blank(),
        )
    )
    if base_dividers:
        p += p9.geom_vline(
            p9.aes(xintercept="base_st"),
            data=base_coords[1:],
            color="black",
            alpha=0.5,
            size=0.05,
        )
    if levels is not None:
        level_df = pl.DataFrame(
            [
                sig_start + seq_to_sig_map[:-1],
                sig_start + seq_to_sig_map[1:],
                levels,
            ],
            schema=["base_st", "base_en", "level"],
            orient="col",
        )
        p += p9.geom_segment(
            p9.aes(x="base_st", xend="base_en", y="level", yend="level"),
            color="orange",
            alpha=0.5,
            size=levels_lw,
            data=level_df,
        )
    return p


def plot_on_base_coords(
    seq,
    sig,
    seq_to_sig_map,
    levels=None,
    start_base=0,
    sig_lw=0.5,
    levels_lw=1,
    ylim=None,
    rev_strand=False,
    t_as_u=False,
    xlab="Base Position",
    ylab="Normalized Signal",
    base_dividers=False,
    sig_pctl_range=METRIC_PCTL_RANGE,
):
    """Plot a single read on base/sequence coordinates.

    Args:
        start_base (int): Coordinate of first base in seq
        seq (str): Sequence to plot
        sig (np.array): Signal to plot
        seq_to_sig_map (np.array): Mapping from sequence to signal coordinates
        levels (np.array): Expected signal levels for bases in region
        sig_lw (int): Linewidth for signal lines
        levels_lw (int): Linewidth for level lines (if applicable)
        ylim (tuple): 2-tuple with y-axis limits
        rev_strand (bool): Plot bases upside down
        t_as_u (bool): Plot T bases as U (RNA)
        xlab (str): X-axis Label
        ylab (str): Y-axis Label
        base_dividers (bool): Add vertical lines separating bases

    Returns:
        plotnine plot objects. Use print(return_value) to display plot in in
        a notebook or return_value.save("remora_plot.pdf") to save to file.
    """
    if ylim is None:
        sig_min, sig_max = np.percentile(sig, sig_pctl_range)
        sig_diff = sig_max - sig_min
        ylim = (sig_min - sig_diff * 0.01, sig_max + sig_diff * 0.01)

    sig_df = pl.DataFrame(
        [
            sig,
            compute_base_space_sig_coords(seq_to_sig_map) + start_base,
        ],
        schema=["Signal", xlab],
        orient="col",
    )
    if t_as_u:
        seq = util.t_to_u(seq)
    base_coords = pl.DataFrame(
        [
            np.arange(start_base, start_base + len(seq)),
            np.arange(start_base + 1, start_base + len(seq) + 1),
            list(seq),
        ],
        schema=["base_st", "base_en", "base"],
        orient="col",
    )
    p = (
        p9.ggplot()
        + p9.geom_rect(
            p9.aes(
                xmin="base_st",
                xmax="base_en",
                fill="base",
                ymin=sig_min,
                ymax=sig_max,
            ),
            data=base_coords,
            alpha=0.1,
            show_legend=False,
        )
        + p9.geom_text(
            p9.aes(x="base_st", label="base", color="base", y=sig_min),
            data=base_coords,
            va="bottom",
            ha="left",
            size=8,
            angle=180 if rev_strand else 0,
            show_legend=False,
        )
        + p9.scale_fill_manual(BASE_COLORS)
        + p9.scale_color_manual(BASE_COLORS)
        + p9.geom_line(p9.aes(x=xlab, y="Signal"), size=sig_lw, data=sig_df)
        + p9.labels.xlab(xlab)
        + p9.labels.ylab(ylab)
        + p9.coords.coord_cartesian(ylim=ylim)
        + p9.theme(
            panel_grid_major_x=p9.element_blank(),
            panel_grid_minor_x=p9.element_blank(),
        )
    )
    if base_dividers:
        p += p9.geom_vline(
            p9.aes(xintercept="base_st"),
            data=base_coords[1:],
            color="black",
            alpha=0.5,
            size=0.05,
        )
    if levels is not None:
        level_df = pl.DataFrame(
            [
                np.arange(start_base, start_base + len(seq)),
                np.arange(start_base + 1, start_base + len(seq) + 1),
                levels,
            ],
            schema=["base_st", "base_en", "level"],
            orient="col",
        )
        p += p9.geom_segment(
            p9.aes(x="base_st", xend="base_en", y="level", yend="level"),
            color="orange",
            alpha=0.5,
            size=levels_lw,
            data=level_df,
        )
    return p


def plot_align(
    io_read,
    sig_st,
    sig_en,
    sig_mp=0,
    t_as_u=False,
    xlab="Signal Position",
    ylab="Normalized Signal",
    sig_pctl_range=METRIC_PCTL_RANGE,
    signal_type="norm",
):
    """Plot a single read in signal space with basecalls and reference
    alignment bases annotated.

    Args:
        io_read (Read): remora.io.Read object
        sig_st (int): Signal start coordinate
        sig_en (int): Signal end coordinate
        sig_mp (int): Signal midpoint (default 0)
        t_as_u (bool): Plot T bases as U (RNA)
        xlab (str): X-axis label
        ylab (str): Y-axis label
        signal_type (str): One of "norm" (default), "pa", "zc_pa", "dac"

    Returns:
        plotnine plot object. Use print(return_value) to display plot in in
        a notebook or return_value.save("remora_plot.pdf") to save to file.
    """
    ref_st = np.searchsorted(io_read.ref_to_signal[:-1], sig_st - 1)
    ref_en = np.searchsorted(io_read.ref_to_signal[:-1], sig_en)
    ref_seq = io_read.ref_seq[ref_st:ref_en]
    if t_as_u:
        ref_seq = util.t_to_u(ref_seq)
    ref_coords = pl.DataFrame(
        [
            io_read.ref_to_signal[ref_st:ref_en].clip(sig_st, sig_en),
            io_read.ref_to_signal[ref_st + 1 : ref_en + 1].clip(sig_st, sig_en),
            list(ref_seq),
        ],
        schema=["sig_st", "sig_en", "base"],
        orient="col",
    )
    bc_st = np.searchsorted(io_read.query_to_signal, sig_st - 1)
    bc_en = np.searchsorted(io_read.query_to_signal, sig_en)
    bc_seq = io_read.seq[bc_st:bc_en]
    if t_as_u:
        bc_seq = util.t_to_u(bc_seq)
    bc_coords = pl.DataFrame(
        [
            io_read.query_to_signal[bc_st:bc_en].clip(sig_st, sig_en),
            io_read.query_to_signal[bc_st + 1 : bc_en + 1].clip(sig_st, sig_en),
            list(bc_seq),
        ],
        schema=["sig_st", "sig_en", "base"],
        orient="col",
    )

    sig = io_read.get_sig_type(signal_type)[sig_st:sig_en]
    sig_min, sig_max = np.percentile(sig, sig_pctl_range)
    sig_diff = sig_max - sig_min
    ylim = (sig_min - sig_diff * 0.02, sig_max + sig_diff * 0.03)
    if sig_mp is None:
        sig_mp = sum(ylim) / 2

    return (
        p9.ggplot()
        + p9.geom_rect(
            p9.aes(
                xmin="sig_st",
                xmax="sig_en",
                fill="base",
                ymin=sig_mp,
                ymax=sig_max,
            ),
            data=ref_coords,
            alpha=0.3,
            show_legend=False,
        )
        + p9.geom_rect(
            p9.aes(
                xmin="sig_st",
                xmax="sig_en",
                fill="base",
                ymin=sig_min,
                ymax=sig_mp,
            ),
            data=bc_coords,
            alpha=0.3,
            show_legend=False,
        )
        + p9.geom_text(
            p9.aes(x="sig_st", label="base", color="base", y=sig_max),
            data=ref_coords,
            va="top",
            ha="left",
            show_legend=False,
        )
        + p9.geom_text(
            p9.aes(x="sig_st", label="base", color="base", y=sig_min),
            data=bc_coords,
            va="bottom",
            ha="left",
            show_legend=False,
        )
        + p9.scale_fill_manual(BASE_COLORS)
        + p9.scale_color_manual(BASE_COLORS)
        + p9.geom_text(
            p9.aes(
                y=[sig_min, sig_max],
                label=["Basecalls", "Reference"],
                va=["top", "bottom"],
            ),
            x=sig_st,
            ha="left",
        )
        + p9.geom_hline(yintercept=sig_mp, linetype="dashed", alpha=0.2)
        + p9.geom_line(p9.aes(x=np.arange(sig_st, sig_en), y=sig))
        + p9.labels.xlab(xlab)
        + p9.labels.ylab(ylab)
        + p9.coords.coord_cartesian(ylim=ylim)
        + p9.theme(
            panel_grid_major_x=p9.element_blank(),
            panel_grid_minor_x=p9.element_blank(),
        )
    )


def plot_ref_region_reads(
    ref_reg,
    samples_read_ref_regs,
    seq,
    levels,
    sig_lw=0.5,
    sig_alpha=0.2,
    levels_lw=1,
    sample_names=None,
    sample_colors=None,
    ylim=None,
    highlight_ranges=None,
    t_as_u=False,
    sig_pctl_range=METRIC_PCTL_RANGE,
    xlab="Reference Position",
    ylab="Normalized Signal",
):
    """Plot read signals over reference region

    Args:
        sample_names (list): Sample names. Default: ["Sample1", "Sample2", ... ]

    Returns:
        plotnine plot object. Use print(return_value) to display plot in in
        a notebook or return_value.save("remora_plot.pdf") to save to file.
    """
    if sample_names is None:
        sample_names = [
            f"Sample{samp_idx}"
            for samp_idx in range(len(samples_read_ref_regs))
        ]
    sig_df = []
    for sample_name, s_reads in zip(sample_names, samples_read_ref_regs):
        sig_df.append(
            pl.DataFrame(
                [
                    np.concatenate([read.ref_sig_coords for read in s_reads]),
                    np.concatenate([read.norm_signal for read in s_reads]),
                    [
                        f"{sample_name}_{read_idx}"
                        for read_idx, read in enumerate(s_reads)
                        for _ in range(read.norm_signal.size)
                    ],
                ],
                schema=["Reference Position", "Signal", "Read"],
                orient="col",
            ).with_columns(pl.lit(sample_name).alias("Sample"))
        )
    sig_df = pl.concat(sig_df)
    if ylim is None:
        sig_min, sig_max = np.percentile(sig_df["Signal"], sig_pctl_range)
        sig_diff = sig_max - sig_min
        ylim = (sig_min - sig_diff * 0.01, sig_max + sig_diff * 0.01)
    if levels is not None:
        level_df = pl.DataFrame(
            [
                np.arange(ref_reg.start, ref_reg.end),
                np.arange(ref_reg.start + 1, ref_reg.end + 1),
                levels,
            ],
            schema=["base_st", "base_en", "level"],
            orient="col",
        )
    if t_as_u:
        seq = util.t_to_u(seq)
    base_coords = pl.DataFrame(
        [
            np.arange(ref_reg.start, ref_reg.end),
            seq,
        ],
        schema=["pos", "base"],
        orient="col",
    )
    p = p9.ggplot()
    if highlight_ranges is not None:
        # plot highlight regions first to be behind signal
        highlight_df = pl.DataFrame(
            highlight_ranges,
            schema=["start", "end", "color"],
            orient="row",
        ).with_columns(
            pl.lit(sig_min).alias("sig_min"), pl.lit(sig_max).alias("sig_max")
        )
        p = (
            p
            + p9.geom_rect(
                p9.aes(
                    xmin="start",
                    xmax="end",
                    ymin="sig_min",
                    ymax="sig_max",
                ),
                # keep fill out of legends (in case fill is used later)
                fill=highlight_df["color"],
                data=highlight_df,
                alpha=0.2,
            )
            + p9.scale_fill_manual(
                dict((c, c) for _, _, c in highlight_ranges), guide=False
            )
        )
    p = (
        p
        + p9.geom_text(
            p9.aes(x="pos", label="base", y=sig_min),
            # keep color out of legend (used for sample names in this plot)
            color=[BASE_COLORS[b] for b in seq],
            data=base_coords,
            va="bottom",
            ha="left",
            size=8,
            angle=180 if ref_reg.strand == "-" else 0,
        )
        + p9.geom_vline(
            p9.aes(xintercept="pos"),
            data=base_coords[1:],
            color="black",
            alpha=0.5,
            size=0.02,
        )
        + p9.geom_line(
            p9.aes(
                x="Reference Position", y="Signal", color="Sample", group="Read"
            ),
            alpha=sig_alpha,
            size=sig_lw,
            data=sig_df,
        )
        + p9.ylim(*ylim)
        + p9.xlim(ref_reg.start, ref_reg.end)
        + p9.labels.xlab(xlab)
        + p9.labels.ylab(ylab)
        + p9.theme(
            panel_grid_major_x=p9.element_blank(),
            panel_grid_minor_x=p9.element_blank(),
        )
    )
    if levels is not None:
        p = p + p9.geom_segment(
            p9.aes(x="base_st", xend="base_en", y="level", yend="level"),
            color="orange",
            alpha=0.5,
            size=levels_lw,
            data=level_df,
        )

    if sample_colors is not None:
        p = p + p9.scale_color_manual(sample_colors, limits=sample_names)
    return p


def plot_signal_at_ref_region(
    ref_reg,
    pod5_bam_pairs,
    sig_map_refiner=None,
    skip_sig_map_refine=False,
    max_reads=50,
    reverse_signal=False,
    pa_scaling=None,
    signal_type="norm",
    missing_ok=False,
    **kwargs,
):
    """Plot signal from reads at a reference region.

    Args:
        ref_reg (RefRegion): Reference position at which to plot signal
        pod5_bam_pairs (list): List of samples. Each element should be a
            2-tuple of 1. pod5.DatasetReader and 2. pysam.AlignmentFile
        sig_map_refiner (SigMapRefiner): For signal mapping and level extract
        skip_sig_map_refine (bool): Skip signal mapping refinement
        max_reads (int): Maximum reads to plot (TODO: add overplotting options)
        reverse_signal (bool): Is nanopore signal 3'>5' orientation?
        pa_scaling (tuple): pico-amp scaling values
        signal_type (string): Signal normalization string
        missing_ok (bool): Ok for BAM reads to be missing from POD5?
        **kwargs: Passed on to plot_ref_region_reads

    Returns:
        plotnine plot object. Use print(return_value) to display plot in in
        a notebook or return_value.save("remora_plot.pdf") to save to file.
    """
    samples_read_ref_regs, reg_bam_reads = get_reads_reference_regions(
        ref_reg,
        pod5_bam_pairs,
        sig_map_refiner=sig_map_refiner,
        skip_sig_map_refine=skip_sig_map_refine,
        max_reads=max_reads,
        reverse_signal=reverse_signal,
        pa_scaling=pa_scaling,
        signal_type=signal_type,
        missing_ok=missing_ok,
    )
    seq, levels = get_ref_seq_and_levels_from_reads(
        ref_reg, chain(*reg_bam_reads), sig_map_refiner
    )
    return plot_ref_region_reads(
        ref_reg,
        samples_read_ref_regs,
        seq,
        levels,
        **kwargs,
    )


def plot_ref_region_metrics(
    ref_reg,
    samples_metrics,
    all_bam_reads,
    sample_names=None,
    sample_colors=None,
    geom=p9.geom_boxplot,
    geom_kwargs=None,
    metric_pctl_range=METRIC_PCTL_RANGE,
    xlab="Reference Position",
):
    ref_pos = list(range(ref_reg.start, ref_reg.end))
    metric_names = list(samples_metrics[0].keys())
    if sample_names is None:
        sample_names = [f"Sample{idx}" for idx in range(len(samples_metrics))]
    dfs = dict((metric, []) for metric in metric_names)
    for samp_name, metric_arrs, samp_bam_reads in zip(
        sample_names, samples_metrics, all_bam_reads
    ):
        for metric, metric_arr in metric_arrs.items():
            df = pl.from_numpy(
                metric_arr, schema=list(map(str, ref_pos))
            ).with_columns(
                [
                    pl.Series(
                        name="read_id",
                        values=[read.query_name for read in samp_bam_reads],
                    ),
                    pl.lit(samp_name).alias("Sample"),
                ]
            )
            dfs[metric].append(
                df.melt(
                    id_vars=["read_id", "Sample"],
                    variable_name="Position",
                ).with_columns(
                    [
                        pl.col("Position").cast(pl.Int32),
                        pl.col("value").cast(pl.Float64),
                    ]
                )
            )
    dfs = dict((metric, pl.concat(mdfs)) for metric, mdfs in dfs.items())

    if geom_kwargs is None:
        geom_kwargs = {}
    plots = []
    for metric in samples_metrics[0].keys():
        m_min, m_max = np.nanpercentile(dfs[metric]["value"], metric_pctl_range)
        m_diff = m_max - m_min
        ylim = (m_min - m_diff * 0.01, m_max + m_diff * 0.01)
        plots.append(
            p9.ggplot(dfs[metric])
            + geom(
                p9.aes(y="value", x="factor(Position)", fill="Sample"),
                **geom_kwargs,
            )
            + p9.scale_x_discrete(
                labels=[str(rp) if rp % 5 == 0 else "" for rp in ref_pos]
            )
            + p9.ylim(*ylim)
            + p9.labels.ylab(metric)
            + p9.labels.xlab(xlab)
        )
        if sample_colors is not None:
            plots[-1] = plots[-1] + p9.scale_fill_manual(
                sample_colors, limits=sample_names
            )
    return plots


def plot_metric_at_ref_region(
    ref_reg,
    pod5_bam_pairs,
    metric="dwell_trimmean",
    sig_map_refiner=None,
    max_reads=None,
    reverse_signal=False,
    sample_names=None,
    sample_colors=None,
    geom=p9.geom_boxplot,
    geom_kwargs=None,
    **kwargs,
):
    """Plot signal from reads at a reference region.

    Args:
        ref_reg (RefRegion): Reference position to plot
        pod5_bam_pairs (list): List of samples. Each element should be a
            2-tuple of 1. pod5.DatasetReader and 2. pysam.AlignmentFile
        metric (str): Named metric (e.g. dwell, mean, sd). Should be a key
            in metrics.METRIC_FUNCS
        sig_map_refiner (SigMapRefiner): For signal mapping and level extract
        max_reads (int): Maximum reads to plot
        reverse_signal (bool): Is nanopore signal 3'>5' orientation?

    Returns:
        plotnine plot objects. Use print(return_value) to display plot in in
        a notebook or return_value.save("remora_plot.pdf") to save to file.
    """
    samples_metrics, all_bam_reads = get_ref_reg_samples_metrics(
        ref_reg,
        pod5_bam_pairs,
        metric=metric,
        sig_map_refiner=sig_map_refiner,
        max_reads=max_reads,
        reverse_signal=reverse_signal,
        **kwargs,
    )
    return plot_ref_region_metrics(
        ref_reg,
        samples_metrics,
        all_bam_reads,
        sample_names=sample_names,
        sample_colors=sample_colors,
        geom=geom,
        geom_kwargs=geom_kwargs,
    )


###########
# IO Read #
###########


@dataclass
class Read:
    """Input/Output Read. Contains signal, basecalls, mapping between the two,
    various signal scaling parameters, and reference alignment.

    All scaling parameters use the following forumla:
        output = (input - shift) / sca;e

    Args:
        read_id (str): Read identifier. Note that for split reads this should
            be the parent ID (original POD5 ID). See child_read_id for the
            read ID listed in the BAM file.
        dacs (np.ndarray): Data ACquisition signal values. As read from the
            pod5_read.signal attribute. Note for "reverse signal" reads (mostly
            RNA) signal should be in the 5' to 3' orientation (opposite of
            original sequencing 3' to 5' direction)
        seq (str): Basecalled sequence
        stride (int): Basecalling model stride. Move table (mv_table) entries
            occur at regular stride intervals through the signal.
        mv_table (np.array): Move table. `1` entries indicate base output
            position. See Dorado documentation/SAM.md. Attribute may be
            deprecated in the future.
        query_to_signal (np.ndarray): Array with length one longer than
            basecalled sequence (`len(io_read.seq) + 1`) where entries
            represent assigned position in io_read.dacs. This attribute is
            updated when io_read.set_refine_signal_mapping is called.
        shift_dacs_to_pa (float): Shift parameter to convert from DACs to
            picoamp scaling. See io.Read.pa_signal for implementation.
        scale_dacs_to_pa (float): Scale parameter to convert from DACs to
            picoamp scaling. See io.Read.pa_signal for implementation.
        shift_pa_to_norm (float): Shift parameter to convert from picoamp to
            norm scaling. Extracted from BAM record tags.
        scale_pa_to_norm (float): Scale parameter to convert from picoamp to
            norm scaling. Extracted from BAM record tags.
        shift_dacs_to_norm (float): Shift parameter to convert from DACs to
            norm scaling. See io.Read.norm_signal for implementation.
        scale_dacs_to_norm (float): Scale parameter to convert from DACs to
            norm scaling. See io.Read.norm_signal for implementation.
        shift_pa_to_zc_pa (float): Shift parameter to convert from picoamp to
            zero-centered picoamp scaling. Extracted from picoamp scaling
            Dorado basecalling model
        scale_pa_to_zc_pa (float): Scale parameter to convert from picoamp to
            zero-centered picoamp scaling. Extracted from picoamp scaling
            Dorado basecalling model
        ref_seq (str): Reference sequence for mapping.
        ref_reg (RefRegion): Reference mapping region coordinates.
        cigar (list): pysam.AlignedSegment.cigartuples
        ref_to_signal (np.ndarray): Array with length one longer than
            reference sequence (`len(io_read.ref_seq) + 1`) where entries
            represent assigned position in io_read.dacs. This attribute is
            updated when io_read.set_refine_signal_mapping(ref_mapping=True)
            is called.
        full_align (dict): Dictionary representation of BAM record.
        is_mapped (bool): Is this record mapped to the reference?
        read_metrics (dict): Metrics related to this read. See util.READ_METRICS
    """

    read_id: str
    dacs: np.ndarray = None
    seq: str = None
    stride: int = None
    mv_table: np.array = None
    query_to_signal: np.ndarray = None
    shift_dacs_to_pa: float = None
    scale_dacs_to_pa: float = None
    shift_pa_to_norm: float = None
    scale_pa_to_norm: float = None
    shift_dacs_to_norm: float = None
    scale_dacs_to_norm: float = None
    shift_pa_to_zc_pa: float = None
    scale_pa_to_zc_pa: float = None
    ref_seq: str = None
    ref_reg: RefRegion = None
    cigar: list = None
    ref_to_signal: np.ndarray = None
    full_align: dict = None
    is_mapped: bool = False
    read_metrics: dict = None
    _child_read_id: str = None
    _sig_len: int = None
    _adjusted_query_to_signal: bool = False
    _trim_tags: dict = None

    @property
    def pa_signal(self):
        """Picoamp convereted signal. Shift and scale values are determined
        from the instrument. These values are generally not recommended for any
        type of analysis.
        """
        if self.scale_dacs_to_pa is None or self.shift_dacs_to_pa is None:
            raise RemoraError("pA scaling factors not set")
        return (self.dacs - self.shift_dacs_to_pa) / self.scale_dacs_to_pa

    @property
    def zero_centered_pa_signal(self):
        """Zero-centered picoamp convereted signal. Shift and scale values
        are determined from the instrument plus the basecalling model. These
        values are linked to a basecalling model and should only be used with
        these models.
        """
        return (self.dacs - self.shift_dacs_to_zc_pa) / self.scale_dacs_to_zc_pa

    @property
    def norm_signal(self):
        """Normalized signal, generally aiming to standardize the signal with
        mean=0 and SD=1, though different methods may be applied to perform
        this operation robustly.
        """
        if self.scale_dacs_to_norm is None or self.shift_dacs_to_norm is None:
            raise RemoraError("Norm scaling factors not set")
        return (self.dacs - self.shift_dacs_to_norm) / self.scale_dacs_to_norm

    def compute_pa_to_norm_scaling(self, factor=PA_TO_NORM_SCALING_FACTOR):
        self.shift_pa_to_norm = np.median(self.pa_signal)
        self.scale_pa_to_norm = max(
            1.0,
            np.median(np.abs(self.pa_signal - self.shift_pa_to_norm)) * factor,
        )

    @property
    def sig_len(self):
        if self._sig_len is None and self.dacs is not None:
            self._sig_len = self.dacs.size
        return self._sig_len

    @property
    def seq_len(self):
        if self.query_to_signal is None:
            if self.seq is None:
                return None
            return len(self.seq)
        return self.query_to_signal.size - 1

    @property
    def ref_seq_len(self):
        if self.ref_to_signal is None:
            if self.ref_seq is None:
                return None
            return len(self.ref_seq)
        return self.ref_to_signal.size - 1

    @property
    def child_read_id(self):
        if self._child_read_id is None:
            return self.read_id
        return self._child_read_id

    @property
    def shift_dacs_to_zc_pa(self):
        """Shift factor from DAC signal values to zero-centered picoamp"""
        if (
            self.shift_dacs_to_pa is None
            or self.scale_dacs_to_pa is None
            or self.shift_pa_to_zc_pa is None
        ):
            raise RemoraError("Zero-centered pA scaling factors not set")
        return self.shift_dacs_to_pa + (
            self.scale_dacs_to_pa * self.shift_pa_to_zc_pa
        )

    @property
    def scale_dacs_to_zc_pa(self):
        """Scale factor from DAC signal values to zero-centered picoamp"""
        if self.scale_dacs_to_pa is None or self.scale_pa_to_zc_pa is None:
            raise RemoraError("Zero-centered pA scaling factors not set")
        return self.scale_dacs_to_pa * self.scale_pa_to_zc_pa

    def prune(self, drop_tags=None):
        """Drop larger memory arrays: dacs, mv_table, *_to_signal"""
        if drop_tags is not None:
            # drop tags from input reads
            self.full_align["tags"] = [
                tag
                for tag in self.full_align["tags"]
                if tag[:2] not in drop_tags
            ]
        # access sig len to save the value
        self.sig_len
        self.dacs = None
        self.mv_table = None
        self.query_to_signal = None
        self.ref_to_signal = None
        return self

    def with_duplex_alignment(self, duplex_read_alignment, duplex_orientation):
        """Return copy with alignment to duplex sequence

        Args:
            duplex_read_alignment (AlignedSegment): pysam alignment record
                containing duplex sequence
            duplex_orientation (bool): Is duplex mapping in the same
                orientation as this simplex read.
        """
        if self.query_to_signal is None:
            raise RemoraError("requires query_to_signal")
        if duplex_read_alignment.query_sequence is None:
            raise RemoraError("no duplex base call sequence?")
        if len(duplex_read_alignment.query_sequence) <= 0:
            raise RemoraError("duplex base call sequence is empty string?")

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
        simplex_duplex_mapping = duplex_utils.map_simplex_to_duplex(
            simplex_seq=read.seq, duplex_seq=duplex_read_sequence
        )
        # duplex_read_to_signal is a mapping of each position in the duplex
        # sequence to a signal datum from the read
        query_to_signal = read.query_to_signal
        duplex_to_read_signal = data_chunks.map_ref_to_signal(
            query_to_signal=query_to_signal,
            ref_to_query_knots=simplex_duplex_mapping.duplex_to_simplex_mapping,
        )
        read.seq = simplex_duplex_mapping.trimmed_duplex_seq
        read.query_to_signal = duplex_to_read_signal

        read.ref_seq = None
        read.ref_to_signal = None
        read.ref_reg = None
        return read, simplex_duplex_mapping.duplex_offset

    def adjust_move_table(self, reverse_signal=False, check=True):
        if self._adjusted_query_to_signal:
            LOGGER.debug("Read move table was previously adjusted")
            return
        # add last point to query_to_signal
        self.query_to_signal = np.concatenate(
            [self.query_to_signal, [self.sig_len]]
        )
        if reverse_signal:
            self.query_to_signal = self.sig_len - self.query_to_signal[::-1]
        self._adjusted_query_to_signal = True
        if check:
            if self.query_to_signal.size - 1 != len(self.seq):
                raise RemoraError("Move table discordant with basecalls")
            if self.mv_table.size != self.sig_len // self.stride:
                raise RemoraError("Move table discordant with signal")

    def trim_signal(self, reverse_signal):
        if self.dacs is None or self._trim_tags is None:
            return
        if reverse_signal:
            self.dacs = self.dacs[::-1]
        # trim for split read sp tag
        self.dacs = self.dacs[self._trim_tags["sp"] :]
        # trim for start and end read trimming
        ns = self._trim_tags["ns"]
        if ns is None:
            ns = self.dacs.size
        self.dacs = self.dacs[self._trim_tags["ts"] : ns]
        if reverse_signal:
            self.dacs = self.dacs[::-1]

    def compute_ref_to_signal(self):
        if (
            not self.is_mapped
            or self.cigar is None
            or self.ref_seq is None
            or self.query_to_signal is None
            or not self._adjusted_query_to_signal
        ):
            return
        self.ref_to_signal = data_chunks.compute_ref_to_signal(
            query_to_signal=self.query_to_signal,
            cigar=self.cigar,
        )
        # +1 because knots include the end position of the last base
        if self.ref_to_signal.size != len(self.ref_seq) + 1:
            LOGGER.debug(
                f"{self.child_read_id} discordant ref seq lengths: "
                f"move+cigar:{self.ref_to_signal.size} "
                f"ref_seq:{len(self.ref_seq)}"
            )
            raise RemoraError("Discordant ref seq lengths")
        self.ref_reg.end = self.ref_reg.start + self.ref_to_signal.size - 1

    def add_signal_metrics(self):
        if self.read_metrics is None:
            self.read_metrics = {}
        if self.dacs is not None:
            for metric_name, metric in util.SIGNAL_METRICS.items():
                try:
                    self.read_metrics[metric_name] = metric.func(self)
                except Exception:
                    LOGGER.debug(
                        f"Parsing {metric_name} from {self.read_id} failed"
                    )
                self.read_metrics[metric_name] = metric.default

    def add_mapping_metrics(self, alignment_record):
        if self.read_metrics is None:
            self.read_metrics = {}
        for metric_name, metric in util.MAPPING_METRICS.items():
            try:
                self.read_metrics[metric_name] = metric.func(alignment_record)
            except Exception:
                LOGGER.debug(
                    f"Parsing {metric_name} from {self.read_id} failed"
                )
                self.read_metrics[metric_name] = metric.default

    def add_alignment(
        self,
        alignment_record,
        parse_ref_align=True,
        reverse_signal=False,
        pa_scaling=None,
    ):
        """Add alignment to read object

        Args:
            alignment_record (pysam.AlignedSegment)
            parse_ref_align (bool): Should reference alignment be parsed
            reverse_signal (bool): Does this read derive from 3' to 5' signal
                (RNA reads)
            pa_scaling (tuple): picoamp to zero-centered picoamp shift and scale
        """
        if pa_scaling is not None:
            self.shift_pa_to_zc_pa = pa_scaling[0]
            self.scale_pa_to_zc_pa = pa_scaling[1]
        if (
            alignment_record.reference_name is None
            and alignment_record.is_reverse
        ):
            raise RemoraError("Unmapped reads cannot map to reverse strand.")
        self.full_align = alignment_record.to_dict()

        tags = dict(alignment_record.tags)
        try:
            self._trim_tags = dict(
                (tag, tags.get(tag, dv))
                for tag, dv in (("sp", 0), ("ts", 0), ("ns", None))
            )
        except KeyError:
            pass
        self.trim_signal(reverse_signal)
        # update signal metrics and add mapping metrics
        self.add_signal_metrics()
        self.add_mapping_metrics(alignment_record)

        parent_read_id = tags.get("pi", None)
        if parent_read_id is None:
            if alignment_record.query_name != self.read_id:
                raise RemoraError("Read IDs mismatch")
        else:
            if parent_read_id != self.read_id:
                raise RemoraError("Split read IDs mismatch")
            self._child_read_id = alignment_record.query_name

        self.seq = alignment_record.query_sequence
        if alignment_record.is_reverse:
            self.seq = util.revcomp(self.seq)
        try:
            self.stride = tags["mv"][0]
            self.mv_table = np.array(tags["mv"][1:])
            self.query_to_signal = np.nonzero(self.mv_table)[0] * self.stride
            if self.dacs is not None:
                self.adjust_move_table(reverse_signal=reverse_signal)
        except KeyError:
            LOGGER.debug(f"Move table not found for {self.child_read_id}")
            self.query_to_signal = self.mv_table = self.stride = None

        try:
            self.shift_pa_to_norm = tags["sm"]
            self.scale_pa_to_norm = tags["sd"]
        except KeyError:
            if self.dacs is not None:
                self.compute_pa_to_norm_scaling()

        if (
            self.shift_pa_to_norm is not None
            and self.shift_dacs_to_pa is not None
        ):
            self.shift_dacs_to_norm = self.shift_dacs_to_pa + (
                self.scale_dacs_to_pa * self.shift_pa_to_norm
            )
            self.scale_dacs_to_norm = (
                self.scale_dacs_to_pa * self.scale_pa_to_norm
            )

        if not parse_ref_align or alignment_record.is_unmapped:
            self.is_mapped = False
            return
        self.is_mapped = True

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
        self.compute_ref_to_signal()

    def add_signal(
        self,
        pod5_read,
        reverse_signal=False,
    ):
        """Add signal to read object

        Args:
            pod5_read (pod5.ReadRecord)
            reverse_signal (bool): Does this read derive from 3' to 5' signal
                (RNA reads)
        """
        self.dacs = pod5_read.signal
        if reverse_signal:
            self.dacs = self.dacs[::-1]
        self.trim_signal(reverse_signal)
        self.adjust_move_table(reverse_signal=reverse_signal)
        if self.shift_pa_to_norm is None:
            self.compute_pa_to_norm_scaling()
        if (
            self.shift_pa_to_norm is not None
            and self.shift_dacs_to_pa is not None
        ):
            self.shift_dacs_to_norm = self.shift_dacs_to_pa + (
                self.scale_dacs_to_pa * self.shift_pa_to_norm
            )
            self.scale_dacs_to_norm = (
                self.scale_dacs_to_pa * self.scale_pa_to_norm
            )
        self.compute_ref_to_signal()

    @classmethod
    def from_pod5_and_alignment(
        cls,
        pod5_read_record,
        alignment_record,
        reverse_signal=False,
        pa_scaling=None,
    ):
        """Initialize read from pod5 and pysam records

        Args:
            pod5_read_record (pod5.ReadRecord)
            alignment_record (pysam.AlignedSegment)
            reverse_signal (bool): Is this 3' to 5' signal
            pa_scaling (tuple): picoamp to zero-centered picoamp shift and scale
        """
        dacs = pod5_read_record.signal
        if reverse_signal:
            dacs = dacs[::-1]
        # calibration offset and scale apply normalization as:
        #     y=(x+a)/b
        # while io.Read stores shfit and scale as:
        #     y=c*(x-d)
        # In other words, as shift=mean and scale=std. dev.
        read = Read(
            read_id=str(pod5_read_record.read_id),
            dacs=dacs,
            shift_dacs_to_pa=-pod5_read_record.calibration.offset,
            scale_dacs_to_pa=1 / pod5_read_record.calibration.scale,
        )
        read.add_alignment(
            alignment_record,
            reverse_signal=reverse_signal,
            pa_scaling=pa_scaling,
        )
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
                    raise RemoraError("Missing reference alignment")
                self.ref_to_signal = data_chunks.compute_ref_to_signal(
                    self.query_to_signal,
                    self.cigar,
                )
                if self.ref_to_signal.size != len(self.ref_seq) + 1:
                    LOGGER.debug(
                        f"{self.child_read_id} discordant ref seq lengths: "
                        f"move+cigar:{self.ref_to_signal.size - 1} "
                        f"ref_seq:{len(self.ref_seq)}"
                    )
                    raise RemoraError("Discordant ref seq lengths")

            trim_dacs = self.dacs[
                self.ref_to_signal[0] : self.ref_to_signal[-1]
            ]
            shift_seq_to_sig = self.ref_to_signal - self.ref_to_signal[0]
            seq = self.ref_seq
        else:
            if self.query_to_signal is None:
                raise RemoraError("Missing query_to_signal (move table)")
            trim_dacs = self.dacs[
                self.query_to_signal[0] : self.query_to_signal[-1]
            ]
            shift_seq_to_sig = self.query_to_signal - self.query_to_signal[0]
            seq = self.seq
        if self.shift_pa_to_zc_pa is None or self.scale_pa_to_zc_pa is None:
            scale_kwargs = {
                "shift": self.shift_dacs_to_norm,
                "scale": self.scale_dacs_to_norm,
            }
        else:
            scale_kwargs = {
                "shift": self.shift_dacs_to_zc_pa,
                "scale": self.scale_dacs_to_zc_pa,
            }
        remora_read = data_chunks.RemoraRead(
            dacs=trim_dacs,
            seq_to_sig_map=shift_seq_to_sig,
            str_seq=seq,
            read_id=self.read_id,
            read_metrics=self.read_metrics,
            **scale_kwargs,
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
            if self.ref_to_signal is None:
                raise RemoraError("Missing ref_to_signal (move table)")
            self.ref_to_signal = (
                remora_read.seq_to_sig_map + self.ref_to_signal[0]
            )
        else:
            if self.query_to_signal is None:
                raise RemoraError("Missing query_to_signal (move table)")
            self.query_to_signal = (
                remora_read.seq_to_sig_map + self.query_to_signal[0]
            )

        self.shift_dacs_to_norm = remora_read.shift
        self.scale_dacs_to_norm = remora_read.scale
        self.shift_pa_to_norm = (
            self.shift_dacs_to_norm - self.shift_dacs_to_pa
        ) / self.scale_dacs_to_pa
        self.scale_pa_to_norm = self.scale_dacs_to_norm / self.scale_dacs_to_pa

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
        mapping = data_chunks.make_sequence_coordinate_mapping(
            self.cigar
        ).astype(int)

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

    def get_sig_type(self, signal_type):
        """Get type of signal specified

        Args:
            signal_type (str): One of "norm" (default), "pa", "zc_pa", "dac"
        """
        if signal_type == "norm":
            return self.norm_signal
        elif signal_type == "pa":
            return self.pa_signal
        elif signal_type == "zc_pa":
            return self.zero_centered_pa_signal
        elif signal_type == "dac":
            return self.dacs
        raise RemoraError(f"Invalid signal_type: {signal_type}")

    def extract_basecall_region(
        self, start_base=None, end_base=None, signal_type="norm"
    ):
        """Extract region of read from basecall coordinates.

        Args:
            start_base (int): Start coordinate for region
            end_base (int): End coordinate for region
            signal_type (str): One of "norm" (default), "pa", "zc_pa", "dac"

        Returns:
            ReadBasecallRegion object
        """
        if self.query_to_signal is None:
            raise RemoraError("Missing query_to_signal (move table)")
        start_base = start_base or 0
        end_base = end_base or self.seq_len
        reg_seq_to_sig = self.query_to_signal[start_base : end_base + 1].copy()
        reg_sig = self.get_sig_type(signal_type)[
            reg_seq_to_sig[0] : reg_seq_to_sig[-1]
        ]
        sig_start = reg_seq_to_sig[0]
        reg_seq_to_sig -= sig_start
        return ReadBasecallRegion(
            read_id=self.read_id,
            norm_signal=reg_sig,
            seq=self.seq[start_base:end_base],
            seq_to_sig_map=reg_seq_to_sig,
            start=start_base,
            sig_start=sig_start,
        )

    def extract_ref_reg(self, ref_reg, signal_type="norm"):
        """Extract region of read from reference coordinates.

        Args:
            ref_reg (RefRegion): Reference region
            signal_type (str): One of "norm" (default), "pa", "zc_pa", "dac"

        Returns:
            ReadRefReg object
        """
        if self.ref_to_signal is None:
            raise RemoraError("Missing ref_to_signal (move table)")
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
        reg_sig = self.get_sig_type(signal_type)[
            reg_seq_to_sig[0] : reg_seq_to_sig[-1]
        ]
        reg_seq = self.ref_seq[reg_st_within_read:reg_en_within_read]
        sig_start = reg_seq_to_sig[0]
        reg_seq_to_sig -= sig_start
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
            sig_start=sig_start,
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
            metric_func (Callable): Function taking two arguments signal and
                a sequence to signal mapping and return a dict of metric names
                to per base metric arrays.
            ref_anchored (bool): Compute metric against reference bases. If
                False, return basecall anchored metrics.
            region (RefRegion): Reference region from which to extract metrics
                of bases. If ref_anchored is False, start and end coordinates
                are in basecall sequence coordinates.
            signal_type (str): One of "norm" (default), "pa", "zc_pa", and "dac"
            **kwargs: Extra args to pass through to metric computations

        Returns:
            Result of metric_func. Preset metrics will return dict with metric
            name keys and a numpy array of per-base metric values.
        """
        if metric is not None:
            metric_func = METRIC_FUNCS[metric]
        if metric_func is None:
            raise RemoraError("Must provide either metric or metric_func")
        st_buf = en_buf = 0
        if region is None:
            seq_to_sig = (
                self.ref_to_signal if ref_anchored else self.query_to_signal
            )
            if seq_to_sig is None:
                raise RemoraError("Missing move table")
        else:
            if ref_anchored:
                if self.ref_to_signal is None:
                    raise RemoraError("Missing ref_to_signal (move table)")
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
                if self.query_to_signal is None:
                    raise RemoraError("Missing query_to_signal (move table)")
                if region.start < 0 or region.start > self.seq_len:
                    raise RemoraError("Region does not overlap read.")
                # TODO deal with partially overlapping region
                st_buf = en_buf = 0
                seq_to_sig = self.query_to_signal[region.start : region.end]
        sig = self.get_sig_type(signal_type)
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
        self.reader = DatasetReader(Path(pod5_path))

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
