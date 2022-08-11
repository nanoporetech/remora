from copy import copy
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import pysam
import numpy as np
from tqdm import tqdm
from pod5_format import CombinedReader

from remora import log
from remora.util import revcomp

LOGGER = log.get_logger()

# Note sm and sd tags are not required, but highly recommended to pass
# basecaller scaling into remora
REQUIRED_TAGS = {"mv", "MD"}


def read_is_primary(read):
    return not (read.is_supplementary or read.is_secondary)


def index_bam(bam_fn, skip_non_primary, req_tags=REQUIRED_TAGS):
    bam_idx = {} if skip_non_primary else defaultdict(list)
    num_reads = 0
    # hid warnings for no index when using unmapped or unsorted files
    pysam_save = pysam.set_verbosity(0)
    with pysam.AlignmentFile(bam_fn, mode="rb", check_sq=False) as bam_fp:
        pbar = tqdm(
            smoothing=0,
            unit=" Reads",
            desc="Indexing BAM by read id",
        )
        while True:
            read_ptr = bam_fp.tell()
            try:
                read = next(bam_fp)
            except StopIteration:
                break
            tags = set(tg[0] for tg in read.tags)
            if len(req_tags.intersection(tags)) != len(req_tags):
                continue
            if skip_non_primary:
                if not read_is_primary(read) or read.query_name in bam_idx:
                    continue
                bam_idx[read.query_name] = [read_ptr]
            else:
                bam_idx[read.query_name].append(read_ptr)
            num_reads += 1
            pbar.update()
    pysam.set_verbosity(pysam_save)
    return dict(bam_idx), num_reads


@dataclass
class Read:
    read_id: str
    signal: np.array = None
    seq: str = None
    stride: int = None
    num_trimmed: int = None
    mv_table: np.array = None
    query_to_signal: np.array = None
    shift_dacs_to_pa: float = None
    scale_dacs_to_pa: float = None
    shift_pa_to_norm: float = None
    scale_pa_to_norm: float = None
    shift_dacs_to_norm: float = None
    scale_dacs_to_norm: float = None
    ref_seq: str = None
    cigar: list = None
    ref_to_signal: np.array = None
    full_align: str = None

    def compute_pa_to_norm_scaling(self, factor=1.4826):
        pa_signal = self.scale_dacs_to_pa * (
            self.signal + self.shift_dacs_to_pa
        )
        self.shift_pa_to_norm = np.median(pa_signal)
        self.scale_pa_to_norm = max(
            1.0, np.median(np.abs(pa_signal - self.shift_pa_to_norm)) * factor
        )


##########################
# Signal then alignments #
##########################


def iter_signal(pod5_fn, num_reads=None, read_ids=None):
    LOGGER.debug("Reading from POD5")
    with CombinedReader(Path(pod5_fn)) as pod5_fp:
        for read_num, read in enumerate(
            pod5_fp.reads(selection=read_ids, preload=["samples"])
        ):
            if num_reads is not None and read_num >= num_reads:
                return
            yield (
                Read(
                    str(read.read_id),
                    signal=read.signal,
                    shift_dacs_to_pa=read.calibration.offset,
                    scale_dacs_to_pa=read.calibration.scale,
                ),
                None,
            )
    LOGGER.debug("Completed signal worker")


def prep_extract_alignments(bam_idx, bam_fn, req_tags=REQUIRED_TAGS):
    pysam_save = pysam.set_verbosity(0)
    bam_fp = pysam.AlignmentFile(bam_fn, mode="rb", check_sq=False)
    pysam.set_verbosity(pysam_save)
    return [bam_idx, bam_fp], {"req_tags": req_tags}


def extract_alignments(read_err, bam_idx, bam_fp, req_tags=REQUIRED_TAGS):
    read, err = read_err
    if read is None:
        return [read_err]
    if read.read_id not in bam_idx:
        return [tuple((None, "Read id not found in BAM file"))]
    read_alignments = []
    for read_bam_ptr in bam_idx[read.read_id]:
        # jump to bam read pointer
        bam_fp.seek(read_bam_ptr)
        bam_read = next(bam_fp)
        tags = dict(bam_read.tags)
        if len(req_tags.intersection(tags)) != len(req_tags):
            continue
        mv_tag = tags["mv"]
        stride = mv_tag[0]
        mv_table = np.array(mv_tag[1:])
        try:
            read.num_trimmed = tags["ts"]
            read.signal = read.signal[read.num_trimmed :]
        except KeyError:
            read.num_trimmed = 0
        query_to_signal = np.nonzero(mv_table)[0] * stride
        query_to_signal = np.concatenate([query_to_signal, [read.signal.size]])
        if mv_table.size != read.signal.size // stride:
            read_alignments.append(
                tuple((None, "Move table discordant with signal"))
            )
        if query_to_signal.size - 1 != len(bam_read.query_sequence):
            read_alignments.append(
                tuple((None, "Move table discordant with basecalls"))
            )
        try:
            read.shift_pa_to_norm = tags["sm"]
            read.scale_pa_to_norm = tags["sd"]
        except KeyError:
            read.compute_pa_to_norm_scaling()
        read.shift_dacs_to_norm = (
            read.shift_pa_to_norm / read.scale_dacs_to_pa
        ) - read.shift_dacs_to_pa
        read.scale_dacs_to_norm = read.scale_pa_to_norm / read.scale_dacs_to_pa

        seq = bam_read.query_sequence
        try:
            ref_seq = bam_read.get_reference_sequence().upper()
        except ValueError:
            ref_seq = None
        cigar = bam_read.cigartuples
        if bam_read.is_reverse:
            seq = revcomp(seq)
            ref_seq = revcomp(ref_seq)
            cigar = cigar[::-1]

        align_read = copy(read)
        align_read.seq = seq
        align_read.stride = stride
        align_read.mv_table = mv_table
        align_read.query_to_signal = query_to_signal
        align_read.ref_seq = ref_seq
        align_read.cigar = cigar
        align_read.full_align = bam_read.to_dict()
        read_alignments.append(tuple((align_read, None)))
    return read_alignments


##########################
# Alignments then signal #
##########################


def iter_alignments(
    bam_fn, num_reads, skip_non_primary, req_tags=REQUIRED_TAGS
):
    pysam_save = pysam.set_verbosity(0)
    with pysam.AlignmentFile(bam_fn, mode="rb", check_sq=False) as bam_fp:
        for read_num, read in enumerate(bam_fp.fetch(until_eof=True)):
            if num_reads is not None and read_num > num_reads:
                return
            if skip_non_primary and not read_is_primary(read):
                continue
            tags = dict(read.tags)
            if len(req_tags.intersection(tags)) != len(req_tags):
                continue
            mv_tag = tags["mv"]
            stride = mv_tag[0]
            mv_table = np.array(mv_tag[1:])
            query_to_signal = np.nonzero(mv_table)[0] * stride
            if query_to_signal.size != len(read.query_sequence):
                yield None, "Move table discordant with basecalls"
            try:
                num_trimmed = tags["ts"]
            except KeyError:
                num_trimmed = 0
            ref_seq = read.get_reference_sequence().upper()
            cigar = read.cigartuples
            if read.is_reverse:
                ref_seq = revcomp(ref_seq)
                cigar = cigar[::-1]
            yield (
                Read(
                    read.query_name,
                    seq=read.query_sequence,
                    stride=stride,
                    mv_table=mv_table,
                    query_to_signal=query_to_signal,
                    num_trimmed=num_trimmed,
                    shift_pa_to_norm=tags.get("sm"),
                    scale_pa_to_norm=tags.get("sd"),
                    ref_seq=ref_seq,
                    cigar=cigar,
                    full_align=read.to_dict(),
                ),
                None,
            )
    pysam.set_verbosity(pysam_save)


def prep_extract_signal(pod5_fn):
    pod5_fp = CombinedReader(Path(pod5_fn))
    return [
        pod5_fp,
    ], {}


def extract_signal(read_err, pod5_fp):
    read, err = read_err
    if read is None:
        return [read_err]
    pod5_read = next(pod5_fp.reads([read.read_id]))
    read.signal = pod5_read.signal[read.num_trimmed :]
    read.query_to_signal = np.concatenate(
        [read.query_to_signal, [read.signal.size]]
    )
    if read.mv_table.size != read.signal.size // read.stride:
        return [tuple((None, "Move table discordant with signal"))]
    read.shift_dacs_to_pa = pod5_read.calibration.offset
    read.scale_dacs_to_pa = pod5_read.calibration.scale
    if read.shift_pa_to_norm is None or read.scale_pa_to_norm is None:
        read.compute_pa_to_norm_scaling()
    read.shift_dacs_to_norm = (
        read.shift_pa_to_norm / read.scale_dacs_to_pa
    ) - read.shift_dacs_to_pa
    read.scale_dacs_to_norm = read.scale_pa_to_norm / read.scale_dacs_to_pa
    return [tuple((read, None))]
