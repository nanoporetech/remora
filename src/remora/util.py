import os
import re
import toml
import array
import queue
import platform
import traceback
from abc import ABC
from time import sleep
from pathlib import Path
from shutil import rmtree
import multiprocessing as mp
from threading import Thread
from itertools import product
from datetime import datetime
from dataclasses import dataclass
from collections import namedtuple
from os.path import realpath, expanduser

import torch
import numpy as np

from remora import log, RemoraError

LOGGER = log.get_logger()

CAN_ALPHABET = "ACGT"
CONV_ALPHABET = "ACGTN"
SINGLE_LETTER_CODE = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "K": "GT",
    "M": "AC",
    "N": "ACGT",
    "R": "AG",
    "S": "CG",
    "V": "ACG",
    "W": "AT",
    "Y": "CT",
}
BASES_TO_CODES = dict((v, k) for k, v in SINGLE_LETTER_CODE.items())
SEQ_MIN = np.array(["A"], dtype="S1").view(np.uint8)[0]
SEQ_TO_INT_ARR = np.full(26, -1, dtype=int)
SEQ_TO_INT_ARR[0] = 0
SEQ_TO_INT_ARR[2] = 1
SEQ_TO_INT_ARR[6] = 2
SEQ_TO_INT_ARR[19] = 3
COMP_BASES = dict(zip(map(ord, "ACGTBVDHKMRY"), map(ord, "TGCAVBHDMKYR")))
NP_COMP_BASES = np.array([3, 2, 1, 0], dtype=np.uintp)
U_TO_T_BASES = {ord("U"): ord("T")}
T_TO_U_BASES = {ord("T"): ord("U")}

DEFAULT_QUEUE_SIZE = 10_000


def str_to_bool(value):
    truthy_values = {"true", "yes", "1", "t", "y"}
    falsy_values = {"false", "no", "0", "f", "n"}

    value_lower = value.strip().lower()
    if value_lower in truthy_values:
        return True
    elif value_lower in falsy_values:
        return False
    else:
        raise ValueError(f"Cannot convert {value} to boolean")


def prepare_out_dir(out_dir, overwrite):
    out_path = Path(out_dir)
    if overwrite:
        if out_path.is_dir():
            rmtree(out_path)
        elif out_path.exists():
            out_path.unlink()
    elif out_path.exists():
        raise RemoraError("Refusing to overwrite existing directory.")
    out_path.mkdir(parents=True, exist_ok=True)
    log.init_logger(os.path.join(out_path, "log.txt"))


def human_format(num):
    num = float("{:.3g}".format(num))
    mag = 0
    while num >= 1000:
        mag += 1
        num /= 1000.0
    return num, ["", "K", "M", "B", "T"][mag]


def parse_device(device):
    if device is None:
        return None
    # convert int devices for legacy settings
    try:
        device = int(device)
    except (ValueError, TypeError):
        pass
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        LOGGER.error(
            "Device option specified, but CUDA not available.\nEnsure torch "
            "installation is compatible with CUDA driver "
            "(https://pytorch.org/get-started/locally/)."
        )
        raise RemoraError("Invalid GPU/CUDA device")
    return device


def comp(seq):
    """Convert seq to its complement sequence.
    Handles IUPAC ambiguous bases.
    """
    return seq.translate(COMP_BASES)


def revcomp(seq):
    """Convert seq to its complement sequence and reverse the sequence.
    Handles IUPAC ambiguous bases.
    """
    return seq.upper().translate(COMP_BASES)[::-1]


def comp_np(np_seq):
    """Convert numpy seq to its complement sequence.
    Does NOT handles IUPAC ambiguous bases.
    """
    return NP_COMP_BASES[np_seq]


def revcomp_np(np_seq):
    """Convert numpy seq to its complement sequence and reverse the sequence.
    Does NOT handles IUPAC ambiguous bases.
    """
    return NP_COMP_BASES[np_seq][::-1]


def u_to_t(seq):
    return seq.translate(U_TO_T_BASES)


def t_to_u(seq):
    return seq.translate(T_TO_U_BASES)


def seq_to_int(seq):
    """Convert string sequence to integer encoded array

    Args:
        seq (str): Nucleotide sequence

    Returns:
        np.array containing integer encoded sequence
    """
    return SEQ_TO_INT_ARR[
        np.array(list(seq), dtype="c").view(np.uint8) - SEQ_MIN
    ]


def int_to_seq(np_seq, alphabet=CONV_ALPHABET):
    """Convert integer encoded array to string sequence

    Args:
        np_seq (np.array): integer encoded sequence

    Returns:
        String nucleotide sequence
    """
    if np_seq.shape[0] == 0:
        return ""
    if np_seq.max() >= len(alphabet):
        raise RemoraError(f"Invalid value in int sequence ({np_seq.max()})")
    return "".join(alphabet[b] for b in np_seq)


def resolve_path(fn_path):
    """Helper function to resolve relative and linked paths that might
    give other packages problems.
    """
    if fn_path is None:
        return None
    return realpath(expanduser(fn_path))


def to_str(value):
    """Try to convert a bytes object to a string. If it is already a string
    catch this error and return the input string. This can be used for read ids
    stored in HDF5 files as they are sometimes returned as bytes and sometimes
    strings.
    """
    try:
        return value.decode()
    except AttributeError:
        return str(value)


def softmax_axis1(x):
    """Compute softmax over axis=1"""
    e_x = np.exp((x.T - np.max(x, axis=1)).T)
    with np.errstate(divide="ignore"):
        return (e_x.T / e_x.sum(axis=1)).T


@dataclass
class Motif:
    """Sequence motif including ambiguous bases along with the focus position
    within the motif.

    Args:
        raw_motif (str): Sequence motif. Use IUPAC single letter base codes for
            ambiguous bases. Variable length motifs are not implemented.
        focus_base (int): 0-based index of the focus position within the motif
    """

    raw_motif: str
    focus_pos: int = 0

    def __post_init__(self):
        try:
            self.focus_pos = int(self.focus_pos)
        except ValueError:
            raise RemoraError(
                f'Motif focus position not an integer: "{self.focus_pos}"'
            )
        if not isinstance(self.raw_motif, str):
            raise RemoraError("Motif sequence must be a string")
        invalid_bases = set(self.raw_motif).difference(SINGLE_LETTER_CODE)
        if len(invalid_bases) > 0:
            raise RemoraError(
                f"Motif contains invalid characters: {invalid_bases}"
            )
        if self.focus_pos >= len(self.raw_motif):
            raise RemoraError(
                "Motif focus position is past the end of the motif"
            )
        # clip Ns from end of new motif
        while len(self.raw_motif) > 1 and self.raw_motif[0] == "N":
            self.raw_motif = self.raw_motif[1:]
            self.focus_pos -= 1
        while len(self.raw_motif) > 1 and self.raw_motif[-1] == "N":
            self.raw_motif = self.raw_motif[:-1]

    def to_tuple(self):
        """Return motif as tuple of raw motif and focus position"""
        return self.raw_motif, self.focus_pos

    def __hash__(self):
        return hash(self.to_tuple())

    @property
    def focus_base(self):
        """Canonical base at the focus position of the motif"""
        return self.raw_motif[self.focus_pos]

    @property
    def num_bases_after_focus(self):
        """Number of bases in the motif after the focus position."""
        return len(self.raw_motif) - self.focus_pos - 1

    @property
    def pattern(self):
        """Python regex for matching a string with the motif"""
        ambig_pat_str = "".join(
            "[{}]".format(SINGLE_LETTER_CODE[letter])
            for letter in self.raw_motif
        )
        # add lookahead group to serach for overlapping motif hits
        return re.compile("(?=({}))".format(ambig_pat_str))

    @property
    def int_pattern(self):
        """Integer encoded bases (A=0, C=1, G=2, T=3) allowed at each position
        within the motif. Stored as a list of numpy arrays.
        """
        return [
            np.array(
                [
                    b_idx
                    for b_idx, b in enumerate(CAN_ALPHABET)
                    if b in SINGLE_LETTER_CODE[letter]
                ]
            )
            for letter in self.raw_motif
        ]

    @property
    def possible_kmers(self):
        """List of all possible k-mers encoded by the motif"""
        return [
            "".join(bs)
            for bs in product(
                *[SINGLE_LETTER_CODE[letter] for letter in self.raw_motif]
            )
        ]

    def findall(self, int_seq):
        """Return numpy array with index of focus position for each hit of
        motif to input integer encoded sequence.
        """
        return np.where(
            np.logical_and.reduce(
                [
                    np.isin(
                        int_seq[
                            po : int_seq.size - len(self.int_pattern) + po + 1
                        ],
                        pi,
                    )
                    for po, pi in enumerate(self.int_pattern)
                ]
            )
        )[0]

    def match(self, int_seq, pos):
        """Test whether the motif matches the motif at the specified position"""
        pat_st = pos - self.focus_pos
        pat_en = pos + self.num_bases_after_focus + 1
        # clip pattern if motif extends beyond read
        int_pat = self.int_pattern
        if pat_st < 0:
            int_pat = int_pat[-pat_st:]
            pat_st = 0
        if pat_en > int_seq.size:
            int_pat = int_pat[: len(int_pat) - pat_en + int_seq.size]
            pat_en = int_seq.size
        return all(
            np.isin(idx_base, idx_pat)
            for idx_pat, idx_base in zip(int_pat, int_seq[pat_st:pat_en])
        )

    def is_super_set(self, other):
        """Determine if this motif is a super-set of another motif. In other
        words, are all sequences represented by the other motif found within
        this motif?
        """
        # first ensure that this motif is equal or shorted than the other motif
        if (
            self.focus_pos > other.focus_pos
            or self.num_bases_after_focus > other.num_bases_after_focus
        ):
            return False
        trim_other_raw_motif = other.raw_motif[
            other.focus_pos
            - self.focus_pos : other.focus_pos
            + self.num_bases_after_focus
            + 1
        ]
        for self_base, other_base in zip(self.raw_motif, trim_other_raw_motif):
            if any(
                ob not in SINGLE_LETTER_CODE[self_base]
                for ob in SINGLE_LETTER_CODE[other_base]
            ):
                return False
        return True

    def merge(self, other):
        """Merge this motif with another motif"""
        if self == other:
            return self
        elif self.is_super_set(other):
            return self
        elif other.is_super_set(self):
            return other
        elif len(self.raw_motif) != len(other.raw_motif):
            raise RemoraError("Cannot merge motifs of different sizes")
        elif self.focus_pos != other.focus_pos:
            raise RemoraError("Cannot merge motifs with different focus pos")
        # attempt to join motifs
        all_kmers = set(self.possible_kmers).union(other.possible_kmers)
        new_motif = Motif(
            "".join(
                BASES_TO_CODES[
                    "".join(sorted(set(kmer[kmer_idx] for kmer in all_kmers)))
                ]
                for kmer_idx in range(len(self.raw_motif))
            ),
            self.focus_pos,
        )
        # if new motif is trimmed with N positions
        if len(new_motif.raw_motif) < len(self.raw_motif):
            st = self.focus_pos - new_motif.focus_pos
            en = len(self.raw_motif) - len(new_motif.raw_motif) - st
            pos_bases = (
                (["ACGT"] * st)
                + [SINGLE_LETTER_CODE[letter] for letter in new_motif.raw_motif]
                + (["ACGT"] * en)
            )
            new_poss_kmers = set(["".join(bs) for bs in product(*pos_bases)])
        else:
            new_poss_kmers = set(new_motif.possible_kmers)
        if all_kmers != new_poss_kmers:
            raise RemoraError("Cannot merge motifs {self} {other}")
        return new_motif


def merge_motifs(motifs):
    """Merge list of motif objects via pairwise merge attempts"""
    # convert to tuples to motifs
    motifs = [
        motif if isinstance(motif, Motif) else (Motif(*motif))
        for motif in motifs
    ]
    prev_motifs = None
    # ensure motifs are unique
    motifs = list(set(motifs))
    while len(motifs) > 1 and (
        prev_motifs is None or set(prev_motifs) != set(motifs)
    ):
        prev_motifs = motifs
        merged_motifs = set()
        motifs = set()
        for motif_a in prev_motifs:
            for motif_b in prev_motifs[1:]:
                try:
                    m_motif = motif_a.merge(motif_b)
                    if m_motif != motif_a:
                        merged_motifs.add(motif_a)
                    if m_motif != motif_b:
                        merged_motifs.add(motif_b)
                    motifs.add(m_motif)
                except RemoraError:
                    # could not merge so add both motifs
                    motifs.update((motif_a, motif_b))
        motifs = list(motifs.difference(merged_motifs))
    return motifs


def find_focus_bases_in_int_sequence(
    int_seq: np.ndarray, motifs: list
) -> np.ndarray:
    """Return numpy array from with position of hit of any motif within the
    input sequence.
    """
    return np.fromiter(
        set(
            mot_pos + mot.focus_pos
            for mot in motifs
            for mot_pos in mot.findall(int_seq)
        ),
        int,
    )


def get_can_converter(alphabet, collapse_alphabet):
    """Compute conversion from full alphabet integer encodings to
    canonical alphabet integer encodings.
    """
    can_bases = "".join(
        (
            can_base
            for mod_base, can_base in zip(alphabet, collapse_alphabet)
            if mod_base == can_base
        )
    )
    return np.array(
        [can_bases.find(b) for b in collapse_alphabet], dtype=np.byte
    )


def get_mod_bases(alphabet, collapse_alphabet):
    return [
        mod_base
        for mod_base, can_base in zip(alphabet, collapse_alphabet)
        if mod_base != can_base
    ]


def validate_mod_bases(
    mod_bases, motifs, alphabet, collapse_alphabet, control=False
):
    """Validate that inputs are mutually consistent. Return label conversion
    from alphabet integer encodings to modified base categories.
    """
    if len(set(mod_bases)) < len(mod_bases):
        raise RemoraError("Single letter modified base codes must be unique.")
    can_base = motifs[0].focus_base
    if any(mot.focus_base != can_base for mot in motifs):
        raise RemoraError(
            "All motifs must be alternatives to the same canonical base"
        )
    can_base_idx = alphabet.find(can_base)
    label_conv = np.full(len(alphabet), -1, dtype=np.byte)
    label_conv[can_base_idx] = 0
    if control:
        return label_conv
    for mod_base in mod_bases:
        if mod_base not in alphabet:
            raise RemoraError("Modified base provided not found in alphabet")
        mod_can_equiv = collapse_alphabet[alphabet.find(mod_base)]
        if mod_can_equiv != can_base:
            raise RemoraError(
                f"Canonical base within motif ({can_base}) does not match "
                f"canonical equivalent for modified base ({mod_can_equiv})"
            )
    for mod_i, mod_base in enumerate(mod_bases):
        label_conv[alphabet.find(mod_base)] = mod_i + 1
    return label_conv


def format_mm_ml_tags(seq, poss, probs, mod_bases, can_base, strand: str = "+"):
    """Format MM and ML tags for BAM output. See
    https://samtools.github.io/hts-specs/SAMtags.pdf for format details.

    Args:
        seq (str): read-centric read sequence. For reference-anchored calls
            this should be the reverse complement sequence.
        poss (list): positions relative to seq
        probs (np.array): probabilities for modified bases
        mod_bases (list): modified base single letter codes
        can_base (str): canonical base
        strand (bool): should be '+' for SEQ-oriented strand and '-' if
            complement strand

    Returns:
        MM string tag and ML array tag
    """

    # initialize dict with all called mods to make sure all called mods are
    # shown in resulting tags
    per_mod_probs = dict((mod_base, []) for mod_base in mod_bases)
    for pos, mod_probs in sorted(zip(poss, probs)):
        # mod_probs is set to None if invalid sequence is encountered or too
        # few events are found around a mod
        if mod_probs is None:
            continue
        for mod_prob, mod_base in zip(mod_probs, mod_bases):
            per_mod_probs[mod_base].append((pos, mod_prob))

    mm_tag, ml_tag = "", array.array("B")
    for mod_base, pos_probs in per_mod_probs.items():
        if len(pos_probs) == 0:
            continue
        mod_poss, probs = zip(*sorted(pos_probs))
        # compute modified base positions relative to the running total of the
        # associated canonical base
        can_base_mod_poss = (
            np.cumsum([1 if b == can_base else 0 for b in seq])[
                np.array(mod_poss)
            ]
            - 1
        )
        mod_gaps = ",".join(
            map(str, np.diff(np.insert(can_base_mod_poss, 0, -1)) - 1)
        )
        mm_tag += f"{can_base}{strand}{mod_base}?,{mod_gaps};"
        # extract mod scores and scale to 0-255 range
        scaled_probs = np.floor(np.array(probs) * 256)
        # last interval includes prob=1
        scaled_probs[scaled_probs == 256] = 255
        ml_tag.extend(scaled_probs.astype(np.uint8))

    return mm_tag, ml_tag


def parse_picoamps(bc_model, sig_map_refiner):
    if bc_model is None:
        return
    if sig_map_refiner.do_rough_rescale or sig_map_refiner.scale_iters > -1:
        raise RemoraError(
            "Cannot specify signal scaling/mapping refinement and "
            "picoamp scaling options."
        )
    bc_cfg = Path(bc_model) / "config.toml"
    if not bc_cfg.exists():
        raise RemoraError(f"Basecalling model config does not exist: {bc_cfg}")
    cfg = toml.load(bc_cfg)
    try:
        std_cfg = cfg["standardisation"]
        do_std = std_cfg["standardise"]
        pa_scaling = (std_cfg["mean"], std_cfg["stdev"])
    except KeyError:
        raise RemoraError("Basecalling model is not picoamp scaling model")
    if do_std != 1:
        raise RemoraError("Basecalling model is not picoamp scaling model")
    return pa_scaling


def profile(prof_path):
    if prof_path is not None:
        import cProfile

    def inner(func):
        def wrapper(*args, **kwargs):
            if prof_path is None:
                return func(*args, **kwargs)
            prof = cProfile.Profile()
            retval = prof.runcall(func, *args, **kwargs)
            prof.dump_stats(prof_path)
            return retval

        return wrapper

    return inner


################
# Read Metrics #
################


REF_DT = datetime(2000, 1, 1)

READ_METRIC = namedtuple("READ_METRIC", ("func", "default"))


def compute_duration(io_read):
    return io_read.dacs.size


SIGNAL_METRICS = {"duration": READ_METRIC(compute_duration, 0)}


def compute_percent_identity(bam_read):
    """Compute percent identity, defined as edit distance over total alignment
    length.
    """
    M, I, D, N, S, H, P, E, X, B, NM = bam_read.get_cigar_stats()[0]
    num_align = M + E + X + I + D
    if num_align == 0:
        return 0
    return 100.0 * ((num_align - NM) / num_align)


def compute_start_time(bam_read):
    try:
        st = bam_read.get_tag("st")
    except KeyError:
        return np.iinfo(np.uint32).max

    if "+" in st:
        st = st.split("+")[0]
    elif "-" in st and st.count("-") > 2:
        st = "-".join(st.split("-")[:3]) + "T" + st.split("T")[1]
    try:
        st = datetime.fromisoformat(st)
    except ValueError:
        # Handle cases with fractional seconds by padding zeros
        st_parts = st.split(".")
        if len(st_parts) < 2:
            return np.iinfo(np.uint32).max
        frac_sec = st_parts[1]
        while len(frac_sec) < 6:
            frac_sec += "0"
        st = st_parts[0] + "." + frac_sec
        st = datetime.fromisoformat(st)
    return int((st - REF_DT).total_seconds())


MAPPING_METRICS = {
    "percent_identity": READ_METRIC(compute_percent_identity, 0),
    "start_time": READ_METRIC(compute_start_time, np.iinfo(np.uint32).max),
}

READ_METRICS = {**SIGNAL_METRICS, **MAPPING_METRICS}


###################
# Multiprocessing #
###################


class AbstractNamedQueue(ABC):
    def put(self, *args, **kwargs):
        self.queue.put(*args, **kwargs)

    def get(self, *args, **kwargs):
        rval = self.queue.get(*args, **kwargs)
        return rval


class NamedMPQueue(AbstractNamedQueue):
    def __init__(self, *args, **kwargs):
        self.maxsize = kwargs.get("maxsize")
        self.name = kwargs.get("name")
        self.queue = mp.Queue(maxsize=self.maxsize)
        self.size = mp.Value("i", 0)

    def put(self, *args, **kwargs):
        self.queue.put(*args, **kwargs)
        with self.size.get_lock():
            self.size.value += 1

    def get(self, *args, **kwargs):
        rval = self.queue.get(*args, **kwargs)
        with self.size.get_lock():
            self.size.value -= 1
        return rval

    def qsize(self):
        return self.size.value


class NamedQueue(AbstractNamedQueue):
    def __init__(self, **kwargs):
        self.maxsize = kwargs.get("maxsize")
        self.name = kwargs.get("name")
        self.queue = queue.Queue(maxsize=self.maxsize)

    def qsize(self):
        return self.queue.qsize()


def _put_item(item, out_q):
    """Put item into queue with timeout to handle KeyboardInterrupt"""
    while True:
        try:
            return out_q.put(item, timeout=0.1)
        except queue.Full:
            continue


def _get_item(in_q):
    """Get item from queue with timeout to handle KeyboardInterrupt"""
    while True:
        try:
            return in_q.get(timeout=0.1)
        except queue.Empty:
            continue


def _queue_iter(in_q, num_proc=1):
    comp_proc = 0
    while comp_proc < num_proc:
        item = _get_item(in_q)
        if item is StopIteration:
            comp_proc += 1
        else:
            yield item


def _fill_q(iterator, in_q, num_recievers):
    try:
        for item in iterator:
            _put_item(item, in_q)
    except KeyboardInterrupt:
        pass
    for _ in range(num_recievers):
        _put_item(StopIteration, in_q)


def _mt_func(func, in_q, out_q, prep_func, name, *args, **kwargs):
    LOGGER.debug(f"Starting {name} worker")
    try:
        if prep_func is not None:
            args, kwargs = prep_func(*args, **kwargs)
        for val in _queue_iter(in_q):
            try:
                out_q.put(func(val, *args, **kwargs))
            except Exception as e:
                # avoid killing thread for a python error.
                LOGGER.debug(
                    f"UNEXPECTED_ERROR in {name} worker: '{e}'.\n"
                    f"Full traceback: {traceback.format_exc()}"
                )
            except KeyboardInterrupt:
                LOGGER.debug(f"stopping {name} due to user interrupt")
                break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        LOGGER.debug(
            f"UNEXPECTED_ERROR in {name} worker: '{e}'.\n"
            f"Full traceback: {traceback.format_exc()}"
        )
    LOGGER.debug(f"Completed {name} worker")
    out_q.put(StopIteration)


def _background_filler(func, args, kwargs, out_q, name):
    LOGGER.debug(f"Starting {name} background filler")
    try:
        for item in func(*args, **kwargs):
            _put_item(item, out_q)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        LOGGER.debug(
            f"UNEXPECTED_ERROR in {name} worker: '{e}'.\n"
            f"Full traceback: {traceback.format_exc()}"
        )
    LOGGER.debug(f"Completed {name} background filler")
    out_q.put(StopIteration)


class MultitaskMap:
    """Map a function to an input queue using multitasking (multiprocessing or
    threading)

    MultitaskMap supports multiprocessing or threading backends via the
    use_process argument.

    Elements of the input queue will be passed as the first argument to the
    func followed by args and kwargs provided.

    MultitaskMap also supports a prepare function to perform work on the input
    arguments within the newly spawned task. The prep_func should take the args
    and kwargs provided to MultitaskMap and return a new set of args and kwargs
    to be passed to the worker function along with elements from the in_q. This
    can be useful for objects that need initialization within a task.

    MultitaskMap supports KeyboardInterrupt without flooding the output with
    stack traces from each killed task to exit gracefully and avoid stalling.
    """

    def __init__(
        self,
        func,
        iterator,
        prep_func=None,
        num_workers=1,
        q_maxsize=DEFAULT_QUEUE_SIZE,
        use_process=False,
        use_mp_queue=True,
        args=(),
        kwargs={},
        name="MultitaskMap",
    ):
        self.name = name
        self.num_workers = num_workers
        q_cls = NamedMPQueue if use_mp_queue else NamedQueue
        self.out_q = q_cls(maxsize=q_maxsize, name=f"{name}.out")
        in_q = q_cls(maxsize=q_maxsize, name=f"{name}.in")

        mt_worker = mp.Process if use_process else Thread
        # TODO save workers to self and provide method to watch workers
        #   for failures to avoid deadlock
        mt_worker(
            target=_fill_q,
            args=(iterator, in_q, self.num_workers),
            name=f"{self.name}_filler",
            daemon=True,
        ).start()
        args = [func, in_q, self.out_q, prep_func, self.name] + list(args)
        for idx in range(self.num_workers):
            mt_worker(
                target=_mt_func,
                args=args,
                kwargs=kwargs,
                name=f"{self.name}_{idx}",
                daemon=True,
            ).start()
        # processes take a second to start up on mac
        if platform.system() == "Darwin":
            wait_time = os.environ.get("MP_WAIT_TIME", 1)
            LOGGER.debug(
                f"MacOS requires that we wait, set MP_WAIT_TIME to modulate, "
                f"waiting {wait_time}s before starting {self.name}"
            )
            sleep(int(wait_time))

    def __iter__(self):
        try:
            yield from _queue_iter(self.out_q, self.num_workers)
        except KeyboardInterrupt:
            LOGGER.debug(f"MultitaskMap {self.name} interrupted")
            pass


class BackgroundIter:
    def __init__(
        self,
        func,
        q_maxsize=DEFAULT_QUEUE_SIZE,
        use_process=False,
        use_mp_queue=True,
        args=(),
        kwargs={},
        name="BackgroundIter",
    ):
        self.name = name
        q_cls = NamedMPQueue if use_mp_queue else NamedQueue
        self.out_q = q_cls(maxsize=q_maxsize, name=f"{name}.out")

        mt_worker = mp.Process if use_process else Thread
        mt_worker(
            target=_background_filler,
            args=(func, args, kwargs, self.out_q, self.name),
            name=f"{self.name}_filler",
            daemon=True,
        ).start()
        # processes take a second to start up on mac
        if platform.system() == "Darwin":
            wait_time = os.environ.get("MP_WAIT_TIME", 1)
            LOGGER.debug(
                f"MacOS requires that we wait, set MP_WAIT_TIME to modulate, "
                f"waiting {wait_time}s before starting {self.name}"
            )
            sleep(int(wait_time))

    def __iter__(self):
        try:
            yield from _queue_iter(self.out_q)
        except KeyboardInterrupt:
            LOGGER.debug(f"BackgroundIter {self.name} interrupted")
            pass


if __name__ == "__main__":
    RuntimeError("This is a module.")
