import os
import re
import array
import queue
import platform
import traceback
from time import sleep
import multiprocessing as mp
from threading import Thread
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
SEQ_MIN = np.array(["A"], dtype="S1").view(np.uint8)[0]
SEQ_TO_INT_ARR = np.full(26, -1, dtype=int)
SEQ_TO_INT_ARR[0] = 0
SEQ_TO_INT_ARR[2] = 1
SEQ_TO_INT_ARR[6] = 2
SEQ_TO_INT_ARR[19] = 3
COMP_BASES = dict(zip(map(ord, "ACGT"), map(ord, "TGCA")))
NP_COMP_BASES = np.array([3, 2, 1, 0], dtype=np.uintp)
U_TO_T_BASES = {ord("U"): ord("T")}
T_TO_U_BASES = {ord("T"): ord("U")}

DEFAULT_QUEUE_SIZE = 10_000


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
        LOGGER.warning("Device option specified, but CUDA not available.")
    return device


def iter_motif_hits(int_seq, motif):
    yield from np.where(
        np.logical_and.reduce(
            [
                np.isin(
                    int_seq[
                        po : int_seq.size - len(motif.int_pattern) + po + 1
                    ],
                    pi,
                )
                for po, pi in enumerate(motif.int_pattern)
            ]
        )
    )[0]


def find_focus_bases_in_int_sequence(
    int_seq: np.ndarray, motifs: list
) -> np.ndarray:
    return np.fromiter(
        set(
            mot_pos + mot.focus_pos
            for mot in motifs
            for mot_pos in iter_motif_hits(int_seq, mot)
        ),
        int,
    )


def comp(seq):
    return seq.translate(COMP_BASES)


def revcomp(seq):
    return seq.upper().translate(COMP_BASES)[::-1]


def comp_np(np_seq):
    return NP_COMP_BASES[np_seq]


def revcomp_np(np_seq):
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
        return value


def softmax_axis1(x):
    """Compute softmax over axis=1"""
    e_x = np.exp((x.T - np.max(x, axis=1)).T)
    with np.errstate(divide="ignore"):
        return (e_x.T / e_x.sum(axis=1)).T


def get_read_ids(bam_idx, pod5_fh, num_reads):
    """Get overlapping read ids from bam index and pod5 file

    Args:
        bam_idx (ReadIndexedBam): Read indexed BAM
        pod5_fh (pod5.Reader): POD5 file handle
        num_reads (int): Maximum number of reads, or None for no max
    """
    LOGGER.info("Extracting read IDs from POD5")
    pod5_read_ids = set(pod5_fh.read_ids)
    num_pod5_reads = len(pod5_read_ids)
    # pod5 will raise when it cannot find a "selected" read id, so we make
    # sure they're all present before starting
    # todo(arand) this could be performed using the read_table instead, but
    #  it's worth checking that it's actually faster and doesn't explode
    #  memory before switching from a sweep throug the pod5 file
    both_read_ids = list(pod5_read_ids.intersection(bam_idx.read_ids))
    num_both_read_ids = len(both_read_ids)
    LOGGER.info(
        f"Found {bam_idx.num_reads} BAM records, {num_pod5_reads} "
        f"POD5 reads, and {num_both_read_ids} in common"
    )
    if num_reads is None:
        num_reads = num_both_read_ids
    else:
        num_reads = min(num_reads, num_both_read_ids)
    return both_read_ids, num_reads


class Motif:
    def __init__(self, raw_motif, focus_pos=0):
        self.raw_motif = raw_motif
        try:
            self.focus_pos = int(focus_pos)
        except ValueError:
            raise RemoraError(
                f'Motif focus position not an integer: "{focus_pos}"'
            )
        if self.focus_pos >= len(self.raw_motif):
            raise RemoraError(
                "Motif focus position is past the end of the motif"
            )

        ambig_pat_str = "".join(
            "[{}]".format(SINGLE_LETTER_CODE[letter]) for letter in raw_motif
        )
        # add lookahead group to serach for overlapping motif hits
        self.pattern = re.compile("(?=({}))".format(ambig_pat_str))

        self.int_pattern = [
            np.array(
                [
                    b_idx
                    for b_idx, b in enumerate(CAN_ALPHABET)
                    if b in SINGLE_LETTER_CODE[letter]
                ]
            )
            for letter in raw_motif
        ]

    def to_tuple(self):
        return self.raw_motif, self.focus_pos

    @property
    def focus_base(self):
        return self.raw_motif[self.focus_pos]

    @property
    def any_context(self):
        return self.raw_motif == "N"

    @property
    def num_bases_after_focus(self):
        return len(self.raw_motif) - self.focus_pos - 1


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
        mod_bases (str): modified base single letter codes
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


###################
# Multiprocessing #
###################


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
        args=(),
        kwargs={},
        name="MultitaskMap",
    ):
        self.name = name
        self.num_workers = num_workers
        self.out_q = mp.Queue(q_maxsize)
        in_q = mp.Queue(q_maxsize)

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
            wait_time = os.environ.get("MP_WAIT_TIME")
            if wait_time is None:
                wait_time = 1
            else:
                try:
                    wait_time = int(wait_time)
                except ValueError as e:
                    raise ValueError(
                        f"failed to interpret MP_WAIT_TIME {wait_time}"
                    ) from e
            LOGGER.debug(
                f"MacOS requires that we wait, set MP_WAIT_TIME to modulate, "
                f"waiting {wait_time}s before starting {self.name}"
            )
            sleep(wait_time)

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
        args=(),
        kwargs={},
        name="BackgroundIter",
    ):
        self.name = name
        self.out_q = mp.Queue(q_maxsize)

        mt_worker = mp.Process if use_process else Thread
        mt_worker(
            target=_background_filler,
            args=(func, args, kwargs, self.out_q, self.name),
            name=f"{self.name}_filler",
            daemon=True,
        ).start()
        # processes take a second to start up on mac
        if platform.system() == "Darwin":
            sleep(1)

    def __iter__(self):
        try:
            yield from _queue_iter(self.out_q)
        except KeyboardInterrupt:
            LOGGER.debug(f"BackgroundIter {self.name} interrupted")
            pass


if __name__ == "__main__":
    RuntimeError("This is a module.")
