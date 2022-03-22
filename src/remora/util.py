import os
import re
import queue
import array
from os.path import realpath, expanduser

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
SEQ_TO_INT_ARR = np.full(26, -1, dtype=np.int)
SEQ_TO_INT_ARR[0] = 0
SEQ_TO_INT_ARR[2] = 1
SEQ_TO_INT_ARR[6] = 2
SEQ_TO_INT_ARR[19] = 3


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


class plotter:
    def __init__(self, outdir):
        self.outdir = outdir
        self.losses = []
        self.accuracy = []

    def append_result(self, accuracy, loss):
        self.losses.append(loss)
        self.accuracy.append(accuracy)

    def save_plots(self):
        import matplotlib.pyplot as plt

        fig1 = plt.figure()
        ax1 = plt.subplot(111)
        ax1.plot(list(range(len(self.accuracy))), self.accuracy)
        ax1.set_ylabel("Validation accuracy")
        ax1.set_xlabel("Epochs")

        fig2 = plt.figure()
        ax2 = plt.subplot(111)
        ax2.plot(list(range(len(self.losses))), self.losses)
        ax2.set_ylabel("Validation loss")
        ax2.set_xlabel("Epochs")

        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)

        fig1.savefig(os.path.join(self.outdir, "accuracy.png"), format="png")
        fig2.savefig(os.path.join(self.outdir, "loss.png"), format="png")


class BatchLogger:
    def __init__(self, out_path):
        self.fp = open(out_path / "batch.log", "w", buffering=1)
        self.fp.write("\t".join(("Iteration", "Loss")) + "\n")

    def close(self):
        self.fp.close()

    def log_batch(self, loss, niter):
        self.fp.write(f"{niter}\t{loss:.6f}\n")


def format_mm_ml_tags(seq, poss, probs, mod_bases, can_base):
    """Format MM and ML tags for BAM output. See
    https://samtools.github.io/hts-specs/SAMtags.pdf for format details.

    Args:
        seq (str): read-centric read sequence. For reference-anchored calls
            this should be the reverse complement sequence.
        poss (list): positions relative to seq
        probs (np.array): probabilties for modified bases
        mod_bases (str): modified base single letter codes
        can_base (str): canonical base

    Returns:
        MM string tag and ML array tag
    """

    # initialize dict with all called mods to make sure all called mods are
    # shown in resulting tags
    per_mod_probs = dict((mod_base, []) for mod_base in mod_bases)
    for pos, mod_probs in sorted(zip(poss, probs)):
        # mod_lps is set to None if invalid sequence is encountered or too
        # few events are found around a mod
        if mod_probs is None:
            continue
        for mod_prob, mod_base in zip(mod_probs, mod_bases):
            mod_prob = mod_prob
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
        mm_tag += "{}+{}{};".format(
            can_base,
            mod_base,
            "".join(
                ",{}".format(d)
                for d in np.diff(np.insert(can_base_mod_poss, 0, -1)) - 1
            ),
        )
        # extract mod scores and scale to 0-255 range
        scaled_probs = np.floor(np.array(probs) * 256)
        # last interval includes prob=1
        scaled_probs[scaled_probs == 256] = 255
        ml_tag.extend(scaled_probs.astype(np.uint8))

    return mm_tag, ml_tag


def queue_iter(in_q, num_proc=1):
    comp_proc = 0
    while True:
        try:
            r_val = in_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if r_val is StopIteration:
            comp_proc += 1
            if comp_proc >= num_proc:
                break
        else:
            yield r_val
