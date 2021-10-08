import os
from os.path import realpath, expanduser
import re

import numpy as np

from remora import log, RemoraError

LOGGER = log.get_logger()

CAN_ALPHABET = "ACGT"
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


def validate_mod_bases(mod_bases, motif, alphabet, collapse_alphabet):
    """Validate that inputs are mutually consistent. Return label conversion
    from alphabet integer encodings to modified base categories.
    """
    if len(set(mod_bases)) < len(mod_bases):
        raise RemoraError("Single letter modified base codes must be unique.")
    for mod_base in mod_bases:
        if mod_base not in alphabet:
            raise RemoraError("Modified base provided not found in alphabet")
        mod_can_equiv = collapse_alphabet[alphabet.find(mod_base)]
        # note this check also requires that all modified bases have the same
        # canonical base equivalent.
        if motif.focus_base != mod_can_equiv:
            raise RemoraError(
                f"Canonical base within motif ({motif.focus_base}) does not "
                "match canonical equivalent for modified base "
                f"({mod_can_equiv})"
            )
    label_conv = np.full(len(alphabet), -1, dtype=np.byte)
    label_conv[alphabet.find(mod_can_equiv)] = 0
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
