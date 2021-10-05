import imp
import os
from os.path import isfile, realpath, expanduser
import re

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from remora import log, RemoraError

LOGGER = log.get_logger()

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


def get_mod_bases(alphabet, collapse_alphabet):
    return [
        mod_base
        for mod_base, can_base in zip(alphabet, collapse_alphabet)
        if mod_base != can_base
    ]


def _load_python_model(model_file, **model_kwargs):

    netmodule = imp.load_source("netmodule", model_file)
    network = netmodule.network(**model_kwargs)
    return network


def continue_from_checkpoint(ckp_path, model_path=None):
    if not isfile(ckp_path):
        raise RemoraError(f"Checkpoint path is not a file ({ckp_path})")
    LOGGER.info(f"Loading trained model from {ckp_path}")
    ckpt = torch.load(ckp_path)
    if ckpt["state_dict"] is None:
        raise RemoraError("Model state not saved in checkpoint.")

    model_path = ckpt["model_path"] if model_path is None else model_path
    model = _load_python_model(model_path, **ckpt["model_params"])
    model.load_state_dict(ckpt["state_dict"])
    return ckpt, model


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
    label_conv = np.full(len(alphabet), -1, dtype=int)
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


class ValidationLogger:
    def __init__(self, out_path, base_pred):
        self.fp = open(out_path / "validation.log", "w", buffering=1)
        self.base_pred = base_pred
        if base_pred:
            self.fp.write(
                "\t".join(
                    ("Val_Type", "Iteration", "Accuracy", "Loss", "Num_Calls")
                )
                + "\n"
            )
        else:
            self.fp.write(
                "\t".join(
                    (
                        "Val_Type",
                        "Iteration",
                        "Accuracy",
                        "Loss",
                        "F1",
                        "Precision",
                        "Recall",
                        "Num_Calls",
                    )
                )
                + "\n"
            )

    def close(self):
        self.fp.close()

    def validate_model(self, model, criterion, dl, niter, val_type="val"):
        with torch.no_grad():
            model.eval()
            all_labels = []
            all_outputs = []
            all_loss = []
            for inputs, labels, _ in dl:
                all_labels.append(labels)
                if torch.cuda.is_available():
                    inputs = (input.cuda() for input in inputs)
                output = model(*inputs).detach().cpu()
                all_outputs.append(output)
                loss = criterion(output, labels)
                all_loss.append(loss.detach().cpu().numpy())
            all_outputs = torch.cat(all_outputs)
            all_labels = torch.cat(all_labels)
            acc = (
                torch.argmax(all_outputs, dim=1) == all_labels
            ).float().sum() / all_outputs.shape[0]
            acc = acc.cpu().numpy()
            mean_loss = np.mean(all_loss)
            if self.base_pred:
                self.fp.write(
                    f"{val_type}\t{niter}\t{acc:.6f}\t{mean_loss:.6f}\t"
                    f"{len(all_labels)}\n"
                )
            else:
                with np.errstate(invalid="ignore"):
                    precision, recall, thresholds = precision_recall_curve(
                        all_labels.numpy(), all_outputs[:, 1].numpy()
                    )
                    f1_scores = 2 * recall * precision / (recall + precision)
                f1_idx = np.argmax(f1_scores)
                self.fp.write(
                    f"{val_type}\t{niter}\t{acc:.6f}\t{mean_loss:.6f}\t"
                    f"{f1_scores[f1_idx]}\t{precision[f1_idx]}\t"
                    f"{recall[f1_idx]}\t{len(all_labels)}\n"
                )
        return acc, mean_loss


class BatchLogger:
    def __init__(self, out_path):
        self.fp = open(out_path / "batch.log", "w", buffering=1)
        self.fp.write("\t".join(("Iteration", "Loss")) + "\n")

    def close(self):
        self.fp.close()

    def log_batch(self, loss, niter):
        self.fp.write(f"{niter}\t{loss:.6f}\n")
