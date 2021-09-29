import imp
import os
from os.path import join, isfile, exists, realpath, expanduser

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from remora import log, RemoraError

LOGGER = log.get_logger()


def resolve_path(fn_path):
    """Helper function to resolve relative and linked paths that might
    give other packages problems.
    """
    if fn_path is None:
        return None
    return realpath(expanduser(fn_path))


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


def validate_motif(motif, alphabet="ACGT", collapse_alphabet="ACGT"):
    mod_base, can_motif, motif_offset = motif
    try:
        motif_offset = int(motif_offset)
    except ValueError:
        raise RemoraError(f'Motif offset not an integer: "{motif_offset}"')
    if motif_offset >= len(motif):
        raise RemoraError("Motif offset is past the end of the motif")
    if any(b not in alphabet for b in can_motif):
        raise RemoraError(
            "Base(s) in motif provided not found in alphabet "
            f'"{set(can_motif).difference(alphabet)}"'
        )
    can_base = can_motif[motif_offset]
    if mod_base not in alphabet:
        LOGGER.warning("Modified base provided not found in alphabet")
        mod_base = None
    else:
        mod_can_equiv = collapse_alphabet[alphabet.find(mod_base)]
        if can_base != mod_can_equiv:
            raise RemoraError(
                f"Canonical base within motif ({can_base}) does not match "
                f"canonical equivalent for modified base ({mod_can_equiv})"
            )
    non_can_motif_bases = set(can_motif).difference(collapse_alphabet)
    if len(non_can_motif_bases) > 0:
        raise RemoraError(
            "Non-canonical bases found in motif "
            f"({','.join(non_can_motif_bases)})"
        )

    return mod_base, can_motif, motif_offset


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
