import imp
import os
from os.path import join, isfile, exists

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from remora import log

LOGGER = log.get_logger()


def _load_python_model(model_file, **model_kwargs):

    netmodule = imp.load_source("netmodule", model_file)
    network = netmodule.network(**model_kwargs)
    return network


def save_checkpoint(state, out_path):
    if not exists(out_path):
        os.makedirs(out_path)
    model_name = os.path.basename(state["model_name"]).split(".")[0]
    filename = join(out_path, f"{model_name}_{state['epoch']}.tar")
    torch.save(state, filename)


def continue_from_checkpoint(dir_path, training_var=None, **kwargs):
    if not exists(dir_path):
        return

    all_ckps = [
        f
        for f in os.listdir(dir_path)
        if isfile(join(dir_path, f)) and ".tar" in f
    ]
    if all_ckps == []:
        return

    ckp_path = join(dir_path, max(all_ckps))

    LOGGER.info(f"Loading trained model from {ckp_path}")

    ckp = torch.load(ckp_path)

    for key, value in kwargs.items():
        if key in ckp:
            try:
                value.load_state_dict(ckp[key])
            except AttributeError:
                continue

    if training_var is not None:
        for var in training_var:
            if var in ckp:
                training_var[var] = ckp[var]


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
            for inputs, labels in dl:
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
                precision, recall, thresholds = precision_recall_curve(
                    all_labels.numpy(), all_outputs[:, 1].numpy()
                )
                with np.errstate(invalid="ignore"):
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
