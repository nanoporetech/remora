import os
from os.path import join, isfile, exists

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from remora import log

LOGGER = log.get_logger()


def save_checkpoint(state, out_path):
    if not exists(out_path):
        os.makedirs(out_path)
    filename = join(out_path, f"{state['model_name']}_{state['epoch']}.tar")
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

    LOGGER.info(f"Continuing training from {ckp_path}")

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
        self.fp = open(out_path / "validation.log", "w")
        self.base_pred = base_pred
        if base_pred:
            self.fp.write(
                "\t".join(
                    ("Validation_Type", "Iteration", "Accuracy", "Num_Calls")
                )
                + "\n"
            )
        else:
            self.fp.write(
                "\t".join(
                    (
                        "Validation_Type",
                        "Iteration",
                        "Accuracy",
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

    def validate_model(self, model, dl, niter, val_type="validation"):
        with torch.no_grad():
            model.eval()
            all_outputs = []
            all_labels = []
            for inputs, labels in dl:
                if torch.cuda.is_available():
                    inputs = (input.cuda() for input in inputs)
                all_outputs.append(model(*inputs).detach().cpu())
                all_labels.append(labels)
            all_outputs = torch.cat(all_outputs)
            preds = torch.argmax(all_outputs, dim=1)
            all_labels = torch.cat(all_labels)
            acc = (preds == all_labels).float().sum() / preds.shape[0]
            acc = acc.cpu().numpy()
            if self.base_pred:
                self.fp.write(
                    f"{val_type}\t{niter}\t{acc:.6f}\t{len(all_labels)}\n"
                )
            else:
                precision, recall, thresholds = precision_recall_curve(
                    all_labels.numpy(), all_outputs[:, 1].numpy()
                )
                with np.errstate(invalid="ignore"):
                    f1_scores = 2 * recall * precision / (recall + precision)
                f1_idx = np.argmax(f1_scores)
                self.fp.write(
                    f"{val_type}\t{niter}\t{acc:.6f}\t{f1_scores[f1_idx]}\t"
                    f"{precision[f1_idx]}\t{recall[f1_idx]}\t"
                    f"{len(all_labels)}\n"
                )
        return acc
