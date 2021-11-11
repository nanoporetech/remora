import argparse
import os

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class plotter:
    def __init__(self):
        self.losses = []
        self.accuracy = []

    def append_result(self, accuracy, loss):
        self.losses.append(loss)
        self.accuracy.append(accuracy)

    def extract_results(self, dir):
        extract = pd.read_csv(os.path.join(dir, "validation.log"), sep="\t")
        trn_ext = extract[extract["Val_Type"] == "trn"]
        val_ext = extract[extract["Val_Type"] == "val"]

        trn_results = {
            "Loss": trn_ext["Loss"],
            "Update": trn_ext["Iteration"],
            "Accuracy": trn_ext["Accuracy"],
            "F1": trn_ext["F1"],
            "Precision": trn_ext["Precision"],
            "Recall": trn_ext["Recall"],
            "Confusion": trn_ext["Confusion"],
        }
        val_results = {
            "Loss": val_ext["Loss"],
            "Update": val_ext["Iteration"],
            "Accuracy": val_ext["Accuracy"],
            "F1": val_ext["F1"],
            "Precision": val_ext["Precision"],
            "Recall": val_ext["Recall"],
            "Confusion": val_ext["Confusion"],
        }
        folder = dir.rsplit("/", 1)[-1]

        return (trn_results, val_results), folder

    def plot_results(self, args):

        folders = []
        fig, axs = plt.subplots(1, 3, figsize=(25, 15), constrained_layout=True)

        for i, arg in enumerate(args.paths):
            res, folder = self.extract_results(arg)
            folders.append(folder + "_training")
            folders.append(folder)

            axs[0].plot(
                res[0]["Update"],
                res[0]["Accuracy"],
                linestyle="dashed",
                color="C" + str(i),
            )
            axs[0].plot(
                res[1]["Update"], res[1]["Accuracy"], color="C" + str(i)
            )

            axs[1].plot(
                res[0]["Update"],
                res[0]["Loss"],
                linestyle="dashed",
                color="C" + str(i),
            )
            axs[1].plot(res[1]["Update"], res[1]["Loss"], color="C" + str(i))

            axs[2].plot(
                res[0]["Update"],
                res[0]["F1"],
                linestyle="dashed",
                color="C" + str(i),
            )
            axs[2].plot(res[1]["Update"], res[1]["F1"], color="C" + str(i))

        fig.legend(folders, loc="lower right")
        axs[0].set(xlabel="Updates", ylabel="Accuracy")
        axs[1].set(xlabel="Updates", ylabel="Loss")
        axs[2].set(xlabel="Updates", ylabel="F1-score")

        fig.savefig(os.path.join(args.out_path, "results.pdf"), format="pdf")

    def plot_conf_mat(self, args):

        folders = []
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        for i, arg in enumerate(args.paths):
            res, folder = self.extract_results(arg)
            folders.append(folder + "_training")
            folders.append(folder)
            lres = res[0]["Confusion"].shape[0]
            last_idx = res[0]["Confusion"].index[lres - 1]
            mat_str = res[0]["Confusion"][last_idx]
            mat_str = mat_str.replace("[", "")
            mat_str = mat_str.replace("]", "")
            conf_matrix = np.fromstring(mat_str, dtype=int, sep=",")
            cl_nr = int(np.sqrt(len(conf_matrix)))
            conf_matrix = np.reshape(conf_matrix, (cl_nr, cl_nr))
            ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax.text(
                        x=j,
                        y=i,
                        s=conf_matrix[i, j],
                        va="center",
                        ha="center",
                        size="xx-large",
                    )
            plt.xlabel("Predictions", fontsize=18)
            plt.ylabel("Ground Truth", fontsize=18)
            fig.savefig(
                os.path.join(args.out_path, "conf_matrix.pdf"), format="pdf"
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths",
        type=str,
        nargs="+",
        help="The paths of the Remora runs that are to be plotted",
    )
    parser.add_argument("--out-path", type=str, help="Saving to out path")

    args = parser.parse_args()

    plotting = plotter()
    plotting.plot_results(args)
    plotting.plot_conf_mat(args)
