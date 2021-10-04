import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


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
        }
        val_results = {
            "Loss": val_ext["Loss"],
            "Update": val_ext["Iteration"],
            "Accuracy": val_ext["Accuracy"],
            "F1": val_ext["F1"],
            "Precision": val_ext["Precision"],
            "Recall": val_ext["Recall"],
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
