import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def extract_scores(dirs):
    scores = {}
    labels = []
    for dir in dirs:
        extract = pd.read_csv(os.path.join(dir, "validation.log"), sep="\t")
        extract = extract[extract.Val_Type == "val"]
        extract = extract.dropna()
        wd = dir.rsplit("/", 3)[-3:]
        if wd[2] not in scores:
            scores[wd[2]] = {}
        if wd[0] not in scores[wd[2]]:
            scores[wd[2]][wd[0]] = {}
        scores[wd[2]][wd[0]][wd[1]] = max(extract.F1)

    return scores


def plot_scores(scores, out_path):

    chunk_sizes = [50, 100, 250]
    legend = []
    for i in scores.keys():
        for k in scores[i].keys():
            plt.plot(chunk_sizes, scores[i][k].values(), marker="^")
            legend.append(i + "-" + k)

    plt.ylim(0, 1)
    plt.ylabel("Validation F1 score")
    plt.xlabel("Chunk Size")
    plt.legend(legend)
    plt.savefig(os.path.join(out_path, "model_comparison.pdf"), format="pdf")


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
    scores = extract_scores(args.paths)
    plot_scores(scores, args.out_path)
