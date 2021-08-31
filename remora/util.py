import torch
import os
from os.path import join, isfile, exists
import pandas as pd


def save_checkpoint(state, out_path):
    if not exists(out_path):
        os.makedirs(out_path)
    filename = join(
        out_path, "%s_%s.tar" % (state["model_name"], state["epoch"])
    )
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

    print("Continuing training from %s" % ckp_path)

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


class resultsWriter:
    def __init__(self, output_path, output_filetype):

        self.output_path = output_path
        self.output_filetype = output_filetype
        self.output_filename = "results"
        self.initialise()

    def initialise(self):

        if not exists(self.output_path):
            os.makedirs(self.output_path)
        if self.output_filetype == None or self.output_filetype == "txt":
            self.extension = ".txt"
        elif self.output_filetype.lower() == "sam":
            self.extension = ".sam"
        elif self.output_filetype.lower() == "bam":
            self.extension = ".bam"

        self.full_output = (
            self.output_path + self.output_filename + self.extension
        )

        column_names = ["Read ID", "Position", "Mod Score"]
        df = pd.DataFrame(columns=column_names)
        df.to_csv(self.full_output, sep="\t")

    def write(self, results_table):

        with open(self.full_output, "a") as f:
            results_table.to_csv(f, header=f.tell() == 0)


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

        fig1 = plt.plot(self.accuracy)
        fig2 = plt.plot(self.losses)

        fig1.savefig(os.path.join(outdir, "accuracy.png"), format="png")
        fig2.savefig(os.path.join(outdir, "loss.png"), format="png")
