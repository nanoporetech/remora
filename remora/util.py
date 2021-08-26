import torch
import os
import pandas as pd


def save_checkpoint(state, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    filename = os.path.join(
        out_path, "%s_%s.tar" % (state["model"], state["epoch"])
    )
    torch.save(state, filename)


class resultsWriter:
    def __init__(self, output_filename, output_filetype):

        self.output_filename = output_filename
        self.output_filetype = output_filetype
        self.initialise()

    def initialise(self):

        if self.output_filetype == None or self.output_filetype == "txt":
            self.extension = ".txt"
        elif self.output_filetype.lower() == "sam":
            self.extension = ".sam"
        elif self.output_filetype.lower() == "bam":
            self.extension = ".bam"

        column_names = ["Read ID", "Position", "Mod Score"]
        df = pd.DataFrame(columns=column_names)
        df.to_csv(self.output_filename + self.extension, sep="\t")

    def write(self, results_table):

        with open(self.output_filename + self.extension, "a") as f:
            results_table.to_csv(f, header=f.tell() == 0)
