import torch
import pandas as pd
import os
from tqdm import tqdm

from remora import util, log
from remora.data_chunks import load_datasets

LOGGER = log.get_logger()


class resultsWriter:
    def __init__(self, output_path):
        self.output_path = output_path
        column_names = ["read_id", "read_pos", "mod_prob", "label"]
        df = pd.DataFrame(columns=column_names)
        df.to_csv(self.output_path, sep="\t")

    def write(self, results_table):
        with open(self.output_path, "a") as f:
            results_table.to_csv(f, header=f.tell() == 0)


def get_results(output, read_data, labels, it):
    y_pred = torch.argmax(output, dim=1)
    result = pd.DataFrame(
        {
            "read_id": [rd[0] for rd in read_data[it : it + len(y_pred)]],
            "read_pos": [rd[1] for rd in read_data[it : it + len(y_pred)]],
            "mod_prob": y_pred.detach().numpy().tolist(),
            "label": labels,
        }
    )
    return result


def infer(
    out_path,
    dataset_path,
    checkpoint_path,
    batch_size,
    nb_workers,
    device,
    full,
):
    LOGGER.info("Detecting modified bases.")

    torch.cuda.set_device(device)

    rw = resultsWriter(os.path.join(out_path, "results.csv"))

    # Some default values that will get overwritten by the checkpoint
    training_var = {
        "model_name": "lstm",
        "state_dict": {},
        "fixed_seq_len_chunks": False,
        "chunk_context": 0,
        "mod_motif": "a",
        "base_pred": False,
    }

    util.continue_from_checkpoint(
        checkpoint_path,
        training_var=training_var,
    )

    model_name = training_var["model_name"]
    fixed_seq_len_chunks = training_var["fixed_seq_len_chunks"]
    chunk_context = training_var["chunk_context"]
    mod_motif = training_var["mod_motif"]
    base_pred = training_var["base_pred"]
    state_dict = training_var["state_dict"]

    model = util._load_python_model(model_name)

    if state_dict != {}:
        model.load_state_dict(state_dict)

    dl_test, _, _ = load_datasets(
        dataset_path,
        chunk_context,
        batch_size=batch_size,
        fixed_seq_len_chunks=fixed_seq_len_chunks,
        mod_motif=mod_motif,
        base_pred=base_pred,
        num_data_workers=nb_workers,
        full=full,
        val_prop=0.0,
    )
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    pbar = tqdm(
        total=len(dl_test),
        desc="Extraction Progress",
        dynamic_ncols=True,
        position=1,
        leave=True,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| " "{n_fmt}/{total_fmt}",
    )
    pbar.n = 0
    pbar.refresh()
    for batch_i, (inputs, labels, read_data) in enumerate(dl_test):
        if torch.cuda.is_available():
            inputs = (input.cuda() for input in inputs)
        output = model(*inputs).detach().cpu()
        result_table = get_results(
            output, read_data, labels, batch_i * batch_size
        )
        rw.write(result_table)
        pbar.update()
        pbar.refresh()


if __name__ == "__main__":
    NotImplementedError("This is a module.")
