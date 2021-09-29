import atexit
import os

import torch
import pandas as pd
from tqdm import tqdm

from remora import util, log
from remora.data_chunks import load_datasets, load_taiyaki_dataset

LOGGER = log.get_logger()


class resultsWriter:
    def __init__(self, output_path):
        self.sep = "\t"
        self.out_fp = open(output_path, "w")
        column_names = ["read_id", "read_pos", "mod_prob", "label"]
        df = pd.DataFrame(columns=column_names)
        df.to_csv(self.out_fp, sep=self.sep, index=False)

    def write_results(self, output, read_data, labels):
        # TODO support multiple mod probs
        mod_prob = output[:, 1]
        read_ids, read_pos = zip(*read_data)
        pd.DataFrame(
            {
                "read_id": read_ids,
                "read_pos": read_pos,
                "mod_prob": mod_prob.detach().numpy(),
                "label": labels,
            }
        ).to_csv(self.out_fp, header=False, index=False, sep=self.sep)

    def close(self):
        self.out_fp.close()


def infer(
    out_path,
    dataset_path,
    checkpoint_path,
    model_path,
    batch_size,
    nb_workers,
    device,
    focus_offset,
    full,
):
    LOGGER.info("Performing Remora inference")

    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    elif device is not None:
        LOGGER.warning(
            "Device option specified, but CUDA is not available from torch."
        )

    rw = resultsWriter(os.path.join(out_path, "results.tsv"))
    atexit.register(rw.close)

    ckpt, model = util.continue_from_checkpoint(checkpoint_path, model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    reads, alphabet, collapse_alphabet = load_taiyaki_dataset(dataset_path)
    mod_motif = ckpt["mod_motif"]
    mod_motif = util.validate_motif(mod_motif, alphabet, collapse_alphabet)
    dl_infer, _, _, _ = load_datasets(
        reads,
        ckpt["chunk_context"],
        batch_size=batch_size,
        focus_offset=focus_offset,
        full=full,
        mod_motif=mod_motif,
        alphabet=alphabet,
        num_data_workers=nb_workers,
        val_prop=0.0,
    )

    pbar = tqdm(
        total=len(dl_infer),
        smoothing=0,
        desc="Call Batches",
        dynamic_ncols=True,
    )
    atexit.register(pbar.close)
    for inputs, labels, read_data in dl_infer:
        if torch.cuda.is_available():
            inputs = (input.cuda() for input in inputs)
        output = model(*inputs).detach().cpu()
        rw.write_results(output, read_data, labels)
        pbar.update()


if __name__ == "__main__":
    NotImplementedError("This is a module.")
