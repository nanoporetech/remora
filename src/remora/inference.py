import atexit
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from remora import util, log
from remora.data_chunks import load_datasets, load_taiyaki_dataset

LOGGER = log.get_logger()


class resultsWriter:
    def __init__(self, output_path):
        self.sep = "\t"
        self.out_fp = open(output_path, "w")
        df = pd.DataFrame(
            columns=[
                "read_id",
                "read_pos",
                "label",
                "class_pred",
                "class_probs",
            ]
        )
        df.to_csv(self.out_fp, sep=self.sep, index=False)

    def write_results(self, output, read_data, labels):
        read_ids, read_pos = zip(*read_data)
        class_preds = output.argmax(dim=1)
        str_probs = [
            ",".join(map(str, r))
            for r in F.softmax(output, dim=1).detach().cpu().numpy()
        ]
        pd.DataFrame(
            {
                "read_id": read_ids,
                "read_pos": read_pos,
                "label": labels,
                "class_pred": class_preds,
                "class_probs": str_probs,
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
    ckpt_attrs = "\n".join(
        f"  {k: >20} : {v}"
        for k, v in ckpt.items()
        if k not in ("state_dict", "opt")
    )
    LOGGER.debug(f"Loaded model attrs\n{ckpt_attrs}\n")
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    LOGGER.info("Loading Taiyaki dataset")
    reads, alphabet, collapse_alphabet = load_taiyaki_dataset(dataset_path)
    motif = util.Motif(*ckpt["motif"])
    if ckpt["base_pred"]:
        if alphabet != "ACGT":
            raise ValueError(
                "Base prediction is not compatible with modified base "
                "training data. It requires a canonical alphabet."
            )
        label_conv = np.arange(4)
    else:
        mod_bases = ckpt["mod_bases"]
        if any(b not in alphabet for b in mod_bases):
            label_conv = np.full(len(alphabet), -1, dtype=int)
        else:
            util.validate_mod_bases(
                mod_bases, motif, alphabet, collapse_alphabet
            )
            mod_can_equiv = collapse_alphabet[alphabet.find(mod_bases[0])]
            label_conv = np.full(len(alphabet), -1, dtype=int)
            label_conv[alphabet.find(mod_can_equiv)] = 0
            for mod_i, mod_base in enumerate(mod_bases):
                label_conv[alphabet.find(mod_base)] = mod_i + 1
    LOGGER.info("Converting dataset for Remora input")
    dl_infer, _, _, _ = load_datasets(
        reads,
        ckpt["chunk_context"],
        batch_size=batch_size,
        val_prop=0.0,
        full=full,
        focus_offset=focus_offset,
        motif=motif,
        label_conv=label_conv,
        base_pred=ckpt["base_pred"],
        num_data_workers=nb_workers,
        kmer_context_bases=ckpt["kmer_context_bases"],
    )

    LOGGER.info("Applying model to loaded data")
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
