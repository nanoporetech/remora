import argparse

import numpy as np

from remora import util
from remora.refine_signal_map import SigMapRefiner
from remora.data_chunks import CoreRemoraDataset, DatasetMetadata


ARR_CONV = {
    "sig_tensor": "signal",
    "seq_array": "sequence",
    "seq_mappings": "sequence_to_signal_mapping",
    "seq_lens": "sequence_lengths",
    "labels": "labels",
    "read_ids": "read_ids",
    "read_focus_bases": "read_focus_bases",
}


def main(args):
    util.prepare_out_dir(args.out_dataset, args.overwrite)
    in_data = np.load(args.in_dataset)

    # load and convert metadata
    metadata = dict(
        (mdn, in_data[mdn]) for mdn in set(in_data.files).difference(ARR_CONV)
    )
    metadata["extra_arrays"] = {
        "read_ids": ("<U36", "Read identifier"),
        "read_focus_bases": ("int64", "Position within read training sequence"),
    }
    metadata["allocate_size"] = in_data["labels"].shape[0]
    metadata["max_seq_len"] = in_data["seq_array"].shape[1] - sum(
        in_data["kmer_context_bases"]
    )
    if len(metadata["mod_bases"].shape) == 0:
        metadata["mod_bases"] = str(metadata["mod_bases"])
    elif metadata["mod_bases"].shape[0] == 0:
        metadata["mod_bases"] = ""
    else:
        metadata["mod_bases"] = str(metadata["mod_bases"][0])
    metadata["motif_offsets"] = metadata["motif_offset"]
    del metadata["motif_offset"]
    metadata["motif_sequences"] = metadata["motifs"]
    del metadata["motifs"]
    metadata["modified_base_labels"] = not metadata["base_pred"]
    del metadata["base_pred"]
    metadata["refine_do_rough_rescale"] = (
        metadata["refine_do_rough_rescale"] == 1
    )
    metadata["sig_map_refiner"] = SigMapRefiner.load_from_metadata(metadata)
    refine_attrs = [k for k in metadata if k.startswith("refine_")]
    for ra in refine_attrs:
        del metadata[ra]
    metadata["base_start_justify"] = metadata["base_start_justify"] == 1
    metadata["version"] = 3

    # shuffle chunks before disk storage
    shuf_idx = np.random.permutation(metadata["allocate_size"])
    out_ds = CoreRemoraDataset(
        args.out_dataset, mode="w", metadata=DatasetMetadata(**metadata)
    )
    out_ds.write_batch(
        dict(
            (new_name, in_data[old_name][shuf_idx])
            for old_name, new_name in ARR_CONV.items()
        )
    )
    out_ds.write_metadata()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dataset")
    parser.add_argument("out_dataset")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
