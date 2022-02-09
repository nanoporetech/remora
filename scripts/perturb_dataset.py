import argparse

import numpy as np
from tqdm import tqdm

from remora.data_chunks import RemoraDataset


def get_parser():
    parser = argparse.ArgumentParser(description="Perturb remora dataset")
    parser.add_argument("input_dataset")
    parser.add_argument("output_dataset")
    parser.add_argument(
        "--sequence-mismatch-rate",
        type=float,
        help="Mismatch rate (between 0 and 1) to introduce into reference "
        "sequencing data.",
    )
    parser.add_argument(
        "--signal-bias",
        type=float,
        help="Add signal bias term to all chunks.",
    )
    parser.add_argument(
        "--signal-shift",
        type=int,
        help="Shift signal to sequence mapping by a set number of "
        "signal positions. Negative values shift signal to the right "
        "relative to the sequence.",
    )

    return parser


def main(args):
    ds = RemoraDataset.load_from_file(
        args.input_dataset,
        shuffle_on_iter=False,
        drop_last=False,
    )
    if args.sequence_mismatch_rate is not None:
        ds.perturn_seq_mismatch(args.sequence_mismatch_rate)
    if args.signal_bias is not None:
        ds.sig_tensor += args.signal_bias
    if args.signal_shift is not None:
        ds.perturb_seq_to_sig_map(args.signal_shift)
    ds.save_dataset(args.output_dataset)


if __name__ == "__main__":
    main(get_parser().parse_args())
