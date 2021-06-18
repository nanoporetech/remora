import torch
import argparse
import numpy as np

import torch.nn.utils.rnn as rnn

from extract_train_data import get_train_set
from chunk_selection import sample_chunks


def get_parser():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--LOG_DIR", default="./output/", help="Path to log folder"
    )
    parser.add_argument(
        "--dataset-path",
        default="toy_training_data.hdf5",
        help="Training dataset",
    )
    parser.add_argument(
        "--chunk_bases",
        default=[],
        type=tuple,
        help="sample smaller chunks from the reads according to bases before and after mod",
    )
    parser.add_argument(
        "--batch-size",
        default=200,
        type=int,
        help="Number of samples per batch.",
    )
    parser.add_argument(
        "--epochs", default=50, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--gpu-id",
        default=0,
        type=int,
        help="ID of GPU that is used for training.",
    )
    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        dest="nb_workers",
        help="Number of workers for dataloader.",
    )
    parser.add_argument("--model", default="mlp", help="Model for training")
    parser.add_argument(
        "--loss", default="Proxy_Anchor", help="Criterion for training"
    )
    parser.add_argument(
        "--optimizer", default="adamw", help="Optimizer setting"
    )
    parser.add_argument(
        "--lr", default=1e-5, type=float, help="Learning rate setting"
    )
    parser.add_argument(
        "--weight-decay", default=1e-4, type=float, help="Weight decay setting"
    )
    parser.add_argument(
        "--mod-offset",
        default=20,
        type=int,
        dest="MOD_OFFSET",
        help="Seed value",
    )
    parser.add_argument("--seed", default=1, type=int, help="Seed value")
    parser.add_argument("--remark", default="", help="Any reamrk")

    return parser


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


class ModDataset(torch.utils.data.Dataset):
    def __init__(self, sigs, labels):
        self.sigs = sigs
        self.labels = labels

    def __getitem__(self, index):
        return [self.sigs[index], self.labels[index]]

    def __len__(self):
        return len(self.sigs)


def collate_fn_padd(batch):
    """
    Pads batch of variable sequence lengths

    note: the output is passed to the pack_padded_sequence,
        so that variable sequence lenghts can be handled by
        the RNN
    """
    # get sequence lengths
    lengths = torch.tensor([t[0].shape[0] for t in batch])
    # get labels
    labels = [t[1] for t in batch]
    # padding
    batch = [torch.Tensor(t[0]) for t in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch)

    return batch, lengths, labels


def main(args):

    out_dir = "/logs"

    LOG_DIR = args.LOG_DIR + out_dir

    if len(args.chunk_bases) == 0:
        sigs, labels, refs, base_locs = get_train_set(
            args.dataset_path, args.MOD_OFFSET
        )
    elif len(args.chunk_bases) == 1:
        if not isinstance(args.chunk_bases[0], int):
            raise ValueError(
                "number of bases before and after mod base must be integer values"
            )

        sigs, labels, refs, base_locs = sample_chunks(
            args.dataset_path,
            args.number_to_sample,
            args.chunk_bases[0],
            args.chunk_bases[0],
            args.MOD_OFFSET,
        )
    else:
        if len(args.chunk_bases) > 2:
            print(
                "Warning: chunk bases larger than 2, only using first and second elements"
            )

        if not all(isinstance(i, int) for i in args.chunk_bases):
            raise ValueError(
                "number of bases before and after mod base must be integer values"
            )

        sigs, labels, refs, base_locs = sample_chunks(
            args.dataset_path,
            args.number_to_sample,
            args.chunk_bases[0],
            args.chunk_bases[1],
            args.MOD_OFFSET,
        )

    trn_set = ModDataset(sigs, labels)

    dl_tr = torch.utils.data.DataLoader(
        trn_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.nb_workers,
        drop_last=True,
        collate_fn=collate_fn_padd,
        pin_memory=True,
    )

    model = torch.nn.LSTM(1, 32, 1)
    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=args.weight_decay,
        momentum=0.9,
        nesterov=True,
    )

    for epoch in range(args.epochs):
        model.train()
        for x, len_x, y in dl_tr:
            x_pack = rnn.pack_padded_sequence(
                x.unsqueeze(2), len_x, enforce_sorted=False
            )
            m = model(x_pack.cuda())
            pdb.set_trace()
            loss = criterion(m, y)


if __name__ == "__main__":
    main(get_parser().parse_args())
