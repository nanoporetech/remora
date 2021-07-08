import torch
import argparse
import numpy as np
from tqdm import *

import torch.nn.utils.rnn as rnn
import torch.nn.functional as F

from extract_train_data import get_train_set
from chunk_selection import sample_chunks
import models


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
        "--loss", default="CrossEntropy", help="Criterion for training"
    )
    parser.add_argument(
        "--optimizer", default="adamw", help="Optimizer setting"
    )
    parser.add_argument(
        "--lr", default=1e-5, type=float, help="Learning rate setting"
    )
    parser.add_argument(
        "--lr-decay-step",
        default=10,
        type=int,
        help="Learning decay step setting",
    )
    parser.add_argument(
        "--lr-decay-gamma",
        default=0.5,
        type=float,
        help="Learning decay gamma setting",
    )
    parser.add_argument(
        "--weight-decay", default=1e-4, type=float, help="Weight decay setting"
    )
    parser.add_argument(
        "--val-prop",
        default=0.1,
        type=float,
        help="Proportion of the dataset to be used as validation",
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


class ModDataset(torch.utils.data.Dataset):
    def __init__(self, sigs, labels):
        self.sigs = sigs
        self.labels = labels

    def __getitem__(self, index):
        # lbls_oh = F.one_hot(torch.tensor(int(self.labels[index])), num_classes=2)
        return [self.sigs[index], int(self.labels[index])]

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


def validate_model(model, dl):
    with torch.no_grad():
        model.eval()
        outputs = []
        labels = []
        for x, x_len, y in dl:
            x_pack = rnn.pack_padded_sequence(
                x.unsqueeze(2), x_len, enforce_sorted=False
            )
            output = model(x_pack.cuda(), x_len)
            outputs.append(output)
            labels.append(y)
        pred = torch.cat(outputs)
        y_pred = torch.argmax(pred, dim=1)
        lbs = np.concatenate(labels)
        acc = (y_pred == torch.tensor(lbs).cuda()).float().sum() / y_pred.shape[
            0
        ]
        return acc.cpu().numpy()


def main(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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

    idx = np.random.permutation(len(sigs))

    sigs = np.array(sigs)[idx]
    labels = np.array(labels)[idx]

    val_idx = int(len(sigs) * args.val_prop)
    val_set = ModDataset(sigs[:val_idx], labels[:val_idx])

    trn_set = ModDataset(sigs[val_idx:], labels[val_idx:])

    dl_tr = torch.utils.data.DataLoader(
        trn_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.nb_workers,
        drop_last=True,
        collate_fn=collate_fn_padd,
        pin_memory=True,
    )

    dl_val = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.nb_workers,
        drop_last=False,
        collate_fn=collate_fn_padd,
        pin_memory=True,
    )

    model = models.SimpleLSTM()
    model = model.cuda()

    if args.loss == "CrossEntropy":
        criterion = torch.nn.CrossEntropyLoss().cuda()

    if args.optimizer == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=args.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    elif args.optimizer == "adam":
        opt = torch.optim.Adam(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=args.weight_decay,
        )
    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=args.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
    )

    for epoch in range(args.epochs):
        model.train()
        losses = []
        pbar = tqdm(enumerate(dl_tr))
        for i, (x, x_len, y) in pbar:
            x_pack = rnn.pack_padded_sequence(
                x.unsqueeze(2), x_len, enforce_sorted=False
            )

            output = model(x_pack.cuda(), x_len)

            loss = criterion(output, torch.tensor(y).cuda())

            opt.zero_grad()
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            opt.step()

        print("Loss: %s" % np.mean(losses))
        acc = validate_model(model, dl_val)
        print("Model accuracy: %s" % acc)


if __name__ == "__main__":
    main(get_parser().parse_args())
