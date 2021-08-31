import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn
from tqdm import tqdm
import os

from remora import models
from remora.extract_train_data import get_train_set, get_centred_train_set
from remora import util
from remora.extract_train_data import get_train_set
from remora.chunk_selection import (
    sample_chunks_bybase,
    sample_chunks_bychunksize,
)


class ModDataset(torch.utils.data.Dataset):
    def __init__(self, sigs, labels):
        self.sigs = sigs
        self.labels = labels

    def __getitem__(self, index):
        # lbls_oh = F.one_hot(
        #    torch.tensor(int(self.labels[index])), num_classes=2
        # )
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
    mask = lengths.ne(0)
    # get labels
    labels = np.array([t[1] for t in batch])
    # padding
    batch = [torch.Tensor(t[0]) for t in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch)

    return batch[:, mask], lengths[mask], labels[mask]


def validate_model(model, dl):
    with torch.no_grad():
        model.eval()
        outputs = []
        labels = []
        for (x, x_len, y) in dl:
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


def validate_cnn_model(model, dl):
    with torch.no_grad():
        model.eval()
        outputs = []
        labels = []
        for x, x_len, y in dl:
            x_pack = torch.from_numpy(np.expand_dims(x.T, 1))
            output = model(x_pack.cuda())
            output = output.to(torch.float32)
            outputs.append(output)
            labels.append(y)
        pred = torch.cat(outputs)
        y_pred = torch.argmax(pred, dim=1)
        lbs = torch.tensor(np.concatenate(labels))
        acc = (y_pred == lbs.cuda()).float().sum() / y_pred.shape[0]
        print(
            f"Nr pred mods {y_pred.sum()}; nr mods {lbs.sum()}; val size {len(lbs)}"
        )
        return acc.cpu().numpy()


def get_results(model, model_type, signals, labels, read_ids, positions):
    with torch.no_grad():
        model.eval()
        # y_val = torch.tensor(labels).unsqueeze(1).float()
        if model_type == "cnn":
            x_pack_val = torch.from_numpy(np.expand_dims(signals, 1)).float()
            output = model(x_pack_val.cuda())
            output = output.to(torch.float32)
            # pred = torch.cat(output)
            y_pred = torch.argmax(output, dim=1).cpu()
            y = labels
        elif model_type == "lstm":
            dataset = ModDataset(signals, labels)
            dl = torch.utils.data.DataLoader(
                dataset,
                batch_size=len(signals),
                shuffle=False,
                num_workers=0,
                drop_last=False,
                collate_fn=collate_fn_padd,
                pin_memory=True,
            )
            for i, (x, x_len, y) in enumerate(dl, 0):
                x_pack = rnn.pack_padded_sequence(
                    x.unsqueeze(2), x_len, enforce_sorted=False
                )

                output = model(x_pack.cuda(), x_len)

        y_pred = torch.argmax(output, dim=1).cpu()
        result = pd.DataFrame(
            {
                "Read_IDs": read_ids,
                "Positions": positions,
                "mod score": y_pred.detach().numpy().tolist(),
                "label": y.tolist(),
            }
        )

    return result


def train_model(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = "/logs"

    LOG_DIR = os.path.join(args.output_path, out_dir)
    plotting_device = util.plotter(LOG_DIR)

    rw = util.resultsWriter(args.output_path, args.output_file_type)

    if args.fixed_chunks:
        if len(args.fixed_chunk_size) == 1:
            (
                sigs,
                labels,
                refs,
                base_locs,
                read_ids,
                positions,
            ) = get_centred_train_set(
                args.dataset_path,
                args.num_chunks,
                args.mod.lower(),
                0,
                0,
                args.fixed_chunk_size[0] // 2,
                args.fixed_chunk_size[0] // 2,
                args.mod_offset,
                args.fixed_chunks,
            )
        else:
            if len(args.chunk_bases) > 2:
                print(
                    "Warning: chunk sizes larger than 2, only using first and second elements"
                )
            if not all(isinstance(i, int) for i in args.chunk_bases):
                raise ValueError(
                    "number of bases before and after mod base must be integer values"
                )
            (
                sigs,
                labels,
                refs,
                base_locs,
                read_ids,
                positions,
            ) = get_centred_train_set(
                args.dataset_path,
                args.num_chunks,
                args.mod.lower(),
                0,
                0,
                args.fixed_chunk_size[0],
                args.fixed_chunk_size[1],
                args.mod_offset,
                args.fixed_chunks,
            )

    else:
        if len(args.chunk_bases) == 0:
            sigs, labels, refs, base_locs = get_train_set(
                args.dataset_path, args.mod_offset
            )
        elif len(args.chunk_bases) == 1:
            if not isinstance(args.chunk_bases[0], int):
                raise ValueError(
                    "number of bases before and after mod base must be integer values"
                )
            (
                sigs,
                labels,
                refs,
                base_locs,
                read_ids,
                positions,
            ) = get_centred_train_set(
                args.dataset_path,
                args.num_chunks,
                args.mod.lower(),
                args.chunk_bases[0],
                args.chunk_bases[0],
                0,
                0,
                args.mod_offset,
                args.fixed_chunks,
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
            (
                sigs,
                labels,
                refs,
                base_locs,
                read_ids,
                positions,
            ) = get_centred_train_set(
                args.dataset_path,
                args.num_chunks,
                args.mod.lower(),
                args.chunk_bases[0],
                args.chunk_bases[1],
                0,
                0,
                args.mod_offset,
                args.fixed_chunks,
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
        drop_last=False,
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

    if args.model == "lstm":
        model = models.SimpleLSTM()
    elif args.model == "cnn":
        model = models.CNN(batch_size=args.batch_size, channel_size=32)
    else:
        raise ValueError("Specify a valid model type to train with")

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

    # training_var = {
    #        "epoch": 0,
    #        "model_name": args.model,
    #    }

    # util.continue_from_checkpoint(
    #        args.output_path,
    #        training_var=training_var,
    #        opt=opt,
    #        model=model,
    #    )

    # start_epoch = training_var["epoch"]
    # model_name = training_var["model_name"]

    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
    )

    start_epoch = 0
    model_name = args.model

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(total=len(dl_tr), leave=True, ncols=100)
        losses = []
        for i, (x, x_len, y) in enumerate(dl_tr):

            if model_name == "lstm":
                x_pack = rnn.pack_padded_sequence(
                    x.unsqueeze(2), x_len, enforce_sorted=False
                )
                output = model.forward(x_pack.cuda(), x_len)

            elif model_name == "cnn":
                x_pack = torch.from_numpy(np.expand_dims(x.T, 1))
                output = model(x_pack.cuda())
                output = output.to(torch.float32)

            target = torch.tensor(y)

            loss = criterion(output, target.cuda())
            opt.zero_grad()
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            opt.step()

            pbar.refresh()
            pbar.set_postfix(loss="%.4f" % loss)
            pbar.update(1)

        pbar.close()
        if model_name == "lstm":
            acc = validate_model(model, dl_val)
        elif model_name == "cnn":
            acc = validate_cnn_model(model, dl_val)

        scheduler.step()

        print(
            f"Model validation accuracy: {acc:.4f}; mean loss: {np.mean(losses):.4f}"
        )
        if int(epoch + 1) % args.save_freq == 0:
            print("Saving model after epoch %s." % int(epoch + 1))
            util.save_checkpoint(
                {
                    "epoch": int(epoch + 1),
                    "model_name": model_name,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "val_accuracy": acc,
                    "mod_offset": args.mod_offset,
                },
                args.output_path,
            )
    result_table = get_results(
        model,
        model_name,
        sigs[:val_idx],
        labels[:val_idx],
        read_ids[:val_idx],
        positions[:val_idx],
    )
    rw.write(result_table)


if __name__ == "__main__":
    NotImplementedError("This is a module.")
