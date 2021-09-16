from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve

from remora import models, util, log, RemoraError
from remora.data_chunks import load_chunks

LOGGER = log.get_logger()


class ModDataset(torch.utils.data.Dataset):
    def __init__(self, sigs, labels):
        self.sigs = sigs
        self.labels = labels

    def __getitem__(self, index):
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


def validate_model(model, dl, fixed_seq_len_chunks):
    with torch.no_grad():
        model.eval()
        outputs = []
        labels = []
        for (x, x_len, y) in dl:
            if fixed_seq_len_chunks:
                x_pack = rnn.pack_padded_sequence(
                    x.unsqueeze(2), x_len, enforce_sorted=False
                )
                output = model(x_pack.cuda(), x_len)
            else:
                x_pack = torch.from_numpy(np.expand_dims(x.T, 1))
                output = model(x_pack.cuda())
                output = output.to(torch.float32)
            outputs.append(output)
            labels.append(y)
        pred = torch.cat(outputs)
        lbs = torch.tensor(np.concatenate(labels))
        precision, recall, thresholds = precision_recall_curve(
            lbs.cpu().numpy(), pred[:, 1].cpu().numpy()
        )
        with np.errstate(invalid="ignore"):
            f1_scores = 2 * recall * precision / (recall + precision)
        LOGGER.info(
            f"F1: {np.max(f1_scores):.4f}, "
            f"prec: {precision[np.argmax(f1_scores)]:.4f}, "
            f"recall: {recall[np.argmax(f1_scores)]:.4f}"
        )
        y_pred = torch.argmax(pred, dim=1)
        acc = (y_pred == lbs.cuda()).float().sum() / y_pred.shape[0]
        LOGGER.info(
            f"Nr pred mods {y_pred.sum()}; nr mods {lbs.sum()}; "
            f"val size {len(lbs)}"
        )
        return acc.cpu().numpy()


def get_results(
    model, fixed_seq_len_chunks, signals, labels, read_ids, positions
):
    with torch.no_grad():
        model.eval()
        # y_val = torch.tensor(labels).unsqueeze(1).float()
        if fixed_seq_len_chunks:
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
        else:
            x_pack_val = torch.from_numpy(np.expand_dims(signals, 1)).float()
            output = model(x_pack_val.cuda())
            output = output.to(torch.float32)
            y_pred = torch.argmax(output, dim=1).cpu()
            y = labels

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


def train_model(
    seed,
    device,
    out_path,
    dataset_path,
    num_chunks,
    mod,
    mod_offset,
    chunk_context,
    fixed_seq_len_chunks,
    val_prop,
    batch_size,
    nb_workers,
    model_name,
    size,
    optimizer,
    lr,
    weight_decay,
    lr_decay_step,
    lr_decay_gamma,
    epochs,
    save_freq,
    plot,
    references,
):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.cuda.set_device(device)

    rw = util.resultsWriter(out_path / "results.txt")
    plot = util.plotter(out_path)

    sigs, labels, refs, base_locs, read_ids, positions = load_chunks(
        dataset_path,
        num_chunks,
        mod,
        mod_offset,
        chunk_context,
        fixed_seq_len_chunks,
    )
    if references:
        from remora.reference_functions import referenceEncoder

        re = referenceEncoder(mod_offset, chunk_context, fixed_seq_len_chunks)
        enc_refs = re.get_reference_encoding(sigs, refs, base_locs, kmer_size=3)

    LOGGER.info(f"Label distribution: {Counter(labels)}")

    idx = np.random.permutation(len(sigs))

    sigs = np.array(sigs)[idx]
    labels = np.array(labels)[idx]

    val_idx = int(len(sigs) * val_prop)
    val_set = ModDataset(sigs[:val_idx], labels[:val_idx])

    trn_set = ModDataset(sigs[val_idx:], labels[val_idx:])

    dl_tr = torch.utils.data.DataLoader(
        trn_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nb_workers,
        drop_last=False,
        collate_fn=collate_fn_padd,
        pin_memory=True,
    )

    dl_val = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nb_workers,
        drop_last=False,
        collate_fn=collate_fn_padd,
        pin_memory=True,
    )

    if model_name == "lstm":
        if fixed_seq_len_chunks:
            model = models.SimpleLSTM(size=size)
        else:
            model = models.SimpleFWLSTM(size=size)
    elif model_name == "cnn":
        if fixed_seq_len_chunks:
            raise RemoraError(
                "Convolutional network not compatoble with variable signal "
                "length chunks."
            )
        model = models.CNN(size=size)
    else:
        raise ValueError("Specify a valid model type to train with")
    model = model.cuda()
    LOGGER.info(f"Model structure:{model}")

    criterion = torch.nn.CrossEntropyLoss().cuda()

    if optimizer == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=float(lr),
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    elif optimizer == "adam":
        opt = torch.optim.Adam(
            model.parameters(),
            lr=float(lr),
            weight_decay=weight_decay,
        )
    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(lr),
            weight_decay=weight_decay,
        )

    training_var = {
        "epoch": 0,
        "model_name": model_name,
    }

    start_epoch = 0

    util.continue_from_checkpoint(
        out_path,
        training_var=training_var,
        opt=opt,
        model=model,
    )

    start_epoch = training_var["epoch"]

    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=lr_decay_step, gamma=lr_decay_gamma
    )

    acc = validate_model(model, dl_val, fixed_seq_len_chunks)
    for epoch in range(start_epoch, epochs):
        model.train()
        pbar = tqdm(total=len(dl_tr), leave=True, ncols=100)
        losses = []
        for i, (x, x_len, y) in enumerate(dl_tr):
            if fixed_seq_len_chunks:
                x_pack = rnn.pack_padded_sequence(
                    x.unsqueeze(2), x_len, enforce_sorted=False
                )
                output = model.forward(x_pack.cuda(), x_len)
            else:
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
            pbar.set_postfix(epoch=f"{epoch}", loss=f"{loss:.4f}")
            pbar.update(1)

        pbar.close()

        acc = validate_model(model, dl_val, fixed_seq_len_chunks)
        plot.append_result(acc, np.mean(losses))

        scheduler.step()

        LOGGER.info(
            f"Model validation accuracy: {acc:.4f}; mean loss: "
            f"{np.mean(losses):.4f}"
        )
        if int(epoch + 1) % save_freq == 0:
            LOGGER.info(f"Saving model after epoch {int(epoch + 1)}.")
            util.save_checkpoint(
                {
                    "epoch": int(epoch + 1),
                    "model_name": model_name,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "val_accuracy": acc,
                    "mod_offset": mod_offset,
                },
                out_path,
            )
    plot.save_plots()
    result_table = get_results(
        model,
        fixed_seq_len_chunks,
        sigs[:val_idx],
        labels[:val_idx],
        read_ids[:val_idx],
        positions[:val_idx],
    )
    rw.write(result_table)


if __name__ == "__main__":
    NotImplementedError("This is a module.")
