import atexit
import os
from shutil import copyfile

import numpy as np
import torch
from tqdm import tqdm

from remora import util, log, RemoraError
from remora.data_chunks import load_datasets

LOGGER = log.get_logger()


def train_model(
    seed,
    device,
    out_path,
    dataset_path,
    num_chunks,
    mod_motif,
    focus_offset,
    chunk_context,
    fixed_seq_len_chunks,
    val_prop,
    batch_size,
    num_data_workers,
    model_name,
    size,
    optimizer,
    lr,
    weight_decay,
    lr_decay_step,
    lr_decay_gamma,
    base_pred,
    epochs,
    save_freq,
    kmer_size,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.cuda.set_device(device)

    val_fp = util.ValidationLogger(out_path, base_pred)
    atexit.register(val_fp.close)
    batch_fp = util.BatchLogger(out_path)
    atexit.register(batch_fp.close)

    dl_trn, dl_val, dl_val_trn = load_datasets(
        dataset_path,
        chunk_context,
        focus_offset,
        batch_size,
        num_chunks,
        fixed_seq_len_chunks,
        mod_motif,
        base_pred,
        val_prop,
        num_data_workers,
        kmer_size,
    )
    model = util._load_python_model(model_name)
    if (fixed_seq_len_chunks and not model._variable_width_possible) or (
        not fixed_seq_len_chunks and model._variable_width_possible
    ):
        raise RemoraError(
            "Trying to use variable chunk sizes with a fixed chunk-size model"
        )

    copyfile(model_name, os.path.join(out_path, "model.py"))
    model = model.cuda()
    LOGGER.info(f"Model structure:\n{model}")

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
        "state_dict": {},
    }
    util.continue_from_checkpoint(
        out_path,
        training_var=training_var,
        opt=opt,
        model=model,
    )
    start_epoch = training_var["epoch"]
    state_dict = training_var["state_dict"]
    if state_dict != {}:
        model.load_state_dict(state_dict)

    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=lr_decay_step, gamma=lr_decay_gamma
    )

    # assess accuracy before first iteration
    val_acc, val_loss = val_fp.validate_model(model, criterion, dl_val, 0)
    trn_acc, trn_loss = val_fp.validate_model(
        model, criterion, dl_val_trn, 0, "trn"
    )
    ebar = tqdm(
        total=epochs,
        smoothing=0,
        desc="Epochs",
        dynamic_ncols=True,
        position=0,
        leave=True,
    )
    ebar.n = start_epoch
    pbar = tqdm(
        total=len(dl_trn),
        desc="Epoch Progress",
        dynamic_ncols=True,
        position=1,
        leave=True,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| " "{n_fmt}/{total_fmt}",
    )
    ebar.set_postfix(
        acc_val=f"{val_acc:.4f}",
        acc_train=f"{trn_acc:.4f}",
        loss_val=f"{val_loss:.6f}",
        loss_train=f"{trn_loss:.6f}",
    )
    atexit.register(ebar.close)
    atexit.register(pbar.close)
    for epoch in range(start_epoch, epochs):
        model.train()
        pbar.n = 0
        pbar.refresh()
        for epoch_i, (inputs, labels) in enumerate(dl_trn):
            if torch.cuda.is_available():
                inputs = (ip.cuda() for ip in inputs)
                labels = labels.cuda()
            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_fp.log_batch(
                loss.detach().cpu(), (epoch * len(dl_trn)) + epoch_i
            )
            pbar.update()
            pbar.refresh()

        val_acc, val_loss = val_fp.validate_model(
            model, criterion, dl_val, (epoch + 1) * len(dl_trn)
        )
        trn_acc, trn_loss = val_fp.validate_model(
            model, criterion, dl_val_trn, (epoch + 1) * len(dl_trn), "trn"
        )

        scheduler.step()

        if int(epoch + 1) % save_freq == 0:
            util.save_checkpoint(
                {
                    "epoch": int(epoch + 1),
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "opt": opt.state_dict(),
                    "focus_offset": focus_offset,
                    "chunk_context": chunk_context,
                    "fixed_seq_len_chunks": fixed_seq_len_chunks,
                    "mod_motif": mod_motif,
                    "base_pred": base_pred,
                },
                out_path,
            )
        ebar.set_postfix(
            acc_val=f"{val_acc:.4f}",
            acc_train=f"{trn_acc:.4f}",
            loss_val=f"{val_loss:.6f}",
            loss_train=f"{trn_loss:.6f}",
        )
        ebar.update()


if __name__ == "__main__":
    NotImplementedError("This is a module.")
