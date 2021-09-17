import atexit

import numpy as np
import torch
from tqdm import tqdm

from remora import models, util, log, RemoraError
from remora.data_chunks import load_datasets

LOGGER = log.get_logger()


def train_model(
    seed,
    device,
    out_path,
    dataset_path,
    num_chunks,
    mod,
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
):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.cuda.set_device(device)

    val_fp = util.ValidationLogger(out_path, base_pred)
    atexit.register(val_fp.close)

    dl_trn, dl_val, dl_val_trn = load_datasets(
        dataset_path,
        focus_offset,
        chunk_context,
        batch_size,
        num_chunks,
        fixed_seq_len_chunks,
        mod,
        base_pred,
        val_prop,
        num_data_workers,
    )

    num_out = 4 if base_pred else 2
    if model_name == "lstm":
        if fixed_seq_len_chunks:
            model = models.SimpleLSTM(size=size, num_out=num_out)
        else:
            model = models.SimpleFWLSTM(size=size, num_out=num_out)
    elif model_name == "cnn":
        if fixed_seq_len_chunks:
            raise RemoraError(
                "Convolutional network not compatible with variable signal "
                "length chunks."
            )
        model = models.double_headed_ConvLSTM(channel_size=size)
    else:
        raise ValueError("Specify a valid model type to train with")
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
    }
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

    # assess accuracy before first iteration
    val_acc = val_fp.validate_model(model, dl_val, 0)
    trn_acc = val_fp.validate_model(model, dl_val_trn, 0, "train")
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
    ebar.set_postfix(acc=f"{val_acc:.4f}", acc_train=f"{trn_acc:.4f}")
    for epoch in range(start_epoch, epochs):
        model.train()
        pbar.n = 0
        pbar.refresh()
        losses = []
        for inputs, labels in dl_trn:
            if torch.cuda.is_available():
                inputs = (ip.cuda() for ip in inputs)
                labels = labels.cuda()
            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.detach().cpu().numpy())
            pbar.update()
            pbar.refresh()

        val_acc = val_fp.validate_model(
            model, dl_val, (epoch + 1) * len(dl_trn)
        )
        trn_acc = val_fp.validate_model(
            model, dl_val_trn, (epoch + 1) * len(dl_trn), "train"
        )

        scheduler.step()

        if int(epoch + 1) % save_freq == 0:
            util.save_checkpoint(
                {
                    "epoch": int(epoch + 1),
                    "model_name": model_name,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "focus_offset": focus_offset,
                },
                out_path,
            )
        ebar.set_postfix(
            loss=f"{loss:.6f}", acc=f"{val_acc:.4f}", acc_train=f"{trn_acc:.4f}"
        )
        ebar.update()
    ebar.close()
    pbar.close()


if __name__ == "__main__":
    NotImplementedError("This is a module.")
