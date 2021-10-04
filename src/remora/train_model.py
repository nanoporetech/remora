import atexit
import os
from shutil import copyfile

import numpy as np
import torch
from tqdm import tqdm

from remora import constants, util, log, RemoraError
from remora.data_chunks import load_taiyaki_dataset, load_chunks, RemoraDataset

LOGGER = log.get_logger()


def load_optimizer(optimizer, model, lr, weight_decay, momentum=0.9):
    if optimizer == constants.SGD_OPT:
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True,
        )
    elif optimizer == constants.ADAM_OPT:
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer == constants.ADAMW_OPT:
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    raise RemoraError(f"Invalid optimizer specified ({optimizer})")


def train_model(
    seed,
    device,
    out_path,
    taiyaki_dataset_path,
    remora_dataset_path,
    num_chunks,
    motif,
    focus_offset,
    chunk_context,
    val_prop,
    batch_size,
    model_path,
    size,
    mod_bases,
    base_pred,
    optimizer,
    lr,
    weight_decay,
    lr_decay_step,
    lr_decay_gamma,
    epochs,
    save_freq,
    kmer_context_bases,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(device)
    elif device is not None:
        LOGGER.warning(
            "Device option specified, but CUDA is not available from torch."
        )

    if taiyaki_dataset_path is None and remora_dataset_path is None:
        raise RemoraError("Must specify either Taiyaki or Remora dataset.")
    elif taiyaki_dataset_path is not None and remora_dataset_path is not None:
        raise RemoraError("Must specify only one of Taiyaki or Remora dataset.")
    if remora_dataset_path is not None:
        LOGGER.info("Loading dataset from Remora file")
        dataset = RemoraDataset.load_from_file(
            remora_dataset_path,
            batch_size=batch_size,
        )
        # load attributes from file
        base_pred = dataset.base_pred
        mod_bases = dataset.mod_bases
        # TODO allow modification of applicable attributes
        # namely, kmer_context_bases, chunk_context can be shrunk from versions
        # in file
        kmer_context_bases = dataset.kmer_context_bases
        chunk_context = dataset.chunk_context
        LOGGER.info(
            "Loaded data info from file:\n"
            f"          base_pred : {base_pred}\n"
            f"          mod_bases : {mod_bases}\n"
            f" kmer_context_bases : {kmer_context_bases}\n"
            f"      chunk_context : {chunk_context}\n"
        )

    val_fp = util.ValidationLogger(out_path, base_pred)
    atexit.register(val_fp.close)
    batch_fp = util.BatchLogger(out_path)
    atexit.register(batch_fp.close)

    if not base_pred and mod_bases is None:
        raise RemoraError(
            "Must specify either modified base or base prediction model "
            "type option."
        )
    elif base_pred and mod_bases is not None:
        raise RemoraError(
            "Must specify either modified base or base prediction model "
            "type option not both."
        )

    LOGGER.info("Loading model")
    copy_model_path = util.resolve_path(os.path.join(out_path, "model.py"))
    copyfile(model_path, copy_model_path)
    num_out = 4 if base_pred else len(mod_bases) + 1
    model_params = {
        "size": size,
        "kmer_len": sum(kmer_context_bases) + 1,
        "num_out": num_out,
    }
    model = util._load_python_model(copy_model_path, **model_params)
    LOGGER.info(f"Model structure:\n{model}")

    LOGGER.info("Preparing training settings")
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    opt = load_optimizer(optimizer, model, lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=lr_decay_step, gamma=lr_decay_gamma
    )

    if taiyaki_dataset_path is not None:
        LOGGER.info("Loading training data Taiyaki file")
        reads, alphabet, collapse_alphabet = load_taiyaki_dataset(
            taiyaki_dataset_path
        )
        if base_pred:
            if alphabet != "ACGT":
                raise ValueError(
                    "Base prediction is not compatible with modified base "
                    "training data. It requires a canonical alphabet."
                )
            label_conv = np.arange(4)
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
        chunks = load_chunks(
            reads,
            motif=motif,
            label_conv=label_conv,
            num_chunks=num_chunks,
            chunk_context=chunk_context,
            kmer_context_bases=kmer_context_bases,
            focus_offset=focus_offset,
            fixed_seq_len_chunks=model._variable_width_possible,
            base_pred=base_pred,
        )
        dataset = RemoraDataset.load_from_chunks(
            chunks,
            batch_size=batch_size,
            chunk_context=chunk_context,
            kmer_context_bases=kmer_context_bases,
            mod_bases=mod_bases,
            base_pred=base_pred,
        )
        del chunks
        # shuffle data before storage for fixed validation chunks on restart
        dataset.shuffle_data()
        dataset.save_dataset(
            os.path.join(out_path, constants.SAVE_DATASET_FILENAME)
        )

    label_counts = dataset.get_label_counts()
    LOGGER.info(f"Label distribution: {label_counts}")
    if len(label_counts) <= 1:
        raise RemoraError(
            "One or fewer output labels found. Ensure --focus-offset and "
            "--mod are specified correctly"
        )
    trn_ds, val_trn_ds, val_ds = dataset.split_data(val_prop)

    LOGGER.info("Running initial validation")
    # assess accuracy before first iteration
    val_acc, val_loss = val_fp.validate_model(model, criterion, val_ds, 0)
    trn_acc, trn_loss = val_fp.validate_model(
        model, criterion, val_trn_ds, 0, "trn"
    )

    LOGGER.info("Start training")
    ebar = tqdm(
        total=epochs,
        smoothing=0,
        desc="Epochs",
        dynamic_ncols=True,
        position=0,
        leave=True,
    )
    pbar = tqdm(
        total=len(trn_ds),
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
    atexit.register(pbar.close)
    atexit.register(ebar.close)
    ckpt_save_data = {
        "epoch": 0,
        "state_dict": model.state_dict(),
        "opt": opt.state_dict(),
        "model_path": copy_model_path,
        "model_params": model_params,
        "chunk_context": chunk_context,
        "fixed_seq_len_chunks": model._variable_width_possible,
        "motif": motif.to_tuple(),
        "mod_bases": mod_bases,
        "base_pred": base_pred,
        "kmer_context_bases": kmer_context_bases,
    }
    for epoch in range(epochs):
        model.train()
        pbar.n = 0
        pbar.refresh()
        for epoch_i, (inputs, labels, _) in enumerate(trn_ds):
            if torch.cuda.is_available():
                inputs = (ip.cuda() for ip in inputs)
                labels = labels.cuda()
            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_fp.log_batch(
                loss.detach().cpu(), (epoch * len(trn_ds)) + epoch_i
            )
            pbar.update()
            pbar.refresh()

        val_acc, val_loss = val_fp.validate_model(
            model, criterion, val_ds, (epoch + 1) * len(trn_ds)
        )
        trn_acc, trn_loss = val_fp.validate_model(
            model, criterion, val_trn_ds, (epoch + 1) * len(trn_ds), "trn"
        )

        scheduler.step()

        if int(epoch + 1) % save_freq == 0:
            ckpt_save_data["epoch"] = epoch + 1
            ckpt_save_data["state_dict"] = model.state_dict()
            ckpt_save_data["opt"] = opt.state_dict()
            torch.save(
                ckpt_save_data,
                os.path.join(out_path, f"model_{epoch + 1:06d}.checkpoint"),
            )
        ebar.set_postfix(
            acc_val=f"{val_acc:.4f}",
            acc_train=f"{trn_acc:.4f}",
            loss_val=f"{val_loss:.6f}",
            loss_train=f"{trn_loss:.6f}",
        )
        ebar.update()
    ebar.close()
    pbar.close()
    LOGGER.info("Saving final model checkpoint")
    ckpt_save_data["epoch"] = epoch + 1
    ckpt_save_data["state_dict"] = model.state_dict()
    ckpt_save_data["opt"] = opt.state_dict()
    torch.save(
        ckpt_save_data,
        os.path.join(out_path, constants.FINAL_MODEL_FILENAME),
    )
    LOGGER.info("Training complete")


if __name__ == "__main__":
    NotImplementedError("This is a module.")
