import os
import atexit
from shutil import copyfile

import torch
import numpy as np
from tqdm import tqdm
from thop import profile

from remora.data_chunks import RemoraDataset
from remora import constants, util, log, RemoraError, encoded_kmers, model_util

LOGGER = log.get_logger()
BREACH_THRESHOLD = 0.8
REGRESSION_THRESHOLD = 0.7


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


def select_scheduler(scheduler, opt, lr_sched_kwargs):
    lr_sched_dict = {}
    if lr_sched_kwargs is None and scheduler is None:
        scheduler = constants.DEFAULT_SCHEDULER
        lr_sched_dict = constants.DEFAULT_SCH_VALUES
    else:
        for lr_key, lr_val, lr_type in lr_sched_kwargs:
            lr_sched_dict[lr_key] = constants.TYPE_CONVERTERS[lr_type](lr_val)
    return getattr(torch.optim.lr_scheduler, scheduler)(opt, **lr_sched_dict)


def save_model(
    model,
    ckpt_save_data,
    out_path,
    epoch,
    opt,
    model_name=constants.BEST_MODEL_FILENAME,
    as_onnx=False,
    model_name_onnx=constants.BEST_ONNX_MODEL_FILENAME,
):
    ckpt_save_data["epoch"] = epoch + 1
    state_dict = model.state_dict()
    if "total_ops" in state_dict.keys():
        state_dict.pop("total_ops", None)
    if "total_params" in state_dict.keys():
        state_dict.pop("total_params", None)
    ckpt_save_data["state_dict"] = state_dict
    ckpt_save_data["opt"] = opt.state_dict()
    torch.save(
        ckpt_save_data,
        os.path.join(out_path, model_name),
    )
    if as_onnx:
        model_util.export_model(
            ckpt_save_data,
            model,
            os.path.join(out_path, model_name_onnx),
        )


def train_model(
    seed,
    device,
    out_path,
    remora_dataset_path,
    chunk_context,
    kmer_context_bases,
    val_prop,
    batch_size,
    model_path,
    size,
    optimizer,
    lr,
    scheduler_name,
    weight_decay,
    epochs,
    save_freq,
    early_stopping,
    conf_thr,
    ext_val,
    lr_sched_kwargs,
    balance,
):
    seed = (
        np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32)
        if seed is None
        else seed
    )
    LOGGER.info(f"Seed selected is {seed}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(device)
    elif device is not None:
        LOGGER.warning(
            "Device option specified, but CUDA is not available from torch."
        )

    LOGGER.info("Loading dataset from Remora file")
    dataset = RemoraDataset.load_from_file(
        remora_dataset_path,
        batch_size=batch_size,
    )
    LOGGER.info(f"Dataset loaded with labels: {dataset.get_label_counts()}")
    if balance:
        dataset = dataset.balance_classes()
        LOGGER.info(f"Dataset balanced: {dataset.get_label_counts()}")
    dataset.trim_kmer_context_bases(kmer_context_bases)
    dataset.trim_chunk_context(chunk_context)
    # load attributes from file
    LOGGER.info(f"Dataset summary:\n{dataset.summary}")

    out_log = out_path / "validation.log"
    val_fp = model_util.ValidationLogger(out_log)
    atexit.register(val_fp.close)
    batch_fp = util.BatchLogger(out_path)
    atexit.register(batch_fp.close)

    LOGGER.info("Loading model")
    copy_model_path = util.resolve_path(os.path.join(out_path, "model.py"))
    copyfile(model_path, copy_model_path)
    num_out = 4 if dataset.base_pred else len(dataset.mod_bases) + 1
    model_params = {
        "size": size,
        "kmer_len": sum(dataset.kmer_context_bases) + 1,
        "num_out": num_out,
    }
    model = model_util._load_python_model(copy_model_path, **model_params)
    LOGGER.info(f"Model structure:\n{model}")

    if ext_val:
        ext_sets = []
        for path in ext_val:
            ext_val_set = RemoraDataset.load_from_file(
                path.strip(),
                batch_size=batch_size,
                shuffle_on_iter=False,
                drop_last=False,
            )
            if ext_val_set.mod_long_names != dataset.mod_long_names:
                ext_val_set.add_fake_base(
                    dataset.mod_long_names, dataset.mod_bases
                )
            ext_val_set.trim_kmer_context_bases(kmer_context_bases)
            ext_val_set.trim_chunk_context(chunk_context)
            ext_sets.append(ext_val_set)

    kmer_dim = int((sum(dataset.kmer_context_bases) + 1) * 4)
    test_input_sig = torch.randn(batch_size, 1, sum(dataset.chunk_context))
    test_input_seq = torch.randn(
        batch_size, kmer_dim, sum(dataset.chunk_context)
    )
    macs, params = profile(
        model, inputs=(test_input_sig, test_input_seq), verbose=False
    )
    LOGGER.info(
        f" Params (k) {params / (1000):.2f} | MACs (M) {macs / (1000 ** 2):.2f}"
    )

    LOGGER.info("Preparing training settings")
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    opt = load_optimizer(optimizer, model, lr, weight_decay)

    scheduler = select_scheduler(scheduler_name, opt, lr_sched_kwargs)

    label_counts = dataset.get_label_counts()
    LOGGER.info(f"Label distribution: {label_counts}")
    if len(label_counts) <= 1:
        raise RemoraError(
            "One or fewer output labels found. Ensure --focus-offset and "
            "--mod are specified correctly"
        )

    trn_ds, val_ds = dataset.split_data(val_prop, stratified=True)
    trn_ds.shuffle()
    val_trn_ds = trn_ds.head(val_prop, shuffle_on_iter=False, drop_last=False)
    LOGGER.info(f"Train label distribution: {trn_ds.get_label_counts()}")
    LOGGER.info(
        f"Held-out validation label distribution: {val_ds.get_label_counts()}"
    )
    LOGGER.info(
        "Training set validation label distribution: "
        f"{val_trn_ds.get_label_counts()}"
    )

    LOGGER.info("Running initial validation")
    # assess accuracy before first iteration
    val_metrics = val_fp.validate_model(
        model, dataset.mod_bases, criterion, val_ds, conf_thr
    )
    trn_metrics = val_fp.validate_model(
        model,
        dataset.mod_bases,
        criterion,
        val_trn_ds,
        conf_thr,
        "trn",
    )

    if ext_val:
        best_alt_val_accs = [0] * len(ext_sets)
        for e_set_idx in range(len(ext_sets)):
            val_fp.validate_model(
                model,
                dataset.mod_bases,
                criterion,
                ext_sets[e_set_idx],
                conf_thr,
                f"e_val_{e_set_idx}",
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
        acc_val=f"{val_metrics.acc:.4f}",
        acc_train=f"{trn_metrics.acc:.4f}",
        loss_val=f"{val_metrics.loss:.6f}",
        loss_train=f"{trn_metrics.loss:.6f}",
    )
    atexit.register(pbar.close)
    atexit.register(ebar.close)

    ckpt_save_data = {
        "epoch": 0,
        "state_dict": model.state_dict(),
        "opt": opt.state_dict(),
        "model_path": copy_model_path,
        "model_params": model_params,
        "chunk_context": dataset.chunk_context,
        "fixed_seq_len_chunks": model._variable_width_possible,
        "motifs": dataset.motifs,
        "num_motifs": dataset.num_motifs,
        "mod_bases": dataset.mod_bases,
        "mod_long_names": dataset.mod_long_names,
        "base_pred": dataset.base_pred,
        "kmer_context_bases": dataset.kmer_context_bases,
        "base_start_justify": dataset.base_start_justify,
        "offset": dataset.offset,
        "model_version": constants.MODEL_VERSION,
        **dataset.sig_map_refiner.get_save_kwargs(),
    }
    bb, ab = dataset.kmer_context_bases
    best_val_acc = 0
    early_stop_epochs = 0
    breached = False
    for epoch in range(epochs):
        model.train()
        pbar.n = 0
        pbar.refresh()
        for epoch_i, ((sigs, seqs, seq_maps, seq_lens), labels, _) in enumerate(
            trn_ds
        ):
            sigs = torch.from_numpy(sigs)
            enc_kmers = torch.from_numpy(
                encoded_kmers.compute_encoded_kmer_batch(
                    bb, ab, seqs, seq_maps, seq_lens
                )
            )
            labels = torch.from_numpy(labels)
            if torch.cuda.is_available():
                sigs = sigs.cuda()
                enc_kmers = enc_kmers.cuda()
                labels = labels.cuda()
            outputs = model(sigs, enc_kmers)
            loss = criterion(outputs, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_fp.log_batch(
                loss.detach().cpu(), (epoch * len(trn_ds)) + epoch_i
            )
            pbar.update()
            pbar.refresh()

        niter = (epoch + 1) * len(trn_ds)
        val_metrics = val_fp.validate_model(
            model,
            dataset.mod_bases,
            criterion,
            val_ds,
            conf_thr,
            nepoch=epoch + 1,
            niter=niter,
        )
        trn_metrics = val_fp.validate_model(
            model,
            dataset.mod_bases,
            criterion,
            val_trn_ds,
            conf_thr,
            "trn",
            nepoch=epoch + 1,
            niter=niter,
        )

        scheduler.step()

        if breached:
            if val_metrics.acc <= REGRESSION_THRESHOLD:
                LOGGER.warning("Remora training unstable")
        else:
            if val_metrics.acc >= BREACH_THRESHOLD:
                breached = True
                LOGGER.debug(
                    f"{BREACH_THRESHOLD * 100}% accuracy threshold surpassed"
                )

        if val_metrics.acc > best_val_acc:
            best_val_acc = val_metrics.acc
            early_stop_epochs = 0
            LOGGER.debug(
                f"Saving best model after {epoch + 1} epochs with "
                f"val_acc {val_metrics.acc}"
            )
            save_model(
                model, ckpt_save_data, out_path, epoch, opt, as_onnx=True
            )
        else:
            early_stop_epochs += 1

        if ext_val:
            for e_set_idx in range(len(ext_sets)):
                e_val_metrics = val_fp.validate_model(
                    model,
                    dataset.mod_bases,
                    criterion,
                    ext_sets[e_set_idx],
                    conf_thr,
                    f"e_val_{e_set_idx}",
                    nepoch=epoch + 1,
                    niter=niter,
                )
                if e_val_metrics.acc > best_alt_val_accs[e_set_idx]:
                    best_alt_val_accs[e_set_idx] = e_val_metrics.acc
                    LOGGER.debug(
                        f"Saving best model based on e_val_{e_set_idx} "
                        f"validation sets after {epoch + 1} epochs "
                        f"with val_acc {val_metrics.acc}"
                    )
                    save_model(
                        model,
                        ckpt_save_data,
                        out_path,
                        epoch,
                        opt,
                        model_name=f"model_e_val_{e_set_idx}_best.checkpoint",
                        as_onnx=True,
                        model_name_onnx=f"model_e_val_{e_set_idx}_best.onnx",
                    )

        if int(epoch + 1) % save_freq == 0:
            save_model(
                model,
                ckpt_save_data,
                out_path,
                epoch,
                opt,
                f"model_{epoch + 1:06d}.checkpoint",
            )

        ebar.set_postfix(
            acc_val=f"{val_metrics.acc:.4f}",
            acc_train=f"{trn_metrics.acc:.4f}",
            loss_val=f"{val_metrics.loss:.6f}",
            loss_train=f"{trn_metrics.loss:.6f}",
        )
        ebar.update()
        if early_stopping and early_stop_epochs >= early_stopping:
            LOGGER.info(
                "No validation accuracy improvement after"
                f" {early_stopping} epoch(s). Stopping training early."
            )
            break
    ebar.close()
    pbar.close()
    LOGGER.info("Saving final model checkpoint")
    save_model(
        model,
        ckpt_save_data,
        out_path,
        epoch,
        opt,
        constants.FINAL_MODEL_FILENAME,
        True,
        constants.FINAL_ONNX_MODEL_FILENAME,
    )
    LOGGER.info("Training complete")


if __name__ == "__main__":
    NotImplementedError("This is a module.")
