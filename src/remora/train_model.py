import os
import json
import atexit
from shutil import copyfile
from itertools import islice

import torch
import numpy as np
from tqdm import tqdm
from thop import profile
from torch.utils.data import DataLoader

from remora.data_chunks import (
    load_dataset,
    dataloader_worker_init,
)
from remora import (
    constants,
    util,
    log,
    RemoraError,
    model_util,
    validate,
)

LOGGER = log.get_logger()
BREACH_THRESHOLD = 0.8
REGRESSION_THRESHOLD = 0.7


def med_mad(
    data, factor=constants.PA_TO_NORM_SCALING_FACTOR, axis=None, keepdims=False
):
    """Compute the Median Absolute Deviation, i.e., the median
    of the absolute deviations from the median, and the median

    Args:
        data (:class:`ndarray`): Data from which to determine med/mad
        factor (float): Factor to scale MAD by. Default (None) is to be
            consistent with the standard deviation of a normal distribution
            (i.e. mad( N(0,sigma^2) ) = sigma).
        axis (int): For multidimensional arrays, which axis to calculate over
        keepdims (bool): If True, axis is kept as dimension of length 1

    :returns: a tuple containing the median and MAD of the data
    """
    dmed = np.median(data, axis=axis, keepdims=True)
    dmad = factor * np.median(abs(data - dmed), axis=axis, keepdims=True)
    if axis is None:
        dmed = dmed.flatten()[0]
        dmad = dmad.flatten()[0]
    elif not keepdims:
        dmed = dmed.squeeze(axis)
        dmad = dmad.squeeze(axis)
    return dmed, dmad


class RollingMAD:
    """Calculate rolling meadian absolute deviation cap over a specified
    window for a vector of values

    For example compute gradient maxima by:

        parameters = [p for p in network.parameters() if p.requires_grad]
        rolling_mads = RollingMAD(len(parameters))
        grad_maxs = [
            float(torch.max(torch.abs(layer_params.grad.detach())))
            for layer_params in parameters]
        grad_max_threshs = rolling_mads.update(grad_maxs)
    """

    def __init__(self, nparams, n_mads=0, window=1000, default_to=None):
        """Set up rolling MAD calculator.

        Args:
            nparams : Number of parameter arrays to track independently
            n_mads : Number of MADs above the median to return
            window : calculation is done over the last <window> data points
            default_to : Return this value before window values have been added
        """
        self.n_mads = n_mads
        self.default_to = default_to
        self._window_data = np.empty((nparams, window), dtype="f4")
        self._curr_iter = 0

    @property
    def nparams(self):
        return self._window_data.shape[0]

    @property
    def window(self):
        return self._window_data.shape[1]

    def update(self, vals):
        """Update with time series values and return MAD thresholds.

        Returns:
            List of `median + (nmods * mad)` of the current window of data.
                Before window number of values have been added via update the
                `default_to` value is returned.
        """
        assert len(vals) == self.nparams, (
            f"Number of values ({len(vals)}) provided does not match number of "
            f"parameters ({self.nparams})."
        )

        self._window_data[:, self._curr_iter % self.window] = vals
        self._curr_iter += 1
        if self._curr_iter < self.window:
            return self.default_to

        med, mad = med_mad(self._window_data, axis=1)
        return med + (mad * self.n_mads)


def apply_clipping(model, grad_max_threshs):
    parameters = [p for p in model.parameters() if p.requires_grad]
    grad_maxs = [
        float(torch.max(torch.abs(param_group.grad.detach())))
        for param_group in parameters
    ]
    if grad_max_threshs is not None:
        for grp_gm, grp_gmt, grp_params in zip(
            grad_maxs, grad_max_threshs, parameters
        ):
            if grp_gm > grp_gmt:
                # clip norm by value
                grp_params.grad.data.clamp_(min=-grp_gmt, max=grp_gmt)
    return grad_maxs


def save_model(
    model,
    ckpt_save_data,
    out_path,
    epoch,
    opt,
    model_name=constants.BEST_MODEL_FILENAME,
    as_torchscript=True,
    model_name_torchscript=constants.BEST_TORCHSCRIPT_MODEL_FILENAME,
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
    if as_torchscript:
        model_util.export_model_torchscript(
            ckpt_save_data,
            model,
            os.path.join(out_path, model_name_torchscript),
        )


def train_model(
    seed,
    device,
    out_path,
    remora_dataset_path,
    chunk_context,
    kmer_context_bases,
    batch_size,
    model_path,
    size,
    train_opts,
    chunks_per_epoch,
    num_test_chunks,
    save_freq,
    filt_frac,
    ext_val,
    ext_val_names,
    high_conf_incorrect_thr_frac,
    finetune_path,
    freeze_num_layers,
    super_batch_size,
    super_batch_sample_frac,
    read_batches_from_disk,
    gradient_clip_num_mads,
):
    seed = (
        np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32)
        if seed is None
        else seed
    )
    LOGGER.info(f"Seed selected is {seed}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(device)

    LOGGER.info("Loading dataset from Remora dataset config")
    # don't load extra arrays for training
    override_metadata = {"extra_metadata_arrays": {"modbase_label"}}
    if kmer_context_bases is not None:
        override_metadata["kmer_context_bases"] = kmer_context_bases
    if chunk_context is not None:
        override_metadata["chunk_context"] = chunk_context
    dataset = load_dataset(
        remora_dataset_path,
        core_ds_kwargs={"override_metadata": override_metadata},
        ds_kwargs={
            "batch_size": batch_size,
            "super_batch_size": super_batch_size,
            "super_batch_sample_frac": super_batch_sample_frac
            if super_batch_sample_frac < 1.0
            else None,
            "return_arrays": ["signal", "modbase_label", "enc_kmer"],
        },
    )
    # TODO move hash computation into background worker and write this from
    # that worker as well. This command stalls startup too much
    with open(os.path.join(out_path, "dataset_config.jsn"), "w") as ds_cfg_fh:
        json.dump(dataset.get_config(), ds_cfg_fh)
    dataset.metadata.write(os.path.join(out_path, "dataset_metadata.jsn"))
    # load attributes from file
    LOGGER.info(f"Dataset summary:\n{dataset.summary}")

    val_fp = open(out_path / "validation.log", mode="w", buffering=1)
    atexit.register(val_fp.close)
    val_fp = validate.ValidationLogger(val_fp)
    batch_fp = open(out_path / "batch.log", "w", buffering=1)
    atexit.register(batch_fp.close)
    if high_conf_incorrect_thr_frac is not None:
        batch_fp.write("Iteration\tLoss\tNumberFiltered\n")
    else:
        batch_fp.write("Iteration\tLoss\n")

    LOGGER.info("Loading model")
    copy_model_path = util.resolve_path(os.path.join(out_path, "model.py"))
    copyfile(model_path, copy_model_path)
    model_params = {
        "size": size,
        "kmer_len": dataset.metadata.kmer_len,
        "num_out": dataset.metadata.num_labels,
    }
    model = model_util._load_python_model(copy_model_path, **model_params)
    LOGGER.info(f"Model structure:\n{model}")

    grad_max_threshs = None
    # TODO add to batch log
    # grad_max_thresh_str = "NaN"
    if gradient_clip_num_mads is None:
        LOGGER.debug("No gradient clipping")
        rolling_mads = None
    else:
        nparams = len([p for p in model.parameters() if p.requires_grad])
        if nparams == 0:
            rolling_mads = None
            LOGGER.warning("No gradient clipping due to missing parameters")
        else:
            rolling_mads = RollingMAD(nparams, gradient_clip_num_mads)
            LOGGER.info(
                "Gradients will be clipped (by value) at "
                f"{rolling_mads.n_mads:3.2f} MADs above the median of the "
                f"last {rolling_mads.window} gradient maximums."
            )

    if finetune_path is not None:
        ckpt, model = model_util.continue_from_checkpoint(
            finetune_path, copy_model_path
        )
        if freeze_num_layers:
            for freeze_iter, (p_name, param) in enumerate(
                model.named_parameters()
            ):
                LOGGER.debug(f"Freezing layer for training: {p_name}")
                param.requires_grad = False
                if freeze_iter >= freeze_num_layers:
                    break

        if ckpt["model_params"]["num_out"] != dataset.metadata.num_labels:
            in_feat = model.fc.in_features
            model.fc = torch.nn.Linear(in_feat, dataset.metadata.num_labels)
        if ckpt["model_params"]["size"] != size:
            LOGGER.warning(
                "Size mismatch between pretrained model and selected size. "
                "Using pretrained model size."
            )
            model_params["size"] = ckpt["model_params"]["size"]
        if dataset.metadata.chunk_context != ckpt["chunk_context"]:
            raise RemoraError(
                "The chunk context of the pre-trained model and the dataset "
                "do not match."
            )
        if dataset.metadata.kmer_context_bases != ckpt["kmer_context_bases"]:
            raise RemoraError(
                "The kmer context bases of the pre-trained model and "
                "the dataset do not match."
            )

    if ext_val is not None:
        LOGGER.info("Loading external validation data")
        if ext_val_names is None:
            ext_val_names = [f"e_val_{idx}" for idx in range(len(ext_val))]
        else:
            assert len(ext_val_names) == len(ext_val)
        ext_datasets = []
        for e_name, e_path in zip(ext_val_names, ext_val):
            ext_val_ds = load_dataset(
                e_path.strip(),
                core_ds_kwargs={
                    "override_metadata": override_metadata,
                    "infinite_iter": False,
                    "do_check_super_batches": True,
                },
                ds_kwargs={
                    "batch_size": batch_size,
                    "return_arrays": ["signal", "modbase_label", "enc_kmer"],
                },
            )
            ext_val_ds.update_metadata(dataset)
            ext_val_ds.set_use_constant_batch_mix(True)
            if not read_batches_from_disk:
                ext_val_ds.load_all_batches()
            ext_datasets.append((e_name, ext_val_ds))

    kmer_dim = int(dataset.metadata.kmer_len * 4)
    test_input_sig = torch.randn(
        batch_size, 1, sum(dataset.metadata.chunk_context)
    )
    test_input_seq = torch.randn(
        batch_size, kmer_dim, sum(dataset.metadata.chunk_context)
    )
    macs, params = profile(
        model, inputs=(test_input_sig, test_input_seq), verbose=False
    )
    LOGGER.info(
        f"Params (k) {params / (1000):.2f} | MACs (M) {macs / (1000 ** 2):.2f}"
    )

    LOGGER.info("Preparing training settings")
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    LOGGER.info(f"Training optimizer and scheduler settings: {train_opts}")
    opt = train_opts.load_optimizer(model)
    scheduler = train_opts.load_scheduler(opt)

    LOGGER.debug("Splitting dataset")
    trn_ds, val_ds = dataset.train_test_split(num_test_chunks)
    val_ds.super_batch_sample_frac = None
    val_ds.do_check_super_batches = True
    val_ds.set_use_constant_batch_mix(True)
    if not read_batches_from_disk:
        val_ds.load_all_batches()
    trn_loader = DataLoader(
        trn_ds,
        batch_size=None,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=10,
        persistent_workers=True,
        worker_init_fn=dataloader_worker_init,
    )
    LOGGER.debug("Extracting head of train dataset")
    val_trn_ds = trn_ds.head(num_test_chunks)
    val_trn_ds.super_batch_sample_frac = None
    val_trn_ds.do_check_super_batches = True
    if not read_batches_from_disk:
        val_trn_ds.load_all_batches()
    LOGGER.info(f"Dataset loaded with labels: {dataset.modbase_label_summary}")
    LOGGER.info(f"Train labels: {trn_ds.modbase_label_summary}")
    LOGGER.info(f"Held-out validation labels: {val_ds.modbase_label_summary}")
    LOGGER.info(
        f"Training set validation labels: {val_trn_ds.modbase_label_summary}"
    )

    LOGGER.info("Running initial validation")
    # assess accuracy before first iteration
    val_metrics = val_fp.validate_model(
        model, dataset.metadata.mod_bases, criterion, val_ds, filt_frac
    )
    trn_metrics = val_fp.validate_model(
        model,
        dataset.metadata.mod_bases,
        criterion,
        val_trn_ds,
        filt_frac,
        "trn",
    )
    batches_per_epoch = int(np.ceil(chunks_per_epoch / batch_size))
    epoch_summ = trn_ds.epoch_summary(batches_per_epoch)
    LOGGER.debug(f"Epoch Summary:\n{epoch_summ}")
    with open(os.path.join(out_path, "epoch_summary.txt"), "w") as summ_fh:
        summ_fh.write(epoch_summ + "\n")

    if ext_val:
        best_alt_val_accs = dict((e_name, 0) for e_name, _ in ext_datasets)
        for ext_name, ext_ds in ext_datasets:
            val_fp.validate_model(
                model,
                dataset.metadata.mod_bases,
                criterion,
                ext_ds,
                filt_frac,
                ext_name,
            )

    LOGGER.info("Start training")
    ebar = tqdm(
        total=train_opts.epochs,
        smoothing=0,
        desc="Epochs",
        dynamic_ncols=True,
        position=0,
        leave=True,
        disable=os.environ.get("LOG_SAFE", False),
    )
    pbar = tqdm(
        total=batches_per_epoch,
        desc="Epoch Progress",
        dynamic_ncols=True,
        position=1,
        leave=True,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| " "{n_fmt}/{total_fmt}",
        disable=os.environ.get("LOG_SAFE", False),
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
        "fixed_seq_len_chunks": model._variable_width_possible,
        "model_version": constants.MODEL_VERSION,
        "chunk_context": dataset.metadata.chunk_context,
        "motifs": dataset.metadata.motifs,
        "num_motifs": dataset.metadata.num_motifs,
        "reverse_signal": dataset.metadata.reverse_signal,
        "mod_bases": dataset.metadata.mod_bases,
        "mod_long_names": dataset.metadata.mod_long_names,
        "dataset_type": dataset.metadata.dataset_type,
        "kmer_context_bases": dataset.metadata.kmer_context_bases,
        "base_start_justify": dataset.metadata.base_start_justify,
        "offset": dataset.metadata.offset,
        "pa_scaling": dataset.metadata.pa_scaling,
        **dataset.metadata.sig_map_refiner.asdict(),
    }
    best_val_acc = 0
    early_stop_epochs = 0
    breached = False
    for epoch in range(train_opts.epochs):
        model.train()
        pbar.n = 0
        pbar.refresh()
        for epoch_i, (sigs, labels, enc_kmers) in enumerate(
            islice(trn_loader, batches_per_epoch)
        ):
            outputs = model(sigs.to(device), enc_kmers.to(device))
            if high_conf_incorrect_thr_frac is None:
                loss = criterion(outputs, labels.to(device))
            else:
                batch_size = outputs.shape[0]
                conf_thresh, max_frac_skip = high_conf_incorrect_thr_frac
                max_nr_skip = int(np.floor(batch_size * max_frac_skip))
                preds = (
                    torch.nn.functional.softmax(outputs, dim=1).detach().cpu()
                )
                highest_preds, high_conf_cl = torch.max(preds, dim=1)
                cl_match = labels == high_conf_cl
                # if there are more mismatches than the maximum number allowed
                # to be skipped, set confidence threshold to allow no more than
                # max_nr_skip items be skipped
                if batch_size - cl_match.sum() > max_nr_skip:
                    mm_preds = highest_preds[~cl_match]
                    mm_preds.sort(descending=True)
                    conf_thresh = max(conf_thresh, mm_preds[max_nr_skip])
                mask = cl_match.logical_or(highest_preds < conf_thresh)
                # avoid sending labels to device until after above computations
                loss = criterion(outputs[mask], labels[mask].to(device))

            opt.zero_grad()
            loss.backward()
            grad_maxs = apply_clipping(model, grad_max_threshs)
            opt.step()
            if rolling_mads is not None:
                grad_max_threshs = rolling_mads.update(grad_maxs)

            batch_fp.write(
                f"{(epoch * batches_per_epoch) + epoch_i}\t"
                f"{loss.detach().cpu():.6f}"
            )
            if high_conf_incorrect_thr_frac is None:
                batch_fp.write("\n")
            else:
                batch_fp.write(f"\t{batch_size - mask.sum()}\n")

            pbar.update()
            pbar.refresh()

        val_metrics = val_fp.validate_model(
            model,
            dataset.metadata.mod_bases,
            criterion,
            val_ds,
            filt_frac,
            nepoch=epoch + 1,
            niter=(epoch + 1) * batches_per_epoch,
            disable_pbar=True,
        )
        trn_metrics = val_fp.validate_model(
            model,
            dataset.metadata.mod_bases,
            criterion,
            val_trn_ds,
            filt_frac,
            "trn",
            nepoch=epoch + 1,
            niter=(epoch + 1) * batches_per_epoch,
            disable_pbar=True,
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
                model,
                ckpt_save_data,
                out_path,
                epoch,
                opt,
            )
        else:
            early_stop_epochs += 1

        if ext_val:
            for ext_name, ext_ds in ext_datasets:
                ext_val_metrics = val_fp.validate_model(
                    model,
                    dataset.metadata.mod_bases,
                    criterion,
                    ext_ds,
                    filt_frac,
                    ext_name,
                    nepoch=epoch + 1,
                    niter=(epoch + 1) * batches_per_epoch,
                    disable_pbar=True,
                )
                if ext_val_metrics.acc <= best_alt_val_accs[ext_name]:
                    continue
                best_alt_val_accs[ext_name] = ext_val_metrics.acc
                early_stop_epochs = 0
                LOGGER.debug(
                    f"Saving best model based on {ext_name} "
                    f"validation set after {epoch + 1} epochs "
                    f"with val_acc {ext_val_metrics.acc}"
                )
                save_model(
                    model,
                    ckpt_save_data,
                    out_path,
                    epoch,
                    opt,
                    model_name=f"model_ext_val_{ext_name}_best.checkpoint",
                    model_name_torchscript=f"model_ext_val_{ext_name}_best.pt",
                )

        if int(epoch + 1) % save_freq == 0:
            save_model(
                model,
                ckpt_save_data,
                out_path,
                epoch,
                opt,
                model_name=f"model_{epoch + 1:06d}.checkpoint",
                model_name_torchscript=f"model_{epoch + 1:06d}.pt",
            )

        ebar.set_postfix(
            acc_val=f"{val_metrics.acc:.4f}",
            acc_train=f"{trn_metrics.acc:.4f}",
            loss_val=f"{val_metrics.loss:.6f}",
            loss_train=f"{trn_metrics.loss:.6f}",
        )
        ebar.update()
        if (
            train_opts.early_stopping
            and early_stop_epochs >= train_opts.early_stopping
        ):
            break
    ebar.close()
    pbar.close()
    if (
        train_opts.early_stopping
        and early_stop_epochs >= train_opts.early_stopping
    ):
        LOGGER.info(
            "No validation accuracy improvement after "
            f"{train_opts.early_stopping} epochs. Training stopped early."
        )
    LOGGER.info("Saving final model checkpoint")
    save_model(
        model,
        ckpt_save_data,
        out_path,
        epoch,
        opt,
        model_name=constants.FINAL_MODEL_FILENAME,
        model_name_torchscript=constants.FINAL_TORCHSCRIPT_MODEL_FILENAME,
    )


if __name__ == "__main__":
    NotImplementedError("This is a module.")
