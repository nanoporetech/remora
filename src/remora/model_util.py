import os
import sys
import copy
import datetime
import importlib
import pkg_resources
from os.path import isfile
from collections import namedtuple

import onnx
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
import onnxruntime as ort
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report

from remora.refine_signal_map import SigMapRefiner
from remora import log, RemoraError, encoded_kmers, util, constants

LOGGER = log.get_logger()
MODEL_DATA_DIR_NAME = "trained_models"
ONNX_NUM_THREADS = 1

VAL_METRICS = namedtuple(
    "VAL_METRICS",
    (
        "loss",
        "acc",
        "min_f1",
        "f1",
        "prec",
        "recall",
        "num_calls",
        "conf_mat",
        "filt_frac",
        "filt_conf_mat",
    ),
)


def compute_metrics(
    all_outputs,
    all_labels,
    all_loss,
    conf_thr,
):
    mean_loss = np.mean(all_loss)
    num_calls = all_labels.size
    pred_labels = np.argmax(all_outputs, axis=1)
    conf_mat = confusion_matrix(all_labels, pred_labels)
    acc = (pred_labels == all_labels).sum() / num_calls
    cl_rep = classification_report(
        all_labels,
        pred_labels,
        digits=3,
        output_dict=True,
        zero_division=0,
    )
    min_f1 = min(cl_rep[str(k)]["f1-score"] for k in np.unique(all_labels))

    prec = recall = f1 = np.nan
    uniq_labs = np.unique(all_labels)
    if uniq_labs.size == 2:
        pos_idx = uniq_labs[1]
        with np.errstate(invalid="ignore"):
            precision, recall, thresholds = precision_recall_curve(
                all_labels, all_outputs[:, pos_idx], pos_label=pos_idx
            )
            f1_scores = 2 * recall * precision / (recall + precision)
        f1_idx = np.argmax(f1_scores)
        prec = precision[f1_idx]
        recall = recall[f1_idx]
        f1 = f1_scores[f1_idx]

    # metrics involving confidence thresholding
    all_probs = util.softmax_axis1(all_outputs)
    conf_chunks = np.max(all_probs > conf_thr, axis=1)
    filt_conf_mat = confusion_matrix(
        all_labels[conf_chunks], np.argmax(all_probs[conf_chunks], axis=1)
    )
    filt_frac = np.logical_not(conf_chunks).sum() / num_calls

    return VAL_METRICS(
        loss=mean_loss,
        acc=acc,
        min_f1=min_f1,
        f1=f1,
        prec=prec,
        recall=recall,
        num_calls=num_calls,
        conf_mat=conf_mat,
        filt_frac=filt_frac,
        filt_conf_mat=conf_mat,
    )


def get_label_coverter(from_labels, to_labels, base_pred):
    if base_pred:
        return np.arange(4)
    if not set(from_labels).issubset(to_labels):
        raise RemoraError(
            f"Cannot convert from superset of labels ({from_labels}) "
            f"to a subset ({to_labels})."
        )
    return np.array([0] + [to_labels.find(mb) + 1 for mb in from_labels])


def validate_model(
    model,
    model_mod_bases,
    criterion,
    dataset,
    conf_thr=constants.DEFAULT_CONF_THR,
    display_progress_bar=True,
):
    label_conv = get_label_coverter(
        dataset.mod_bases, model_mod_bases, dataset.base_pred
    )
    is_torch_model = isinstance(model, nn.Module)
    if is_torch_model:
        model.eval()
        torch.set_grad_enabled(False)

    bb, ab = dataset.kmer_context_bases
    all_labels = []
    all_outputs = []
    all_loss = []
    ds_iter = (
        tqdm(dataset, smoothing=0, desc="Batches", leave=False)
        if display_progress_bar
        else dataset
    )
    for (sigs, seqs, seq_maps, seq_lens), labels, _ in ds_iter:
        model_labels = label_conv[labels]
        all_labels.append(model_labels)
        enc_kmers = encoded_kmers.compute_encoded_kmer_batch(
            bb, ab, seqs, seq_maps, seq_lens
        )
        if is_torch_model:
            sigs = torch.from_numpy(sigs)
            enc_kmers = torch.from_numpy(enc_kmers)
            if torch.cuda.is_available():
                sigs = sigs.cuda()
                enc_kmers = enc_kmers.cuda()
            output = model(sigs, enc_kmers).detach().cpu().numpy()
        else:
            output = model.run([], {"sig": sigs, "seq": enc_kmers})[0]
        all_outputs.append(output)
        all_loss.append(
            criterion(torch.from_numpy(output), torch.from_numpy(model_labels))
            .detach()
            .cpu()
            .numpy()
        )
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels)
    if is_torch_model:
        torch.set_grad_enabled(True)
    return compute_metrics(all_outputs, all_labels, all_loss, conf_thr)


class ValidationLogger:
    def __init__(self, out_path):
        self.fp = open(out_path, "w", buffering=1)
        self.fp.write(
            "\t".join(
                (
                    "Val_Type",
                    "Epoch",
                    "Iteration",
                    "Accuracy",
                    "Loss",
                    "Min_F1",
                    "F1",
                    "Precision",
                    "Recall",
                    "Num_Calls",
                    "Confusion_Matrix",
                    "Filtered_Fraction",
                    "Filtered_Confusion_Matrix",
                )
            )
            + "\n"
        )

    def close(self):
        self.fp.close()

    def validate_model(
        self,
        model,
        model_mod_bases,
        criterion,
        dataset,
        conf_thr=constants.DEFAULT_CONF_THR,
        val_type="val",
        nepoch=0,
        niter=0,
        display_progress_bar=False,
    ):
        ms = validate_model(
            model,
            model_mod_bases,
            criterion,
            dataset,
            conf_thr,
            display_progress_bar=display_progress_bar,
        )
        cm_str = np.array2string(ms.conf_mat.flatten(), separator=",").replace(
            "\n", ""
        )
        fcm_str = np.array2string(
            ms.filt_conf_mat.flatten(), separator=","
        ).replace("\n", "")
        self.fp.write(
            f"{val_type}\t{nepoch}\t{niter}\t{ms.acc:.6f}\t{ms.loss:.6f}\t"
            f"{ms.min_f1:.6f}\t{ms.f1:.6f}\t{ms.prec:.6f}\t{ms.recall:.6f}\t"
            f"{ms.num_calls}\t{cm_str}\t{ms.filt_frac:.4f}\t{fcm_str}\n"
        )
        return ms


def _load_python_model(model_file, **model_kwargs):
    loader = importlib.machinery.SourceFileLoader("netmodule", model_file)
    netmodule = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(loader.name, loader)
    )
    loader.exec_module(netmodule)
    network = netmodule.network(**model_kwargs)
    return network


def continue_from_checkpoint(ckp_path, model_path=None):
    """Load a checkpoint in order to continue training."""
    if not isfile(ckp_path):
        raise RemoraError(f"Checkpoint path is not a file ({ckp_path})")
    LOGGER.info(f"Loading trained model from {ckp_path}")
    ckpt = torch.load(ckp_path)
    if ckpt["state_dict"] is None:
        raise RemoraError("Model state not saved in checkpoint.")

    model_path = ckpt["model_path"] if model_path is None else model_path
    model = _load_python_model(model_path, **ckpt["model_params"])
    model.load_state_dict(ckpt["state_dict"])
    return ckpt, model


def export_model(ckpt, model, save_filename):
    kmer_len = sum(ckpt["kmer_context_bases"]) + 1
    sig_len = sum(ckpt["chunk_context"])
    sig = torch.from_numpy(np.zeros((1, 1, sig_len), dtype=np.float32))
    seq = torch.from_numpy(
        np.zeros((1, kmer_len * 4, sig_len), dtype=np.float32)
    )
    model.eval()
    if next(model.parameters()).is_cuda:
        model_to_save = copy.deepcopy(model).cpu()
    else:
        model_to_save = model
    with torch.no_grad():
        torch.onnx.export(
            model_to_save,
            (sig, seq),
            save_filename,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=["sig", "seq"],
            output_names=["output"],
            dynamic_axes={
                "sig": {0: "batch_size"},
                "seq": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
    onnx_model = onnx.load(save_filename)
    meta = onnx_model.metadata_props.add()
    meta.key = "creation_date"
    meta.value = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    # add simple metadata
    for ckpt_key in (
        "base_pred",
        "mod_bases",
        "refine_kmer_center_idx",
        "refine_do_rough_rescale",
        "refine_scale_iters",
        "refine_algo",
        "refine_half_bandwidth",
        "base_start_justify",
        "offset",
    ):
        meta = onnx_model.metadata_props.add()
        meta.key = ckpt_key
        meta.value = str(ckpt[ckpt_key])

    if ckpt["mod_bases"] is not None:
        for mod_idx in range(len(ckpt["mod_bases"])):
            meta = onnx_model.metadata_props.add()
            meta.key = f"mod_long_names_{mod_idx}"
            meta.value = str(ckpt["mod_long_names"][mod_idx])
    for key in ("chunk_context", "kmer_context_bases"):
        for idx in range(2):
            meta = onnx_model.metadata_props.add()
            meta.key = f"{key}_{idx}"
            meta.value = str(ckpt[key][idx])

    meta = onnx_model.metadata_props.add()
    meta.key = "num_motifs"
    meta.value = str(ckpt["num_motifs"])
    for idx, (motif, motif_offset) in enumerate(ckpt["motifs"]):
        m_key = f"motif_{idx}"
        mo_key = f"motif_offset_{idx}"
        meta = onnx_model.metadata_props.add()
        meta.key = m_key
        meta.value = str(motif)
        meta = onnx_model.metadata_props.add()
        meta.key = mo_key
        meta.value = str(motif_offset)

    meta = onnx_model.metadata_props.add()
    meta.key = "num_motifs"
    meta.value = str(len(ckpt["motifs"]))

    # store refine arrays as bytes
    meta = onnx_model.metadata_props.add()
    meta.key = "refine_kmer_levels"
    meta.value = (
        ckpt["refine_kmer_levels"].astype(np.float32).tobytes().decode("cp437")
    )
    meta = onnx_model.metadata_props.add()
    meta.key = "refine_sd_arr"
    meta.value = (
        ckpt["refine_sd_arr"].astype(np.float32).tobytes().decode("cp437")
    )

    onnx_model.doc_string = "Nanopore Remora model"
    try:
        onnx_model.model_version = ckpt["model_version"]
    except KeyError:
        LOGGER.warning("Model version not found in checkpoint. Setting to 0.")
        onnx_model.model_version = 0
    onnx.save(onnx_model, save_filename)


def load_onnx_model(model_filename, device=None, quiet=False):
    """Load onnx model. If device is specified load onto specified device.

    Args:
        model_filename (str): Model path
        device (int): GPU device ID
        quiet (bool): Don't log full model loading info

    Returns:
        2-tuple containing:
          1. ort.InferenceSession object for calling mods
          2. Model metadata dictionary with information concerning data prep
    """
    providers = ["CPUExecutionProvider"]
    provider_options = None
    if device is not None:
        if quiet:
            LOGGER.debug("Loading Remora model onto GPU")
        else:
            LOGGER.info("Loading Remora model onto GPU")
        if ort.get_device() != "GPU":
            raise RemoraError(
                "onnxruntime not compatible with GPU execution. Install "
                "compatible package via `pip install onnxruntime-gpu`"
            )
        providers = ["CUDAExecutionProvider"]
        provider_options = [{"device_id": str(device)}]
    # set severity to error so CPU fallback messages are masked
    ort.set_default_logger_severity(3)
    LOGGER.debug(f"Using {ONNX_NUM_THREADS} thread(s) for ONNX")
    so = ort.SessionOptions()
    so.inter_op_num_threads = ONNX_NUM_THREADS
    so.intra_op_num_threads = ONNX_NUM_THREADS
    model_sess = ort.InferenceSession(
        model_filename,
        providers=providers,
        provider_options=provider_options,
        sess_options=so,
    )
    LOGGER.debug(f"Remora model ONNX providers: {model_sess.get_providers()}")
    if device is not None and model_sess.get_providers()[0].startswith("CPU"):
        raise RemoraError(
            "Model not loaded on GPU. Check install settings. See "
            "requirements here https://onnxruntime.ai/docs"
            "/execution-providers/CUDA-ExecutionProvider.html#requirements"
        )
    model_metadata = dict(model_sess.get_modelmeta().custom_metadata_map)
    model_metadata["base_pred"] = model_metadata["base_pred"] == "True"
    if model_metadata["mod_bases"] == "None":
        model_metadata["mod_bases"] = None
        model_metadata["mod_long_names"] = None
    else:
        model_metadata["mod_long_names"] = []
        for mod_idx in range(len(model_metadata["mod_bases"])):
            model_metadata["mod_long_names"].append(
                model_metadata[f"mod_long_names_{mod_idx}"]
            )
    model_metadata["kmer_context_bases"] = (
        int(model_metadata["kmer_context_bases_0"]),
        int(model_metadata["kmer_context_bases_1"]),
    )
    model_metadata["kmer_len"] = sum(model_metadata["kmer_context_bases"]) + 1
    model_metadata["chunk_context"] = (
        int(model_metadata["chunk_context_0"]),
        int(model_metadata["chunk_context_1"]),
    )
    model_metadata["chunk_len"] = sum(model_metadata["chunk_context"])

    if "num_motifs" not in model_metadata:
        model_metadata["motifs"] = [
            (model_metadata["motif"], int(model_metadata["motif_offset"]))
        ]
        model_metadata["motif_offset"] = int(model_metadata["motif_offset"])
    else:
        num_motifs = int(model_metadata["num_motifs"])

        motifs = []
        motif_offsets = []
        for mot in range(num_motifs):
            motifs.append(model_metadata[f"motif_{mot}"])
            motif_offsets.append(model_metadata[f"motif_offset_{mot}"])

        model_metadata["motifs"] = [
            (mot, int(mot_off)) for mot, mot_off in zip(motifs, motif_offsets)
        ]

    model_metadata["can_base"] = model_metadata["motifs"][0][0][
        model_metadata["motifs"][0][1]
    ]

    # allowed settings for this attribute are a single motif or all-contexts
    # note that inference core methods use motifs attribute
    if len(model_metadata["motifs"]) == 1:
        model_metadata["motif"] = model_metadata["motifs"][0]
    else:
        model_metadata["motif"] = (model_metadata["can_base"], 0)

    mod_str = "; ".join(
        f"{mod_b}={mln}"
        for mod_b, mln in zip(
            model_metadata["mod_bases"], model_metadata["mod_long_names"]
        )
    )
    model_metadata["alphabet_str"] = (
        "loaded modified base model to call (alt to "
        f"{model_metadata['can_base']}): {mod_str}"
    )

    if "refine_kmer_levels" in model_metadata:
        # load sig_map_refiner
        levels_array = np.frombuffer(
            model_metadata["refine_kmer_levels"].encode("cp437"),
            dtype=np.float32,
        )
        refine_sd_arr = np.frombuffer(
            model_metadata["refine_sd_arr"].encode("cp437"), dtype=np.float32
        )
        model_metadata["sig_map_refiner"] = SigMapRefiner(
            _levels_array=levels_array,
            center_idx=int(model_metadata["refine_kmer_center_idx"]),
            do_rough_rescale=bool(
                int(model_metadata["refine_do_rough_rescale"])
            ),
            scale_iters=int(model_metadata["refine_scale_iters"]),
            algo=model_metadata["refine_algo"],
            half_bandwidth=int(model_metadata["refine_half_bandwidth"]),
            sd_arr=refine_sd_arr,
        )

        model_metadata["base_start_justify"] = (
            model_metadata["base_start_justify"] == "True"
        )
        model_metadata["offset"] = int(model_metadata["offset"])
    else:
        # handle original models without sig_map_refiner
        model_metadata["sig_map_refiner"] = SigMapRefiner()
        model_metadata["base_start_justify"] = False
        model_metadata["offset"] = 0

    if not quiet:
        # skip attributes included in parsed values
        ckpt_attrs = "\n".join(
            f"  {k: >20} : {v}"
            for k, v in model_metadata.items()
            if not any(
                k.startswith(val)
                for val in (
                    "mod_long_names_",
                    "kmer_context_bases_",
                    "chunk_context_",
                    "motif_",
                    "refine_kmer_levels",
                    "refine_sd_arr",
                    "refine_kmer_center_idx",
                    "refine_do_rough_rescale",
                    "refine_scale_iters",
                    "refine_algo",
                    "refine_half_bandwidth",
                )
            )
        )
        LOGGER.debug(f"Loaded Remora model attrs\n{ckpt_attrs}\n")
    return model_sess, model_metadata


def load_model(
    model_filename=None,
    *,
    pore=None,
    basecall_model_type=None,
    basecall_model_version=None,
    modified_bases=None,
    remora_model_type=None,
    remora_model_version=None,
    device=None,
    quiet=True,
):
    if model_filename is not None:
        if not isfile(model_filename):
            raise RemoraError(
                f"Remora model file ({model_filename}) not found."
            )
        return load_onnx_model(model_filename, device, quiet=quiet)

    if pore is None:
        raise RemoraError("Must specify a pore.")
    try:
        pore = pore.lower()
        submodels = constants.MODEL_DICT[pore]
    except (AttributeError, KeyError):
        pores = ", ".join(constants.MODEL_DICT.keys())
        raise RemoraError(
            f"No trained Remora models for {pore}. Options: {pores}"
        )

    if basecall_model_type is None:
        raise RemoraError("Must specify a basecall model type.")
    try:
        basecall_model_type = basecall_model_type.lower()
        submodels = submodels[basecall_model_type]
    except (AttributeError, KeyError):
        model_types = ", ".join(submodels.keys())
        raise RemoraError(
            f"No trained Remora models for {basecall_model_type} "
            f"(with {pore}). Options: {model_types}"
        )

    if basecall_model_version is None:
        LOGGER.info(
            "Basecall model version not supplied. Using default Remora model "
            f"for {pore}_{basecall_model_type}."
        )
        basecall_model_version = constants.DEFAULT_BASECALL_MODEL_VERSION
    try:
        submodels = submodels[basecall_model_version]
    except KeyError:
        LOGGER.warning(
            "Remora model for basecall model version "
            f"({basecall_model_version}) not found. Using default Remora "
            f"model for {pore}_{basecall_model_type}."
        )
        basecall_model_version = constants.DEFAULT_BASECALL_MODEL_VERSION
        submodels = submodels[basecall_model_version]

    if modified_bases is None:
        LOGGER.info(
            "Modified bases not supplied. Using default "
            f"{constants.DEFAULT_MOD_BASE}."
        )
        modified_bases = constants.DEFAULT_MOD_BASE
    try:
        modified_bases = "_".join(sorted(x.lower() for x in modified_bases))
        submodels = submodels[modified_bases]
    except (AttributeError, KeyError):
        LOGGER.error(
            f"Remora model for modified bases {modified_bases} not found "
            f"for {pore}_{basecall_model_type}@{basecall_model_version}."
        )
        sys.exit(1)

    if remora_model_type is None:
        LOGGER.info(
            "Modified bases model type not supplied. Using default "
            f"{constants.DEFAULT_MODEL_TYPE}."
        )
        remora_model_type = constants.DEFAULT_MODEL_TYPE
    try:
        submodels = submodels[remora_model_type]
    except (AttributeError, KeyError):
        LOGGER.error(
            "Remora model type {remora_model_type} not found "
            f"for {pore}_{basecall_model_type}@{basecall_model_version} "
            f"{modified_bases}."
        )
        sys.exit(1)

    if remora_model_version is None:
        LOGGER.info("Remora model version not specified. Using latest.")
        remora_model_version = submodels[-1]
    if remora_model_version not in submodels:
        LOGGER.warning(
            f"Remora model version {remora_model_version} not found. "
            "Using latest."
        )
        remora_model_version = submodels[-1]
    remora_model_version = f"v{remora_model_version}"

    path = pkg_resources.resource_filename(
        "remora",
        os.path.join(
            MODEL_DATA_DIR_NAME,
            pore,
            basecall_model_type,
            basecall_model_version,
            modified_bases,
            remora_model_type,
            remora_model_version,
            constants.MODBASE_MODEL_NAME,
        ),
    )
    if not os.path.exists(path):
        raise RemoraError(
            f"No pre-trained Remora model for this configuration {path}."
        )
    return load_onnx_model(path, device, quiet=quiet)


def get_pretrained_models(
    pore=None,
    basecall_model_type=None,
    basecall_model_version=None,
    modified_bases=None,
    remora_model_type=None,
    remora_model_version=None,
):
    def filter_dataframe(models, args):
        running_sequence = []
        for x, y in zip(args, models.columns):
            if x is None:
                continue
            else:
                if y in ["Pore", "Basecall_Model_Type"]:
                    x = x.lower()
                elif y in ["Modified_Bases"]:
                    x = "_".join(sorted(z.lower() for z in x))
                elif y in ["Remora_Model_Type"]:
                    x = x.upper()
                running_sequence.append(x)
                models = models[models[y] == x]
                if len(models) == 0:
                    LOGGER.info(
                        f" {y} {','.join(running_sequence)} not found in "
                        "library of pre-trained models."
                    )
                    sys.exit(1)

        return models

    header = [
        "Pore",
        "Basecall_Model_Type",
        "Basecall_Model_Version",
        "Modified_Bases",
        "Remora_Model_Type",
        "Remora_Model_Version",
    ]

    models = []
    for pore_type, bc_types in constants.MODEL_DICT.items():
        for bc_type, bc_vers in bc_types.items():
            for bc_ver, mod_bases in bc_vers.items():
                for mod_base, remora_types in mod_bases.items():
                    for remora_type, remora_vers in remora_types.items():
                        for remora_ver in remora_vers:
                            models.append(
                                (
                                    pore_type,
                                    bc_type,
                                    bc_ver,
                                    mod_base,
                                    remora_type,
                                    remora_ver,
                                )
                            )
    models = pd.DataFrame(models)
    models.columns = header

    args = [
        pore,
        basecall_model_type,
        basecall_model_version,
        modified_bases,
        remora_model_type,
        remora_model_version,
    ]

    models = filter_dataframe(models, args)

    return models, header
