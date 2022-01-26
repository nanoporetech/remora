import copy
import datetime
import importlib
import os
from os.path import isfile
import pkg_resources
import sys

import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
import torch
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report

from remora import log, RemoraError, encoded_kmers, util, constants

LOGGER = log.get_logger()
MODEL_DATA_DIR_NAME = "trained_models"
ONNX_NUM_THREADS = 1


class ValidationLogger:
    def __init__(self, out_path):
        self.fp = open(out_path, "w", buffering=1)
        self.fp.write(
            "\t".join(
                (
                    "Val_Type",
                    "Iteration",
                    "Accuracy",
                    "Loss",
                    "F1",
                    "Precision",
                    "Recall",
                    "Num_Calls",
                    "Flt_Confusion",
                    "Conf_frac",
                    "Min_F1",
                )
            )
            + "\n"
        )

    def close(self):
        self.fp.close()

    def get_results(
        self,
        all_outputs,
        all_labels,
        all_loss,
        conf_thr,
        val_type,
        niter,
    ):
        pred_labels = np.argmax(all_outputs, axis=1)
        acc = (pred_labels == all_labels).sum() / all_outputs.shape[0]
        mean_loss = np.mean(all_loss)
        cl_rep = classification_report(
            all_labels,
            pred_labels,
            digits=3,
            output_dict=True,
            zero_division=0,
        )
        min_f1 = min(cl_rep[str(k)]["f1-score"] for k in np.unique(all_labels))
        all_out_soft = util.softmax_axis1(all_outputs)
        conf_ids = np.max(all_out_soft > conf_thr, axis=1)
        conf_calls = all_out_soft[conf_ids]
        pred_conf_labels = np.argmax(conf_calls, axis=1)
        conf_mat = confusion_matrix(all_labels[conf_ids], pred_conf_labels)
        conf_frac = len(all_labels[conf_ids]) / len(all_labels)

        cm_flat_str = np.array2string(
            conf_mat.flatten(), separator=","
        ).replace("\n", "")

        if len(np.unique(all_labels)) > 2:
            precision = [float("NaN")]
            recall = [float("NaN")]
            f1_scores = [float("NaN")]
            f1_idx = 0
        else:
            uniq_labs = np.unique(all_labels)
            pos_idx = np.max(uniq_labs)
            with np.errstate(invalid="ignore"):
                precision, recall, thresholds = precision_recall_curve(
                    all_labels, all_outputs[:, pos_idx], pos_label=pos_idx
                )
                f1_scores = 2 * recall * precision / (recall + precision)
            f1_idx = np.argmax(f1_scores)
        self.fp.write(
            f"{val_type}\t{niter}\t{acc:.6f}\t{mean_loss:.6f}\t"
            f"{f1_scores[f1_idx]:.6f}\t{precision[f1_idx]:.6f}\t"
            f"{recall[f1_idx]:.6f}\t{len(all_labels)}\t"
            f"{cm_flat_str}\t{conf_frac:.2f}\t{min_f1:.6f}\n"
        )
        return acc, mean_loss

    def validate_model(
        self,
        model,
        criterion,
        dataset,
        niter,
        val_type="val",
        conf_thr=constants.DEFAULT_CONF_THR,
    ):
        model.eval()
        bb, ab = dataset.kmer_context_bases
        with torch.no_grad():
            all_labels = []
            all_outputs = []
            all_loss = []
            for (sigs, seqs, seq_maps, seq_lens), labels, _ in dataset:
                all_labels.append(labels)
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
                output = model(sigs, enc_kmers).detach().cpu()
                all_outputs.append(output)
                loss = criterion(output, labels)
                all_loss.append(loss.detach().cpu().numpy())
            all_outputs = np.concatenate(all_outputs, axis=0)
            all_labels = np.concatenate(all_labels)
        return self.get_results(
            all_outputs, all_labels, all_loss, conf_thr, val_type, niter
        )

    def validate_onnx_model(
        self,
        model,
        criterion,
        dataset,
        niter,
        val_type="val",
        conf_thr=constants.DEFAULT_CONF_THR,
    ):
        bb, ab = dataset.kmer_context_bases
        all_labels = []
        all_outputs = []
        all_loss = []
        for (sigs, seqs, seq_maps, seq_lens), labels, _ in dataset:
            all_labels.append(labels)
            enc_kmers = encoded_kmers.compute_encoded_kmer_batch(
                bb, ab, seqs, seq_maps, seq_lens
            )
            output = model.run([], {"sig": sigs, "seq": enc_kmers})[0]
            all_outputs.append(output)
            loss = criterion(torch.from_numpy(output), torch.from_numpy(labels))
            all_loss.append(loss.numpy())
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels)
        return self.get_results(
            all_outputs, all_labels, all_loss, conf_thr, val_type, niter
        )


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
    meta = onnx_model.metadata_props.add()
    meta.key = "base_pred"
    meta.value = str(ckpt["base_pred"])
    meta = onnx_model.metadata_props.add()
    meta.key = "mod_bases"
    meta.value = str(ckpt["mod_bases"])
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
    LOGGER.debug(f"Using {ONNX_NUM_THREADS} thread for ONNX")
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
        model_metadata["can_base"] = model_metadata["motifs"][0][0][
            model_metadata["motifs"][0][1]
        ]
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

        model_metadata["can_base"] = [
            mot[0][mot[1]] for mot in model_metadata["motifs"]
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
    if not quiet:
        ckpt_attrs = "\n".join(
            f"  {k: >20} : {v}" for k, v in model_metadata.items()
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
