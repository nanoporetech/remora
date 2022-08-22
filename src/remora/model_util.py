import os
import sys
import copy
import json
import toml
import datetime
import warnings
import importlib
import pkg_resources
from os.path import isfile

import onnx
import torch
from torch.nn.utils.fusion import fuse_conv_bn_eval
import numpy as np
import pandas as pd
from torch import nn
import onnxruntime as ort

from remora import log, RemoraError, constants
from remora.refine_signal_map import SigMapRefiner

LOGGER = log.get_logger()
MODEL_DATA_DIR_NAME = "trained_models"
ONNX_NUM_THREADS = 1


#############
# Exporting #
#############


def export_model_onnx(ckpt, model, save_filename):
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
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


def export_model_torchscript(ckpt, model, save_filename):
    model.eval()
    m = torch.jit.script(model)
    meta = {}
    meta["creation_date"] = datetime.datetime.now().strftime(
        "%m/%d/%Y, %H:%M:%S"
    )
    meta["kmer_context_bases"] = ckpt["kmer_context_bases"]
    meta["chunk_context"] = ckpt["chunk_context"]

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
        "model_params",
    ):
        meta[ckpt_key] = ckpt[ckpt_key]

    if ckpt["mod_bases"] is not None:
        for mod_idx in range(len(ckpt["mod_bases"])):
            meta[f"mod_long_names_{mod_idx}"] = str(
                ckpt["mod_long_names"][mod_idx]
            )
    meta["num_motifs"] = str(ckpt["num_motifs"])
    for idx, (motif, motif_offset) in enumerate(ckpt["motifs"]):
        m_key = f"motif_{idx}"
        mo_key = f"motif_offset_{idx}"
        meta[m_key] = str(motif)
        meta[mo_key] = str(motif_offset)

    meta["num_motifs"] = str(len(ckpt["motifs"]))

    # store refine arrays as bytes
    meta["refine_kmer_levels"] = (
        ckpt["refine_kmer_levels"].astype(np.float32).tobytes().decode("cp437")
    )
    meta["refine_sd_arr"] = (
        ckpt["refine_sd_arr"].astype(np.float32).tobytes().decode("cp437")
    )
    meta["doc_string"] = "Nanopore Remora model"
    try:
        meta["model_version"] = ckpt["model_version"]
    except KeyError:
        LOGGER.warning("Model version not found in checkpoint. Setting to 0.")
        meta["model_version"] = 0
    json_object = json.dumps(meta, indent=4)
    extra_files = {"meta.txt": json_object}
    torch.jit.save(m, save_filename, _extra_files=extra_files)


def export_model_dorado(ckpt, model, save_dir):
    save_dir = os.path.expanduser(save_dir)
    if os.path.exists(save_dir):
        LOGGER.info(
            f'Directory "{save_dir}" already exists. Exported files will '
            "be overwritten."
        )
    os.makedirs(save_dir, exist_ok=True)

    def save_tensor(fn, x):
        m = torch.nn.Module()
        par = nn.Parameter(x, requires_grad=False)
        m.register_parameter("0", par)
        tensors = torch.jit.script(m)
        tensors.save(f"{save_dir}/{fn}.tensor")
        LOGGER.info(f"{save_dir}/{fn}.tensor")

    model.eval()
    layer_names = set()

    conv_to_bn = {
        "sig_conv1": "sig_bn1",
        "sig_conv2": "sig_bn2",
        "sig_conv3": "sig_bn3",
        "seq_conv1": "seq_bn1",
        "seq_conv2": "seq_bn2",
        "seq_conv3": "seq_bn3",
        "merge_conv1": "merge_bn",
        "merge_conv2": "merge_bn2",
        "merge_conv3": "merge_bn3",
        "merge_conv4": "merge_bn4",
    }

    for name, module in model.named_modules():
        if name == "" or "bn" in name or "drop" in name:
            continue
        if name in conv_to_bn.keys():
            fused = fuse_conv_bn_eval(module, getattr(model, conv_to_bn[name]))
            module.weight = fused.weight
            module.bias = fused.bias
        for k, v in module.state_dict().items():
            save_tensor(f"{name}.{k}", v)
        layer_names.add(name)

    metadata = {}
    general = {}
    modbases = {}

    general["creation_date"] = datetime.datetime.now().strftime(
        "%m/%d/%Y, %H:%M:%S"
    )

    lstm_model_layers = {
        "sig_conv1",
        "sig_conv2",
        "sig_conv3",
        "seq_conv1",
        "seq_conv2",
        "merge_conv1",
        "lstm1",
        "lstm2",
        "fc",
    }

    conv_model_layers = {
        "sig_conv1",
        "sig_conv2",
        "sig_conv3",
        "seq_conv1",
        "seq_conv2",
        "seq_conv3",
        "merge_conv1",
        "merge_conv2",
        "merge_conv3",
        "merge_conv4",
        "fc",
    }

    if layer_names == conv_model_layers:
        general["model"] = "conv_only"
    elif layer_names == lstm_model_layers:
        general["model"] = "conv_lstm"
    else:
        LOGGER.warning("Unknown layer setup in export")
        general["model"] = "unknown"

    # add refinement metadata
    refinement = {}
    refinement["refine_do_rough_rescale"] = ckpt["refine_do_rough_rescale"]
    if refinement["refine_do_rough_rescale"]:
        refinement["refine_kmer_center_idx"] = ckpt["refine_kmer_center_idx"]
        try:
            refine_kmer_levels = torch.Tensor(
                ckpt["refine_kmer_levels"].astype(np.float32)
            )
        except AttributeError:
            refine_kmer_levels = torch.Tensor(
                np.frombuffer(
                    ckpt["refine_kmer_levels"].encode("cp437"), dtype=np.float32
                )
            )
        save_tensor("refine_kmer_levels", refine_kmer_levels)

    # add simple metadata
    for ckpt_key in (
        "mod_bases",
        "offset",
    ):
        modbases[ckpt_key] = ckpt[ckpt_key]

    if ckpt["mod_bases"] is not None:
        for mod_idx in range(len(ckpt["mod_bases"])):
            modbases[f"mod_long_names_{mod_idx}"] = str(
                ckpt["mod_long_names"][mod_idx]
            )
    for key in ("chunk_context", "kmer_context_bases"):
        for idx in range(2):
            modbases[f"{key}_{idx}"] = ckpt[key][idx]

    if len(ckpt["motifs"]) > 1:
        raise RemoraError("Dorado only supports models with a single motif")

    for (motif, motif_offset) in ckpt["motifs"]:
        modbases["motif"] = motif
        modbases["motif_offset"] = motif_offset

    metadata["general"] = general
    metadata["model_params"] = ckpt["model_params"]
    metadata["modbases"] = modbases
    metadata["refinement"] = refinement

    toml.dump(metadata, open(os.path.join(save_dir, "config.toml"), "w"))


###########
# Loading #
###########


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


def add_derived_metadata(model_metadata):
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
    if "kmer_context_bases" not in model_metadata:
        model_metadata["kmer_context_bases"] = (
            int(model_metadata["kmer_context_bases_0"]),
            int(model_metadata["kmer_context_bases_1"]),
        )
    model_metadata["kmer_len"] = sum(model_metadata["kmer_context_bases"]) + 1
    if "chunk_context" not in model_metadata:
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


def repr_model_metadata(metadata):
    # skip attributes included in parsed values
    return "\n".join(
        f"  {k: >20} : {v}"
        for k, v in metadata.items()
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
    add_derived_metadata(model_metadata)
    if not quiet:
        md_str = repr_model_metadata(model_metadata)
        LOGGER.debug(f"Loaded Remora model attrs\n{md_str}\n")
    return model_sess, model_metadata


def load_torchscript_model(model_filename, device=None, quiet=False):
    """Load onnx model. If device is specified load onto specified device.

    Args:
        model_filename (str): Model path
        device (int): GPU device ID
        quiet (bool): Print model info to debug

    Returns:
        2-tuple containing:
          1. Compiled model object for calling mods
          2. Model metadata dictionary with information concerning data prep
    """

    # values will be replaced with data
    extra_files = {"meta.txt": ""}
    if device is None:
        model = torch.jit.load(
            model_filename, _extra_files=extra_files, map_location="cpu"
        )
    else:
        model = torch.jit.load(
            model_filename,
            _extra_files=extra_files,
            map_location=torch.device(device),
        )
    model_metadata = json.loads(extra_files["meta.txt"])
    add_derived_metadata(model_metadata)
    if not quiet:
        md_str = repr_model_metadata(model_metadata)
        LOGGER.debug(f"Loaded Remora model attrs\n{md_str}\n")
    return model, model_metadata


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
        try:
            return load_torchscript_model(model_filename, device, quiet=quiet)
        except (AttributeError, RuntimeError):
            LOGGER.warning("Failed loading torchscript model. Trying onnx.")
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
    try:
        return load_torchscript_model(path, device)
    except (AttributeError, RuntimeError):
        LOGGER.warning("Failed loading torchscript model. Trying onnx.")
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
