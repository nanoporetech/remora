import os
import sys
import json
import base64
import argparse

import torch
import numpy as np
from torch.nn.utils.fusion import fuse_conv_bn_eval

from remora.model_util import continue_from_checkpoint, load_torchscript_model


OPTYPE_RENAMES = {
    "Conv": "convolution",
    "Gemm": "softmax",
    "LSTM": "LSTM",
}

MODULE_RENAMES = {
    "Conv1d": "convolution",
    "LSTM": "LSTM",
    "Linear": "softmax",
}


def _get_module_type(module_type: str) -> str:
    return MODULE_RENAMES.get(module_type, module_type.lower())


def _make_feedforward_node(size: int):
    params = {
        "W_binary": base64.b64encode(
            np.eye(size, size, dtype=np.float32).tobytes()
        ).decode()
    }
    activation_node = {
        "insize": size,
        "size": size,
        "type": "feed-forward",
        "activation": "swish",
        "params": params,
    }
    return activation_node


def _translate_metadata(md_dict):
    if int(md_dict.get("num_motifs", 0)) > 1:
        raise ValueError(
            "Conversion to Guppy format not currently compatible with "
            f"multiple motifs. Found {md_dict['num_motifs']}"
        )
    if int(md_dict.get("refine_scale_iters", -1)) != -1:
        raise ValueError(
            "Conversion to Guppy format not currently compatible with "
            f"refine_scale_iters. Found {md_dict['refine_scale_iters']}"
        )
    numeric_values = {
        "refine_kmer_center_idx",
    }
    base64_values = {
        "refine_kmer_levels",
    }
    bool_values = {
        "refine_do_rough_rescale",
    }
    metadata = {}
    metadata["kmer_context_bases_0"] = md_dict["kmer_context_bases"][0]
    metadata["kmer_context_bases_1"] = md_dict["kmer_context_bases"][1]
    metadata["chunk_context_0"] = md_dict["chunk_context"][0]
    metadata["chunk_context_1"] = md_dict["chunk_context"][1]
    metadata["motif"] = md_dict["motif"][0]
    metadata["motif_offset"] = md_dict["motif"][1]
    # modified base specification
    metadata["mod_bases"] = md_dict["mod_bases"]
    for mod_idx in range(len(md_dict["mod_bases"])):
        metadata[f"mod_long_names_{mod_idx}"] = md_dict[
            f"mod_long_names_{mod_idx}"
        ]
    for kv_key in md_dict:
        if kv_key in numeric_values:
            metadata[kv_key] = int(md_dict[kv_key])
        elif kv_key in bool_values:
            metadata[kv_key] = (
                md_dict[kv_key] == "True"
                or md_dict[kv_key] == "1"
                or md_dict[kv_key] == 1
            )
        elif kv_key in base64_values:
            try:
                value = np.frombuffer(md_dict[kv_key], dtype=np.float32)
            except TypeError:
                value = np.frombuffer(
                    md_dict[kv_key].encode("cp437"), dtype=np.float32
                )
            metadata[kv_key + "_binary"] = base64.b64encode(value).decode()
    return metadata


def _module_to_dict(module, name, reverse=False):
    module_info = {"type": _get_module_type(name)}

    if name == "Conv1d":
        params = {}
        params["W_binary"] = base64.b64encode(
            module.weight.detach().numpy().tobytes()
        ).decode()
        if module.bias is not None:
            params["b_binary"] = base64.b64encode(
                module.bias.detach().numpy().tobytes()
            ).decode()
        module_info["insize"] = module.in_channels
        module_info["size"] = module.out_channels
        module_info["activation"] = "swish"
        module_info["stride"] = module.stride[0]
        module_info["winlen"] = module.kernel_size[0]
        module_info["padding"] = [module.padding[0], module.padding[0]]
        module_info["bias"] = module.bias is not None
        module_info["params"] = params

    elif name == "LSTM":
        hidden_size = module.hidden_size
        insize = module.input_size
        params = {}
        iW = module.weight_ih_l0.reshape(4, hidden_size, insize)
        params["iW_binary"] = base64.b64encode(
            iW.detach().numpy().tobytes()
        ).decode()
        sW = module.weight_hh_l0.reshape(4, hidden_size, hidden_size)
        params["sW_binary"] = base64.b64encode(
            sW.detach().numpy().tobytes()
        ).decode()
        if module.bias:
            b = module.bias_ih_l0.reshape(4, hidden_size).detach().numpy()
            b += module.bias_hh_l0.reshape(4, hidden_size).detach().numpy()
            params["b_binary"] = base64.b64encode(b.tobytes()).decode()

        module_info["insize"] = insize
        module_info["size"] = hidden_size
        module_info["activation"] = "tanh"
        module_info["gate"] = "sigmoid"
        module_info["bias"] = module.bias
        module_info["params"] = params
        if reverse:
            module_info = {"sublayers": module_info, "type": "reverse"}
        module_info = {
            "sublayers": [module_info, _make_feedforward_node(hidden_size)],
            "type": "serial",
        }
    elif name == "Linear":
        params = {}
        params["W_binary"] = base64.b64encode(
            module.weight.detach().numpy().tobytes()
        ).decode()
        if module.bias is not None:
            params["b_binary"] = base64.b64encode(
                module.bias.detach().numpy().tobytes()
            ).decode()
        module_info["insize"] = module.in_features
        module_info["size"] = module.out_features
        module_info["bias"] = module.bias is not None
        module_info["rotate"] = False
        module_info["params"] = params

    return module_info


def convert(model_path: str, json_path: str = None, json_indent: int = 2):
    # Convert Torchscript model to JSON

    model_file_path, model_ext = os.path.splitext(model_path)
    if model_ext == ".pt":
        model, model_metadata = load_torchscript_model(model_path)
    elif model_ext == ".checkpoint":
        checkpoint = torch.load(model_path)
        model_metadata, model = continue_from_checkpoint(
            model_path, checkpoint["model_path"]
        )

    model_metadata = _translate_metadata(model_metadata)
    layer_list = []
    reverse = False

    all_layers = {}

    for name, module in model.named_modules():
        all_layers[name] = module

    for name, module in model.named_modules():
        try:
            org_name = module.original_name
        except AttributeError:
            org_name = module.__class__.__name__
        if org_name in ["BatchNorm1d", "Dropout", "network"]:
            continue
        if org_name == "Conv1d":
            s_name = name.split("_conv")
            for ln in all_layers.keys():
                if s_name[0] in ln and s_name[1] in ln and "bn" in ln:
                    fused = fuse_conv_bn_eval(module, all_layers[ln])
                    module.weight = fused.weight
                    module.bias = fused.bias
                if s_name[0] in ln and "merge" in ln and "bn" in ln:
                    fused = fuse_conv_bn_eval(module, all_layers[ln])
                    module.weight = fused.weight
                    module.bias = fused.bias
        module_dict = _module_to_dict(module, org_name, reverse)
        if "LSTM" in org_name:
            reverse = not reverse
        layer_list.append(module_dict)

    node_output = []
    arch = [
        "input",
        "convolution",
        "convolution",
        "convolution",
        "stash",
        "convolution",
        "convolution",
        "concat",
        "convolution",
        "serial",
        "serial",
        "slice",
        "softmax",
    ]

    insize = 1
    stash_size = 0
    layer_iter = 0
    for layer in arch:
        if layer == "input":
            insize = layer_list[0]["insize"] + layer_list[3]["insize"]
            size = layer_list[0]["insize"]
            node_info = {
                "type": "input",
                "insize": insize,
                "size": size,
            }
            node_output.append(node_info)
        elif layer in ["convolution", "serial", "softmax"]:
            node_output.append(layer_list[layer_iter])
            layer_iter += 1
        elif layer == "stash":
            temp_size = layer_list[layer_iter]["insize"]
            insize = layer_list[layer_iter + 1]["size"]
            node_info = {
                "type": "stash",
                "insize": insize,
                "size": temp_size,
            }
            stash_size = insize
            node_output.append(node_info)
        elif layer == "concat":
            node_info = {"type": layer}
            insize = layer_list[layer_iter]["size"]
            insizes = [insize, stash_size]
            output_size = sum(insizes)
            node_info["insize"] = insize
            node_info["size"] = output_size
            node_info["insizes"] = insizes
            node_output.append(node_info)
        elif layer == "slice":
            node_info = {"type": layer}
            node_info["insize"] = layer_list[layer_iter]["insize"]
            node_info["size"] = layer_list[layer_iter]["insize"]
            node_info["start"] = -1
            node_info["end"] = sys.maxsize
            node_output.append(node_info)

    output_dict = {
        "sublayers": node_output,
        "type": "serial",
        "metadata": model_metadata,
    }
    with open(json_path if json_path else f"{model_file_path}.jsn", "w") as fp:
        json.dump(output_dict, fp, indent=json_indent)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Input torchscript model path (*.pt)",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Output JSON file path (*.jsn)",
    )
    parser.add_argument(
        "--json-indent",
        type=int,
        default=2,
        help="Number of indentations in JSON (default=2)",
    )
    return parser


def main(args):
    convert(args.model_path, args.json_path, args.json_indent)


if __name__ == "__main__":
    main(get_parser().parse_args())
