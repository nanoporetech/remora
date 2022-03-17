import os
import sys
import json
import base64
import argparse

import onnx
import numpy as np
from onnx.onnx_ml_pb2 import NodeProto, TensorProto

UNSAVES_NODES = set(("Transpose", "Squeeze", "Shape"))
SKIP_COUNTS = {"Shape": 7, "Conv": 2, "LSTM": 3}


def _get_node_type(op_type: str) -> str:
    try:
        optype_renames = {
            "Conv": "convolution",
            "Gemm": "softmax",
            "LSTM": "LSTM",
        }
        return optype_renames[op_type]
    except KeyError:
        return op_type.lower()


def _get_attribute_from_node(node: NodeProto, attribute: str):
    for x in node.attribute:
        if x.name != attribute:
            continue
        return x.ints[0] if x.type == x.INTS else x.i


def _get_attribute_list_from_node(node: NodeProto, attribute: str):
    for x in node.attribute:
        if x.name != attribute:
            continue
        return list(x.ints)


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


def _translate_metadata(metadata_props):
    numeric_values = [
        "motif_offset",
        "kmer_context_bases_0",
        "kmer_context_bases_1",
        "chunk_context_0",
        "chunk_context_1",
    ]
    bool_values = ["base_pred"]
    metadata = {}
    for kv_pair in metadata_props:
        if kv_pair.key in numeric_values:
            metadata[kv_pair.key] = int(kv_pair.value)
        elif kv_pair.key in bool_values:
            metadata[kv_pair.key] = kv_pair.value == "True"
        else:
            metadata[kv_pair.key] = kv_pair.value
    return metadata


def _node_to_dict(
    node: NodeProto,
    insize: int,
    stash_size: int,
    reverse: bool,
    initializers: "dict[str, TensorProto]",
    constants: "dict[str: int]",
):
    node_info = {"type": _get_node_type(node.op_type), "insize": insize}

    if node.op_type == "Conv":
        param_names = ["W_binary", "b_binary"]
        params = {}
        input_number = 0
        output_size = 0
        for input in node.input:
            try:
                init = initializers[input]
                vals = np.frombuffer(init.raw_data, dtype=np.float32).reshape(
                    init.dims[0:]
                )
                if input_number == 0:
                    # W_binary shape sets the size parameter
                    output_size = init.dims[0]
                params[param_names[input_number]] = base64.b64encode(
                    vals.tobytes()
                ).decode()
                input_number += 1
            except KeyError:
                pass
        node_info["size"] = output_size
        node_info["activation"] = "swish"
        node_info["stride"] = _get_attribute_from_node(node, "strides")
        node_info["winlen"] = _get_attribute_from_node(node, "kernel_shape")
        node_info["padding"] = _get_attribute_list_from_node(node, "pads")
        node_info["bias"] = input_number == 2
        node_info["params"] = params

    elif node.op_type == "LSTM":
        hidden_size = _get_attribute_from_node(node, "hidden_size")
        output_size = hidden_size

        param_names = ["iW_binary", "sW_binary", "b_binary"]
        shapes = [
            [4, hidden_size, insize],
            [4, hidden_size, hidden_size],
            [8, hidden_size],
        ]
        params = {}
        input_number = 0
        for input_i in node.input:
            try:
                init = initializers[input_i]
                param_values = np.frombuffer(
                    init.raw_data, dtype=np.float32
                ).reshape(shapes[input_number])
                if param_values.shape[0] == 8:
                    # sum the two sets of biases
                    param_values = param_values[0:4] + param_values[4:8]
                # reorder from iofc to ifco for guppy
                param_values = param_values[[0, 2, 3, 1]]
                params[param_names[input_number]] = base64.b64encode(
                    param_values.tobytes()
                ).decode()
                input_number += 1
            except KeyError:
                pass
        node_info["size"] = (hidden_size,)
        node_info["activation"] = "tanh"
        node_info["gate"] = "sigmoid"
        node_info["bias"] = input_number == 3
        node_info["params"] = params
        if reverse:
            node_info = {"sublayers": node_info, "type": "reverse"}
        node_info = {
            "sublayers": [node_info, _make_feedforward_node(hidden_size)],
            "type": "serial",
        }

    elif node.op_type == "Gemm":
        param_names = ["W_binary", "b_binary"]
        params = {}
        input_number = 0
        output_size = 0
        for input in node.input:
            try:
                init = initializers[input]
                param = np.frombuffer(init.raw_data, dtype=np.float32)
                if input_number == 0:
                    # W_binary shape sets the insize and size parameters
                    output_size = init.dims[0]
                    transpose = _get_attribute_from_node(node, "transB") == 0
                    if transpose:
                        # not sure about this, but the current models have
                        # transB=1 so we're not exercising it
                        param = param.reshape(init.dims[0], init.dims[1])
                        param = np.transpose(param)
                params[param_names[input_number]] = base64.b64encode(
                    param.tobytes()
                ).decode()
                input_number += 1
            except KeyError:
                pass
        node_info["size"] = output_size
        node_info["bias"] = input_number == 2
        node_info["rotate"] = False
        node_info["params"] = params

    elif node.op_type == "Concat":
        insizes = [insize, stash_size]
        output_size = sum(insizes)
        node_info["size"] = output_size
        node_info["insizes"] = insizes

    elif node.op_type == "Slice":
        # check inputs: if it's a reverse, set the flag for the next layer
        if len(node.input) == 5 and constants[node.input[-1]] == -1:
            # flip layers have a 5th "steps" input which is -1
            return None, insize, not reverse
        # else, it's the final slice layer, so slice it
        output_size = insize
        node_info["size"] = insize
        node_info["start"] = constants[node.input[1]]
        node_info["end"] = constants[node.input[2]]

    elif node.op_type == "Flatten":
        # how do we determine this???
        output_size = insize * 3
        node_info["size"] = output_size
        node_info["axis"] = _get_attribute_from_node(node, "axis")

    return node_info, output_size, reverse


def convert(model_path: str, json_path: str = None, json_indent: int = 2):
    # Convert onnx model to JSON
    model_file_path, model_ext = os.path.splitext(model_path)
    onnx_model = onnx.load(model_path)

    initializers = {str(t.name): t for t in onnx_model.graph.initializer}
    graph_inputs = {str(t.name): t for t in onnx_model.graph.input}

    node_output = []
    insize = 1
    stash_size = 0
    reverse = False

    constants = {}
    input_node_sizes = []
    node_iter = iter(onnx_model.graph.node)
    node = next(node_iter)
    while node is not None:
        if node.op_type == "Constant":
            constants[node.output[0]] = int.from_bytes(
                node.attribute[0].t.raw_data, sys.byteorder, signed=True
            )
            node = next(node_iter, None)
            continue
        if node.op_type not in UNSAVES_NODES:
            for input_i in node.input:
                if input_i not in graph_inputs:
                    continue
                temp_size = (
                    graph_inputs[input_i]
                    .type.tensor_type.shape.dim[1]
                    .dim_value
                )
                input_node_sizes.append(temp_size)
                if node_output:
                    # if this isn't the very first node, stash this data away
                    node_info = {
                        "type": "stash",
                        "insize": insize,
                        "size": temp_size,
                    }
                    stash_size = insize
                    node_output.append(node_info)
                insize = temp_size

            node_info, insize, reverse = _node_to_dict(
                node, insize, stash_size, reverse, initializers, constants
            )
            if node_info is not None:
                node_output.append(node_info)

        # save constant node from shape layers
        if node.op_type == "Shape":
            const_node = next(node_iter)
            constants[const_node.output[0]] = int.from_bytes(
                const_node.attribute[0].t.raw_data, sys.byteorder, signed=True
            )
        # skip nodes not required by guppy
        for _ in range(SKIP_COUNTS.get(node.op_type, 0)):
            next(node_iter)
        node = next(node_iter, None)

    # create layers to store additional inputs
    input_node_sizes.reverse()
    for i in range(0, len(input_node_sizes) - 1):
        total_input_size = sum(input_node_sizes[i:])
        output_size = sum(input_node_sizes[i + 1 :])
        node_output.insert(
            0,
            {"type": "input", "insize": total_input_size, "size": output_size},
        )

    metadata_info = _translate_metadata(onnx_model.metadata_props)

    output_dict = {
        "sublayers": node_output,
        "type": "serial",
        "metadata": metadata_info,
    }
    with open(json_path if json_path else f"{model_file_path}.jsn", "w") as fp:
        json.dump(output_dict, fp, indent=json_indent)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Input onnx model path (*.onnx)",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Output JSON file path (*.jsn)",
    )
    parser.add_argument(
        "--json_indent",
        type=int,
        default=2,
        help="Number of indentations in JSON (default=2)",
    )
    return parser


def main(args):
    convert(args.model_path, args.json_path, args.json_indent)


if __name__ == "__main__":
    main(get_parser().parse_args())
