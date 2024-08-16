import argparse

import torch

from remora import model_util


def main(args):
    model, model_metadata = model_util._raw_load_torchscript_model(
        args.torchscript_input_model
    )
    model_metadata["epoch"] = 0
    model_metadata["chunk_context"] = tuple(model_metadata["chunk_context"])
    model_metadata["kmer_context_bases"] = tuple(
        model_metadata["kmer_context_bases"]
    )
    state_dict = model.state_dict()
    if "total_ops" in state_dict.keys():
        state_dict.pop("total_ops", None)
    if "total_params" in state_dict.keys():
        state_dict.pop("total_params", None)
    model_metadata["state_dict"] = state_dict
    model_metadata["opt"] = None
    torch.save(model_metadata, args.checkout_output_model)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("torchscript_input_model")
    parser.add_argument("checkout_output_model")
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
