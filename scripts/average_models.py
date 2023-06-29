import argparse
import os

import torch
from remora import model_util


def average_weights(model, weights_paths):
    model_weights = list()
    for weights_path in weights_paths:
        print(weights_path)
        model_data = torch.load(weights_path)
        model.load_state_dict(model_data["state_dict"])

        model_weights.append(model_data["state_dict"])

    nr_models = len(model_weights)
    if nr_models == 1:
        print("Only one model provided. Returning the initial model")
        return model_weights[0]

    for key in model_weights[0].keys():
        for i in range(1, nr_models):
            model_weights[0][key] += model_weights[i][key]
        model_weights[0][key] = torch.div(model_weights[0][key], nr_models)
        model.load_state_dict(model_weights[0])
    return model_weights[0]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Input checkpoint model paths (*.checkpoint)",
    )
    parser.add_argument(
        "--model-weights-paths",
        type=str,
        nargs="+",
        required=True,
        help="Input checkpoint model paths (*.checkpoint)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./",
        help="Path to the output directory.",
    )
    return parser


def main(args):
    model_data = torch.load(args.model_weights_paths[0])

    model_params = {
        "size": model_data["model_params"]["size"],
        "kmer_len": model_data["model_params"]["kmer_len"],
        "num_out": model_data["model_params"]["num_out"],
    }

    model = model_util._load_python_model(args.model_path, **model_params)
    model = model.cuda()
    new_state_dict = average_weights(model, args.model_weights_paths)

    model_data["state_dict"] = new_state_dict
    torch.save(
        model_data, os.path.join(args.output_path, "average_model.checkpoint")
    )


if __name__ == "__main__":
    main(get_parser().parse_args())
