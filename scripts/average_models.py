import argparse
import os
import copy

import torch
import numpy as np
from remora.data_chunks import RemoraDataset
from remora import model_util, train_model


def average_weights(model, weights_paths, val_set, val_fp, avg_mode="simple"):
    initial_accs = list()
    model_weights = list()
    criterion = torch.nn.CrossEntropyLoss()
    for weights_path in weights_paths:
        print(weights_path)
        model_data = torch.load(weights_path)
        model.load_state_dict(model_data["state_dict"])

        model_weights.append(model_data["state_dict"])

        val_metrics = val_fp.validate_model(
            model, val_set.mod_bases, criterion, val_set
        )
        initial_accs.append(val_metrics[1])

    print("Individual model accuracies", initial_accs)

    if avg_mode == "simple":
        nr_models = len(model_weights)
        if nr_models == 1:
            print("Only one model provided. Returning the initial model")
            return model_weights[0]
        else:
            for key in model_weights[0].keys():
                for i in range(1, nr_models):
                    model_weights[0][key] += model_weights[i][key]
                model_weights[0][key] = torch.div(
                    model_weights[0][key], nr_models
                )
            model.load_state_dict(model_weights[0])
            val_metrics = val_fp.validate_model(
                model, val_set.mod_bases, criterion, val_set
            )
            print(
                f"Averaged model accuracy with {avg_mode} averaging is {val_metrics[1]}."
            )
            return model_weights[0]
    else:
        top_acc = max(initial_accs)
        best_id = np.argmax(initial_accs)
        best_weights = [copy.deepcopy(model_weights[best_id])]
        best_candidate = copy.deepcopy(model_weights[best_id])
        for i in range(len(model_weights)):
            candidate_weights = {}
            if i != best_id:
                for bw in best_weights:
                    for key in model_weights[0].keys():
                        if key in candidate_weights.keys():
                            candidate_weights[key] += bw[key]
                        else:
                            candidate_weights[key] = bw[key]
                for key in model_weights[0].keys():
                    candidate_weights[key] += model_weights[i][key]
                candidate_weights[key] = torch.div(
                    model_weights[0][key], len(best_weights) + 1
                )
                model.load_state_dict(candidate_weights)
                val_metrics = val_fp.validate_model(
                    model, val_set.mod_bases, criterion, val_set
                )
                if val_metrics[1] > top_acc:
                    print(f"New best: {val_metrics[1]}.")
                    top_acc = val_metrics[1]
                    best_weights.append(copy.deepcopy(model_weights[i]))
                    best_candidate = copy.deepcopy(candidate_weights)
        print(
            f"Averaged model accuracy with {avg_mode} averaging is {top_acc}."
        )
        return best_candidate


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
        "--val-set-path",
        type=str,
        default=None,
        help="Validation set used for evaluating weight averaging.",
    )
    parser.add_argument(
        "--avg-mode",
        type=str,
        default="simple",
        choices=["simple", "greedy"],
        help="Type of averaging to do: simple or greedy."
        "Default: %(default)f",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./",
        help="Path to the output directory.",
    )
    return parser


def main(args):
    ds = RemoraDataset.load_from_file(
        args.val_set_path,
        shuffle_on_iter=False,
        drop_last=False,
    )

    model_data = torch.load(args.model_weights_paths[0])

    model_params = {
        "size": model_data["model_params"]["size"],
        "kmer_len": model_data["model_params"]["kmer_len"],
        "num_out": model_data["model_params"]["num_out"],
    }

    ds.trim_kmer_context_bases(model_data["kmer_context_bases"])
    ds.trim_chunk_context(model_data["chunk_context"])

    model = model_util._load_python_model(args.model_path, **model_params)
    model = model.cuda()
    val_fp = model_util.ValidationLogger(
        os.path.join(args.output_path, "weight_averaging_log.txt")
    )
    new_state_dict = average_weights(
        model, args.model_weights_paths, ds, val_fp, args.avg_mode
    )

    model_data["state_dict"] = new_state_dict
    torch.save(
        model_data, os.path.join(args.output_path, "average_model.checkpoint")
    )


if __name__ == "__main__":
    main(get_parser().parse_args())
