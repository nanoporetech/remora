"""Parsers module provides all implementations of command line interfaces.

Each command should implment a `register_` function and `run_` function. The
register function should take a parser and add the appropriate entries for the
command. The run function should accept the parser.parse_args() object and
execute the command.  run commands should contain minimal logic. If run
functions become too complex consider moving logic into an appropriate betta
module. Imports required for a particular command should be made inside of the
run commands to avoid loading all modules when a user simply wants the help
string.
"""

import argparse
import os
from pathlib import Path
from shutil import rmtree


class SubcommandHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Helper function to prettier print subcommand help. This removes some
    extra lines of output when a final command parser is not selected.
    """

    def _format_action(self, action):
        parts = super(SubcommandHelpFormatter, self)._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = "\n".join(parts.split("\n")[1:])
        return parts


######################
# remora train_model #
######################


def register_train_model(parser):
    subparser = parser.add_parser(
        "train_model",
        description="Train modified base model",
        help="Train modified base model",
        formatter_class=SubcommandHelpFormatter,
    )

    data_grp = subparser.add_argument_group("Data Arguments")
    data_grp.add_argument(
        "--dataset-path",
        default="remora_modified_base_training_dataset.hdf5",
        help="Training dataset. Default: %(default)s",
    )
    data_grp.add_argument(
        "--num-chunks",
        type=int,
        help="Total number of chunks loaded for training. "
        "Default: All chunks loaded",
    )
    data_grp.add_argument(
        "--val-prop",
        default=0.1,
        type=float,
        help="Proportion of the dataset to be used as validation. "
        "Default: %(default)f",
    )
    data_grp.add_argument(
        "--mod-offset",
        default=50,
        type=int,
        help="Offset into stored chunks to be predicted. Default: %(default)d",
    )
    data_grp.add_argument(
        "--mod",
        default="a",
        type=str,
        help="The mod base being considered. Default: %(default)s",
    )
    data_grp.add_argument(
        "--fixed-sequence-length-chunks",
        action="store_true",
        help="Select chunks with a fixed sequence length. Default: Fixed "
        "signal length",
    )
    data_grp.add_argument(
        "--chunk-context",
        default=[25, 25],
        type=int,
        nargs=2,
        help="Number of context signal points or bases to select around the "
        "central position. Default: %(default)s",
    )
    data_grp.add_argument(
        "--batch-size",
        default=200,
        type=int,
        help="Number of samples per batch. Default: %(default)d",
    )

    out_grp = subparser.add_argument_group("Output Arguments")
    out_grp.add_argument(
        "--output-path",
        default="remora_results",
        help="Path to the output files. Default: %(default)s",
    )
    out_grp.add_argument(
        "--save-freq",
        default=10,
        type=int,
        help="After how many epochs to save the model. Default %(default)d",
    )
    out_grp.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if existing.",
    )

    mdl_grp = subparser.add_argument_group("Model Arguments")
    # TODO convert to file input specifying the model (see taiyaki)
    mdl_grp.add_argument("--model", default="lstm", help="Model for training")
    mdl_grp.add_argument(
        "--size", default=64, help="Model layer size. Default: %(default)d"
    )

    train_grp = subparser.add_argument_group("Training Arguments")
    train_grp.add_argument(
        "--epochs",
        default=50,
        type=int,
        help="Number of training epochs. Default: %(default)d",
    )
    # TODO convert to choices devired from central module
    train_grp.add_argument(
        "--optimizer",
        default="adamw",
        help="Optimizer setting. Default: %(default)s",
    )
    train_grp.add_argument(
        "--lr",
        default=1e-5,
        type=float,
        help="Learning rate setting. Default: %(default)f",
    )
    train_grp.add_argument(
        "--lr-decay-step",
        default=10,
        type=int,
        help="Learning decay step setting. Default: %(default)d",
    )
    train_grp.add_argument(
        "--lr-decay-gamma",
        default=0.5,
        type=float,
        help="Learning decay gamma setting. Default: %(default)f",
    )
    train_grp.add_argument(
        "--weight-decay",
        default=1e-4,
        type=float,
        help="Weight decay setting. Default: %(default)f",
    )
    train_grp.add_argument(
        "--seed", default=1, type=int, help="Seed value. Default: %(default)d"
    )

    comp_grp = subparser.add_argument_group("Compute Arguments")
    comp_grp.add_argument(
        "--device",
        default=0,
        type=int,
        help="ID of GPU that is used for training. Default: %(default)d",
    )
    comp_grp.add_argument(
        "--workers",
        default=0,
        type=int,
        dest="nb_workers",
        help="Number of workers for dataloader. Default: %(default)d",
    )

    subparser.set_defaults(func=run_train_model)


def run_train_model(args):
    from remora import RemoraError, log
    from remora.train_model import train_model

    out_path = Path(args.output_path)
    if args.overwrite:
        if out_path.is_dir():
            rmtree(out_path)
        elif out_path.exists():
            out_path.unlink()
    elif out_path.exists():
        raise RemoraError("Refusing to overwrite existing table.")
    out_path.mkdir(parents=True, exist_ok=True)
    log.init_logger(os.path.join(out_path, "log.txt"))

    # TODO preprocess some step to reduce args to train_model
    train_model(
        args.seed,
        args.device,
        out_path,
        args.dataset_path,
        args.num_chunks,
        args.mod,
        args.mod_offset,
        args.chunk_context,
        args.fixed_sequence_length_chunks,
        args.val_prop,
        args.batch_size,
        args.nb_workers,
        args.model,
        args.size,
        args.optimizer,
        args.lr,
        args.weight_decay,
        args.lr_decay_step,
        args.lr_decay_gamma,
        args.epochs,
        args.save_freq,
    )


#####################
# remora infer #
#####################


def register_infer(parser):
    subparser = parser.add_parser(
        "infer",
        description="Run a model for inference on a given dataset.",
        help="Use modified base model for inference.",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "--dataset-path",
        default="toy_training_data.hdf5",
        help="Dataset to detect modified bases.",
    )
    subparser.add_argument(
        "--checkpoint-path",
        default="./models",
        help="Path to a pretrained modified base model",
    )


def run_infer(args):
    from remora.inference import infer

    infer(args)
