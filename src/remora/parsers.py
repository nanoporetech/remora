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

from remora import constants
from remora import log, RemoraError

LOGGER = log.get_logger()


class SubcommandHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Helper function to prettier print subcommand help. This removes some
    extra lines of output when a final command parser is not selected.
    """

    def _format_action(self, action):
        parts = super(SubcommandHelpFormatter, self)._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = "\n".join(parts.split("\n")[1:])
        return parts


#####################################
# remora prepare_taiyaki_train_data #
#####################################


def register_prepare_taiyaki_train_data(parser):
    subparser = parser.add_parser(
        "prepare_taiyaki_train_data",
        description="Prepare Remora training data in Taiyaki format",
        help="Prepare Remora training data in Taiyaki format",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "mapped_signal_file",
        help="Taiyaki mapped signal file.",
    )
    subparser.add_argument(
        "--output-mapped-signal-file",
        default="remora_chunk_training_dataset.hdf5",
        help="Output Taiyaki mapped signal file. Default: %(default)s",
    )
    subparser.add_argument(
        "--motif",
        nargs=2,
        metavar=("MOTIF", "FOCUS_POSITION"),
        default=["N", "0"],
        help="Extract training chunks centered on a defined motif. Argument "
        "takes 2 values representing 1) sequence motif and 2) focus position "
        "within the motif. For example to restrict to CpG sites use "
        '"--motif CG 0". Default: Any context ("N 0")',
    )
    subparser.add_argument(
        "--context-bases",
        type=int,
        default=constants.DEFAULT_FOCUS_OFFSET,
        help="Number of bases to either side of central base. "
        "Default: %(default)s",
    )
    subparser.add_argument(
        "--max-chunks-per-read",
        type=int,
        default=10,
        help="Maxiumum number of chunks to extract from a single read. "
        "Default: %(default)s",
    )
    subparser.add_argument(
        "--batch-size",
        type=int,
        default=100000,
        help="Number of chunks per batch in output file. "
        "Default: %(default)s",
    )

    subparser.set_defaults(func=run_prepare_taiyaki_train_data_mod)


def run_prepare_taiyaki_train_data_mod(args):
    import atexit

    from taiyaki.mapped_signal_files import MappedSignalReader, BatchHDF5Writer

    from remora.util import get_mod_bases, Motif, validate_mod_bases
    from remora.prepare_taiyaki_train_data import extract_chunk_dataset

    LOGGER.info("Opening mapped signal files")
    input_msf = MappedSignalReader(args.mapped_signal_file)
    atexit.register(input_msf.close)
    alphabet_info = input_msf.get_alphabet_information()
    output_msf = BatchHDF5Writer(
        args.output_mapped_signal_file,
        alphabet_info,
        batch_size=args.batch_size,
    )
    atexit.register(output_msf.close)
    motif = Motif(*args.motif)
    mod_bases = get_mod_bases(
        alphabet_info.alphabet, alphabet_info.collapse_alphabet
    )
    if len(mod_bases) > 0:
        validate_mod_bases(
            mod_bases,
            motif,
            alphabet_info.alphabet,
            alphabet_info.collapse_alphabet,
        )
    extract_chunk_dataset(
        input_msf,
        output_msf,
        motif,
        args.context_bases,
        args.max_chunks_per_read,
    )


######################
# remora train_model #
######################


def register_train_model(parser):
    subparser = parser.add_parser(
        "train_model",
        description="Train Remora model",
        help="Train Remora model",
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
        default=constants.DEFAULT_VAL_PROP,
        type=float,
        help="Proportion of the dataset to be used as validation. "
        "Default: %(default)f",
    )
    data_grp.add_argument(
        "--focus-offset",
        default=constants.DEFAULT_FOCUS_OFFSET,
        type=int,
        help="Offset into stored chunks to be predicted. Default: %(default)d",
    )
    data_grp.add_argument(
        "--motif",
        nargs=2,
        metavar=("MOTIF", "REL_POSITION"),
        default=["N", 0],
        help="Restrict prediction to a defined motif. Argument takes 2 "
        "values representing 1) sequence motif and 2) relative "
        "focus position within the motif. For example to restrict to CpG "
        'sites use "--motif CG 0". Default: All bases',
    )
    data_grp.add_argument(
        "--chunk-context",
        default=constants.DEFAULT_CHUNK_CONTEXT,
        type=int,
        nargs=2,
        help="Number of context signal points or bases to select around the "
        "central position. Signal or base positions is determined by whether "
        "the input model takes fixed signal or fixed sequence length inputs. "
        "Default: %(default)s",
    )
    data_grp.add_argument(
        "--kmer-context-bases",
        nargs=2,
        default=constants.DEFAULT_KMER_CONTEXT_BASES,
        type=int,
        help="Definition of k-mer (derived from the reference) passed into "
        "the model along with each signal position. Default: %(default)s",
    )
    data_grp.add_argument(
        "--batch-size",
        default=constants.DEFAULT_BATCH_SIZE,
        type=int,
        help="Number of samples per batch. Default: %(default)d",
    )

    out_grp = subparser.add_argument_group("Output Arguments")
    out_grp.add_argument(
        "--output-path",
        default="remora_train_results",
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
    mdl_grp.add_argument(
        "--model", required=True, help="Model architecture file (required)"
    )
    mdl_grp.add_argument(
        "--size",
        type=int,
        default=constants.DEFAULT_NN_SIZE,
        help="Model layer size. Default: %(default)d",
    )
    mdl_grp.add_argument(
        "--mod-bases", help="Single letter codes for modified bases to predict."
    )
    mdl_grp.add_argument(
        "--base-pred",
        action="store_true",
        help="Train to predict bases and not mods.",
    )

    train_grp = subparser.add_argument_group("Training Arguments")
    train_grp.add_argument(
        "--epochs",
        default=constants.DEFAULT_EPOCHS,
        type=int,
        help="Number of training epochs. Default: %(default)d",
    )
    train_grp.add_argument(
        "--optimizer",
        default=constants.OPTIMIZERS[0],
        choices=constants.OPTIMIZERS,
        help="Optimizer setting. Default: %(default)s",
    )
    train_grp.add_argument(
        "--lr",
        default=constants.DEFAULT_LR,
        type=float,
        help="Learning rate setting. Default: %(default)f",
    )
    train_grp.add_argument(
        "--lr-decay-step",
        default=constants.DEFAULT_DECAY_STEP,
        type=int,
        help="Learning decay step setting. Default: %(default)d",
    )
    train_grp.add_argument(
        "--lr-decay-gamma",
        default=constants.DEFAULT_DECAY_GAMMA,
        type=float,
        help="Learning decay gamma setting. Default: %(default)f",
    )
    train_grp.add_argument(
        "--weight-decay",
        default=constants.DEFAULT_WEIGHT_DECAY,
        type=float,
        help="Weight decay setting. Default: %(default)f",
    )
    train_grp.add_argument(
        "--seed", default=1, type=int, help="Seed value. Default: %(default)d"
    )

    comp_grp = subparser.add_argument_group("Compute Arguments")
    comp_grp.add_argument(
        "--device",
        type=int,
        help="ID of GPU that is used for training. Default: Use CPU.",
    )

    subparser.set_defaults(func=run_train_model)


def run_train_model(args):
    from remora.train_model import train_model
    from remora.util import Motif

    out_path = Path(args.output_path)
    if args.overwrite:
        if out_path.is_dir():
            rmtree(out_path)
        elif out_path.exists():
            out_path.unlink()
    elif out_path.exists():
        raise RemoraError("Refusing to overwrite existing training directory.")
    out_path.mkdir(parents=True, exist_ok=True)
    log.init_logger(os.path.join(out_path, "log.txt"))
    motif = Motif(*args.motif)
    # TODO preprocess some step to reduce args to train_model
    train_model(
        args.seed,
        args.device,
        out_path,
        args.dataset_path,
        args.num_chunks,
        motif,
        args.focus_offset,
        args.chunk_context,
        args.val_prop,
        args.batch_size,
        args.model,
        args.size,
        args.mod_bases,
        args.base_pred,
        args.optimizer,
        args.lr,
        args.weight_decay,
        args.lr_decay_step,
        args.lr_decay_gamma,
        args.epochs,
        args.save_freq,
        args.kmer_context_bases,
    )


################
# remora infer #
################


def register_infer(parser):
    subparser = parser.add_parser(
        "infer",
        description="Run a model for inference on a given dataset.",
        help="Use modified base model for inference.",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "dataset_path",
        help="Taiyaki mapped signal file on which to perform inference.",
    )
    subparser.add_argument(
        "checkpoint_path",
        help="Path to a pretrained model checkpoint.",
    )
    subparser.add_argument(
        "--model-path",
        help="Path to a model architecture. Default: Use path from checkpoint.",
    )
    subparser.add_argument(
        "--focus-offset",
        default=constants.DEFAULT_FOCUS_OFFSET,
        type=int,
        help="Offset into stored chunks to be inferred. Default: %(default)d",
    )
    subparser.add_argument(
        "--full",
        action="store_true",
        help="Make predictions at all motif matches. Default: Only predict at "
        "--focus-offset position in each read (for chunk training dataset "
        "validation).",
    )
    subparser.add_argument(
        "--output-path",
        default="remora_infer_results",
        help="Path to the output files. Default: %(default)s",
    )
    subparser.add_argument(
        "--device",
        type=int,
        help="ID of GPU that is used for inference. Default: CPU",
    )
    subparser.add_argument(
        "--batch-size",
        default=200,
        type=int,
        help="Number of input units per batch. Default: %(default)d",
    )
    subparser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if existing.",
    )

    subparser.set_defaults(func=run_infer)


def run_infer(args):
    from remora.inference import infer

    out_path = Path(args.output_path)
    if args.overwrite:
        if out_path.is_dir():
            rmtree(out_path)
        elif out_path.exists():
            out_path.unlink()
    elif out_path.exists():
        raise RemoraError("Refusing to overwrite existing inference results.")
    out_path.mkdir(parents=True, exist_ok=True)
    log.init_logger(os.path.join(out_path, "log.txt"))

    infer(
        out_path,
        args.dataset_path,
        args.checkpoint_path,
        args.model_path,
        args.batch_size,
        args.device,
        args.focus_offset,
        args.full,
    )
