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
from remora import log

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
    ssubparser = subparser.add_subparsers(title="prepare Taiyaki")
    # Since `prepare_taiyaki_train_data` has several sub-commands,
    # print help as default
    subparser.set_defaults(func=lambda x: subparser.print_help())
    # Register  sub commands
    register_prepare_taiyaki_train_data_can(ssubparser)
    register_prepare_taiyaki_train_data_mod(ssubparser)


def register_prepare_taiyaki_train_data_can(parser):
    subparser = parser.add_parser(
        "canonical",
        description="Prepare canonical Remora training data in Taiyaki format",
        help="Prepare canonical Remora training data in Taiyaki format",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "mapped_signal_file",
        help="Taiyaki mapped signal file.",
    )
    subparser.add_argument(
        "--output-mapped-signal-file",
        default="remora_canonical_base_training_dataset.hdf5",
        help="Output Taiyaki mapped signal file. Default: %(default)s",
    )
    subparser.add_argument(
        "--context-bases",
        type=int,
        default=50,
        help="Modified base. Default: %(default)s",
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

    subparser.set_defaults(func=run_prepare_taiyaki_train_data_can)


def run_prepare_taiyaki_train_data_can(args):
    import atexit

    from taiyaki.mapped_signal_files import MappedSignalReader, BatchHDF5Writer

    from remora.prepare_taiyaki_train_data import extract_canonical_dataset

    LOGGER.info("Opening mapped signal files")
    input_msf = MappedSignalReader(args.mapped_signal_file)
    atexit.register(input_msf.close)
    output_msf = BatchHDF5Writer(
        args.output_mapped_signal_file,
        input_msf.get_alphabet_information(),
        batch_size=args.batch_size,
    )
    atexit.register(output_msf.close)
    extract_canonical_dataset(
        input_msf,
        output_msf,
        args.context_bases,
        args.max_chunks_per_read,
    )


def register_prepare_taiyaki_train_data_mod(parser):
    subparser = parser.add_parser(
        "modbase",
        description="Prepare modbase Remora training data in Taiyaki format",
        help="Prepare modbase Remora training data in Taiyaki format",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "mapped_signal_file",
        help="Taiyaki mapped signal file.",
    )
    subparser.add_argument(
        "--output-mapped-signal-file",
        default="remora_modified_base_training_dataset.hdf5",
        help="Output Taiyaki mapped signal file. Default: %(default)s",
    )
    subparser.add_argument(
        "--mod-motif",
        nargs=3,
        metavar=("BASE", "MOTIF", "REL_POSITION"),
        default=["m", "CG", 0],
        help="Extract training chunks centered on a defined motif. Argument "
        "takes 3 values representing 1) the single letter modified base(s), 2) "
        "sequence motif and 3) relative modified base position. For "
        'example to restrict to CpG sites use "--mod-motif m CG 0" (default).',
    )
    subparser.add_argument(
        "--context-bases",
        type=int,
        default=50,
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

    from remora.data_chunks import validate_motif
    from remora.prepare_taiyaki_train_data import extract_modbase_dataset

    LOGGER.info("Opening mapped signal files")
    input_msf = MappedSignalReader(args.mapped_signal_file)
    atexit.register(input_msf.close)
    output_msf = BatchHDF5Writer(
        args.output_mapped_signal_file,
        input_msf.get_alphabet_information(),
        batch_size=args.batch_size,
    )
    atexit.register(output_msf.close)
    mod_base, int_can_motif, motif_offset = validate_motif(
        input_msf, args.mod_motif
    )
    extract_modbase_dataset(
        input_msf,
        output_msf,
        mod_base,
        int_can_motif,
        motif_offset,
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
        default=0.1,
        type=float,
        help="Proportion of the dataset to be used as validation. "
        "Default: %(default)f",
    )
    data_grp.add_argument(
        "--focus-offset",
        default=50,
        type=int,
        help="Offset into stored chunks to be predicted. Default: %(default)d",
    )
    data_grp.add_argument(
        "--mod-motif",
        nargs=3,
        metavar=("BASE", "MOTIF", "REL_POSITION"),
        default=["m", "CG", 0],
        help="Extract training chunks centered on a defined motif. Argument "
        "takes 3 values representing 1) the single letter modified base(s), 2) "
        "sequence motif and 3) relative modified base position. For "
        'example to restrict to CpG sites use "--mod-motif m CG 0" (default).',
    )
    data_grp.add_argument(
        "--chunk-context",
        default=[25, 25],
        type=int,
        nargs=2,
        help="Number of context signal points or bases to select around the "
        "central position. Signal or base positions is determined by whether "
        "the input model takes fixed signal or fixed sequence length inputs. "
        "Default: %(default)s",
    )
    data_grp.add_argument(
        "--context-bases",
        nargs=2,
        default=constants.DEFAULT_CONTEXT_BASES,
        type=int,
        help="Definition of k-mer (derived from the reference) passed into "
        "the model along with each signal position. Default: %(default)s",
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
    mdl_grp.add_argument("--model", default="lstm", help="Model for training")
    mdl_grp.add_argument(
        "--size",
        type=int,
        default=64,
        help="Model layer size. Default: %(default)d",
    )
    mdl_grp.add_argument(
        "--base-pred",
        action="store_true",
        help="Train to predict bases and not mods.",
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
        type=int,
        help="ID of GPU that is used for training. Default: Use CPU.",
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
        args.mod_motif,
        args.focus_offset,
        args.chunk_context,
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
        args.base_pred,
        args.epochs,
        args.save_freq,
        args.context_bases,
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
        "--dataset-path",
        default="toy_training_data.hdf5",
        help="Dataset to detect modified bases.",
    )
    subparser.add_argument(
        "--checkpoint-path",
        default="./models",
        help="Path to a pretrained modified base model",
    )
    subparser.add_argument(
        "--full",
        action="store_true",
        help="Detect mods only on the position given by the offset or "
        "everywhere the motif matches.",
    )
    subparser.add_argument(
        "--output-path",
        default="remora_results",
        help="Path to the output files",
    )
    subparser.add_argument(
        "--device",
        default=0,
        type=int,
        help="ID of GPU that is used for inference.",
    )
    subparser.add_argument(
        "--batch-size",
        default=200,
        type=int,
        help="Number of samples per batch.",
    )
    subparser.add_argument(
        "--workers",
        default=0,
        type=int,
        dest="nb_workers",
        help="Number of workers for dataloader.",
    )
    subparser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if existing.",
    )

    subparser.set_defaults(func=run_infer)


def run_infer(args):
    from remora import RemoraError, log
    from remora.inference import infer

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

    infer(
        out_path,
        args.dataset_path,
        args.checkpoint_path,
        args.batch_size,
        args.nb_workers,
        args.device,
        args.full,
    )
