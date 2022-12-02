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

import os
import sys
import atexit
import argparse
from pathlib import Path
from shutil import rmtree

from remora import constants
from remora import log, RemoraError

LOGGER = log.get_logger()


class SubcommandHelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    """Helper function to prettier print subcommand help. This removes some
    extra lines of output when a final command parser is not selected.
    """

    def _format_action(self, action):
        parts = super(SubcommandHelpFormatter, self)._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = "\n".join(parts.split("\n")[1:])
        return parts


##################
# remora dataset #
##################


def register_dataset(parser):
    subparser = parser.add_parser(
        "dataset",
        description="Remora dataset operations",
        help="Create or perform operations on a Remora dataset",
        formatter_class=SubcommandHelpFormatter,
    )
    ssubparser = subparser.add_subparsers(title="dataset commands")
    #  Since `dataset` has several sub-commands, print help as default
    subparser.set_defaults(func=lambda x: subparser.print_help())
    #  Register dataset sub commands
    register_dataset_prepare(ssubparser)
    register_dataset_split(ssubparser)
    register_dataset_merge(ssubparser)
    register_dataset_inspect(ssubparser)


def register_dataset_prepare(parser):
    subparser = parser.add_parser(
        "prepare",
        description="Prepare Remora training dataset",
        help="Prepare Remora training dataset.",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "pod5",
        help="POD5 file corresponding to bam file.",
    )
    subparser.add_argument(
        "bam",
        help="BAM file containing mv tags.",
    )

    out_grp = subparser.add_argument_group("Output Arguments")
    out_grp.add_argument(
        "--output-remora-training-file",
        default="remora_training_dataset.npz",
        help="Output Remora training dataset file. Default: %(default)s",
    )
    out_grp.add_argument(
        "--log-filename",
        help="Log filename. Default: Don't output log file.",
    )

    data_grp = subparser.add_argument_group("Data Arguments")
    data_grp.add_argument(
        "--motif",
        nargs=2,
        action="append",
        metavar=("MOTIF", "FOCUS_POSITION"),
        help="Extract training chunks centered on a defined motif. Argument "
        "takes 2 values representing 1) sequence motif and 2) focus position "
        "within the motif. For example to restrict to CpG sites use "
        '"--motif CG 0". Default: Any context ("N 0")',
    )
    data_grp.add_argument(
        "--focus-reference-positions",
        help="BED file containing reference positions around which to extract "
        "training chunks.",
    )
    data_grp.add_argument(
        "--chunk-context",
        default=constants.DEFAULT_CHUNK_CONTEXT,
        type=int,
        nargs=2,
        help="Number of context signal points to select around the central "
        "position. Default: %(default)s",
    )
    data_grp.add_argument(
        "--min-samples-per-base",
        type=int,
        default=constants.DEFAULT_MIN_SAMPLES_PER_BASE,
        help="Minimum number of samples per base. This sets the size of the "
        "ragged arrays of chunk sequences. Default: %(default)s",
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
        "--max-chunks-per-read",
        type=int,
        default=15,
        help="Maxiumum number of chunks to extract from a single read. "
        "Default: %(default)s",
    )
    data_grp.add_argument(
        "--base-start-justify",
        action="store_true",
        help="Justify extracted chunk against the start of the base of "
        "interest. Default justifies chunk to middle of signal of the base "
        "of interest.",
    )
    data_grp.add_argument(
        "--offset",
        default=0,
        type=int,
        help="Offset selected chunk position by a number of bases. "
        "Default: %(default)d",
    )
    data_grp.add_argument(
        "--num-reads",
        default=None,
        type=int,
        help="Number of reads.",
    )
    data_grp.add_argument(
        "--base-call-anchor",
        action="store_true",
        help="makes dataset from base call sequence instead of aligned "
        "reference sequence",
    )

    refine_grp = subparser.add_argument_group("Signal Mapping Refine Arguments")
    refine_grp.add_argument(
        "--refine-kmer-level-table",
        help="Tab-delimited file containing no header and two fields: "
        "1. string k-mer sequence and 2. float expected normalized level. "
        "All k-mers must be the same length and all combinations of the bases "
        "'ACGT' must be present in the file.",
    )
    refine_grp.add_argument(
        "--refine-rough-rescale",
        action="store_true",
        help="Apply a rough rescaling using quantiles of signal+move table "
        "and levels.",
    )
    refine_grp.add_argument(
        "--refine-scale-iters",
        default=constants.DEFAULT_REFINE_SCALE_ITERS,
        type=int,
        help="Number of iterations of signal mapping refinement and signal "
        "re-scaling to perform. Set to 0 (default) in order to perform signal "
        "mapping refinement, but skip re-scaling. Set to -1 to skip signal "
        "mapping (potentially using levels for rough rescaling).",
    )
    refine_grp.add_argument(
        "--refine-half-bandwidth",
        default=constants.DEFAULT_REFINE_HBW,
        type=int,
        help="Half bandwidth around signal mapping over which to search for "
        "new path.",
    )
    refine_grp.add_argument(
        "--refine-algo",
        default=constants.DEFAULT_REFINE_ALGO,
        choices=constants.REFINE_ALGOS,
        help="Refinement algorithm to apply (if kmer level table is provided).",
    )
    refine_grp.add_argument(
        "--refine-short-dwell-parameters",
        default=constants.DEFAULT_REFINE_SHORT_DWELL_PARAMS,
        type=float,
        nargs=3,
        metavar=("TARGET", "LIMIT", "WEIGHT"),
        help="Short dwell penalty refiner parameters. Dwells shorter than "
        "LIMIT will be penalized a value of `WEIGHT * (dwell - TARGET)^2`. "
        "Default: %(default)s",
    )

    label_grp = subparser.add_argument_group("Label Arguments")
    label_grp.add_argument(
        "--mod-base",
        nargs=2,
        metavar=("SINGLE_LETTER_CODE", "MOD_BASE"),
        default=None,
        help="Modified base information. Example: `--mod-base m 5mC`",
    )
    label_grp.add_argument(
        "--mod-base-control",
        action="store_true",
        help="Is this a modified bases control sample?",
    )
    label_grp.add_argument(
        "--base-pred",
        action="store_true",
        help="Train to predict bases (SNPs) and not mods.",
    )

    comp_grp = subparser.add_argument_group("Compute Arguments")
    comp_grp.add_argument(
        "--num-extract-alignment-workers",
        type=int,
        default=1,
        help="Number of signal extraction workers. Default: %(default)d",
    )
    comp_grp.add_argument(
        "--num-extract-chunks-workers",
        type=int,
        default=1,
        help="Number of chunk extraction workers. If performing signal "
        "refinement this should be increased. Default: %(default)d",
    )

    subparser.set_defaults(func=run_dataset_prepare)


def run_dataset_prepare(args):
    from remora.io import parse_bed
    from remora.util import Motif
    from remora.refine_signal_map import SigMapRefiner
    from remora.prepare_train_data import extract_chunk_dataset

    if args.log_filename is not None:
        log.init_logger(args.log_filename)
    if args.mod_base is None and not args.mod_base_control:
        LOGGER.error("Must specify either --mod-base or --mod-base-control")
        sys.exit(1)
    motifs = [("N", 0)] if args.motif is None else args.motif
    motifs = [Motif(*mo) for mo in motifs]
    focus_ref_pos = (
        None
        if args.focus_reference_positions is None
        else parse_bed(args.focus_reference_positions)
    )

    sig_map_refiner = SigMapRefiner(
        kmer_model_filename=args.refine_kmer_level_table,
        do_rough_rescale=args.refine_rough_rescale,
        scale_iters=args.refine_scale_iters,
        algo=args.refine_algo,
        half_bandwidth=args.refine_half_bandwidth,
        sd_params=args.refine_short_dwell_parameters,
        do_fix_guage=True,
    )
    extract_chunk_dataset(
        args.bam,
        args.pod5,
        args.output_remora_training_file,
        args.mod_base,
        args.mod_base_control,
        motifs,
        focus_ref_pos,
        args.chunk_context,
        args.min_samples_per_base,
        args.max_chunks_per_read,
        sig_map_refiner,
        args.base_pred,
        args.kmer_context_bases,
        args.base_start_justify,
        args.offset,
        args.num_reads,
        args.num_extract_alignment_workers,
        args.num_extract_chunks_workers,
        base_call_anchor=args.base_call_anchor,
    )


def register_dataset_split(parser):
    subparser = parser.add_parser(
        "split",
        description="Split Remora dataset",
        help="Split Remora dataset",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "input_remora_dataset",
        help="Remora training dataset to be split",
    )
    subparser.add_argument(
        "--output-basename",
        default="split_remora_dataset",
        help="Basename for output datasets. Default: %(default)s",
    )
    subparser.add_argument(
        "--val-prop",
        type=float,
        help="The proportion of data to be split into validation set, where "
        "val-prop in [0,0.5).",
    )
    subparser.add_argument(
        "--val-num",
        type=int,
        help="Number of validation chunks to select.",
    )
    subparser.add_argument(
        "--unstratified",
        action="store_true",
        help="For --val-prop split, perform unstratified splitting. Default "
        "will perform split stratified over labels.",
    )
    subparser.add_argument(
        "--by-label",
        action="store_true",
        help="Split dataset into one dataset for each unique label.",
    )
    subparser.set_defaults(func=run_dataset_split)


def run_dataset_split(args):
    from remora.data_chunks import RemoraDataset

    dataset = RemoraDataset.load_from_file(
        args.input_remora_dataset,
        shuffle_on_iter=False,
        drop_last=False,
    )
    LOGGER.info(f"Loaded dataset summary:\n{dataset.summary}")

    if args.by_label:
        for label, label_dataset in dataset.split_by_label():
            label_dataset.save(f"{args.output_basename}.{label}.npz")
            LOGGER.info(
                f"Wrote {label_dataset.nchunks} chunks to "
                f"{args.output_basename}.{label}.npz"
            )
    else:
        trn_set, val_set = dataset.split_data(
            val_prop=args.val_prop,
            val_num=args.val_num,
            stratified=not args.unstratified,
        )
        LOGGER.info(
            f"Train set label distribution: {trn_set.get_label_counts()}"
        )
        LOGGER.info(f"Val set label distribution: {val_set.get_label_counts()}")
        trn_set.save(f"{args.output_basename}.split_train.npz")
        val_set.save(f"{args.output_basename}.split_val.npz")


def register_dataset_inspect(parser):
    subparser = parser.add_parser(
        "inspect",
        description="Inspect Remora dataset",
        help="Inspect Remora dataset",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "remora_dataset_path",
        help="Remora training dataset",
    )
    subparser.set_defaults(func=run_dataset_inspect)


def run_dataset_inspect(args):
    from remora.data_chunks import RemoraDataset

    dataset = RemoraDataset.load_from_file(
        args.remora_dataset_path,
        shuffle_on_iter=False,
        drop_last=False,
    )
    print(f"Dataset summary:\n{dataset.summary}")


def register_dataset_merge(parser):
    subparser = parser.add_parser(
        "merge",
        description="Merge Remora datasets",
        help="Merge Remora datasets",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "--input-dataset",
        nargs=2,
        action="append",
        help="1) Remora training dataset path and 2) max number of chunks "
        "to extract from this dataset.",
    )
    subparser.add_argument(
        "--output-dataset",
        required=True,
        help="Output path for dataset",
    )
    subparser.add_argument(
        "--balance",
        action="store_true",
        help="Automatically balance classes when merging",
    )
    subparser.set_defaults(func=run_dataset_merge)


def run_dataset_merge(args):
    from remora.data_chunks import merge_datasets

    input_datasets = [
        (ds_path, int(num_chunks)) for ds_path, num_chunks in args.input_dataset
    ]
    output_dataset = merge_datasets(input_datasets, args.balance)
    output_dataset.save(args.output_dataset)


################
# remora model #
################


def register_model(parser):
    subparser = parser.add_parser(
        "model",
        description="Remora model operations",
        help="Train or perform operations on Remora models",
        formatter_class=SubcommandHelpFormatter,
    )
    ssubparser = subparser.add_subparsers(title="model commands")
    #  Since `model` has several sub-commands, print help as default
    subparser.set_defaults(func=lambda x: subparser.print_help())
    #  Register model sub commands
    register_model_train(ssubparser)
    register_model_export(ssubparser)
    register_model_list_pretrained(ssubparser)
    register_model_download(ssubparser)


def register_model_train(parser):
    subparser = parser.add_parser(
        "train",
        description="Train Remora model",
        help="Train Remora model",
        formatter_class=SubcommandHelpFormatter,
    )

    subparser.add_argument(
        "remora_dataset_path",
        help="Remora training dataset",
    )

    data_grp = subparser.add_argument_group("Data Arguments")
    data_grp.add_argument(
        "--val-prop",
        default=constants.DEFAULT_VAL_PROP,
        type=float,
        help="Proportion of the dataset to be used as validation. "
        "Default: %(default)f",
    )
    data_grp.add_argument(
        "--batch-size",
        default=constants.DEFAULT_BATCH_SIZE,
        type=int,
        help="Number of samples per batch. Default: %(default)d",
    )
    data_grp.add_argument(
        "--chunk-context",
        type=int,
        nargs=2,
        help="Override chunk context from data prep. Number of context signal "
        "points to select around the central position.",
    )
    data_grp.add_argument(
        "--kmer-context-bases",
        nargs=2,
        type=int,
        help="Override kmer context bases from data prep. Definition of "
        "k-mer (derived from the reference) passed into the model along with "
        "each signal position.",
    )
    data_grp.add_argument(
        "--ext-val",
        nargs="+",
        help="Path(s) to the external validation Remora datasets.",
    )
    data_grp.add_argument(
        "--balance",
        action="store_true",
        help="Balance classes exactly prior to training",
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
        "--scheduler",
        default=None,
        help="Scheduler setting. Default: %(default)s",
    )
    train_grp.add_argument(
        "--lr",
        default=constants.DEFAULT_LR,
        type=float,
        help="Learning rate setting. Default: %(default)f",
    )
    train_grp.add_argument(
        "--weight-decay",
        default=constants.DEFAULT_WEIGHT_DECAY,
        type=float,
        help="Weight decay setting. Default: %(default)f",
    )
    train_grp.add_argument(
        "--early-stopping",
        default=10,
        type=int,
        help="Stops training after a number of epochs without improvement."
        "If set to 0 no stopping is done. Default: %(default)d",
    )
    train_grp.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Seed value. Default: Random seed",
    )
    train_grp.add_argument(
        "--filter-fraction",
        default=constants.DEFAULT_FILT_FRAC,
        type=float,
        help="Fraction of predictions to filter in validation reporting. "
        "Un-filtered validation metrics will always be reported as well. "
        "Default: %(default)f",
    )
    train_grp.add_argument(
        "--lr-sched-kwargs",
        nargs=3,
        action="append",
        default=None,
        metavar=("NAME", "VALUE", "TYPE"),
    )
    train_grp.add_argument(
        "--balanced-batch",
        action="store_true",
        help="Balance classes exactly for each batch in training",
    )

    comp_grp = subparser.add_argument_group("Compute Arguments")
    comp_grp.add_argument(
        "--device",
        type=int,
        help="ID of GPU that is used for training. Default: Use CPU.",
    )

    subparser.set_defaults(func=run_model_train)


def run_model_train(args):
    from remora.train_model import train_model

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
    train_model(
        args.seed,
        args.device,
        out_path,
        args.remora_dataset_path,
        args.chunk_context,
        args.kmer_context_bases,
        args.val_prop,
        args.batch_size,
        args.model,
        args.size,
        args.optimizer,
        args.lr,
        args.scheduler,
        args.weight_decay,
        args.epochs,
        args.save_freq,
        args.early_stopping,
        args.filter_fraction,
        args.ext_val,
        args.lr_sched_kwargs,
        args.balance,
        args.balanced_batch,
    )


def register_model_export(parser):
    subparser = parser.add_parser(
        "export",
        description="Export a model to TorchScript format for inference.",
        help="Export a model to TorchScript format for inference.",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "checkpoint_path",
        help="Path to a pretrained model checkpoint.",
    )
    subparser.add_argument(
        "output_path",
        help="Path to save the model file, or the directory in which to "
        "save the dorado tensor files if '--format dorado' has been specified.",
    )
    subparser.add_argument(
        "--model-path",
        help="Path to a model architecture. Default: Use path from checkpoint.",
    )
    subparser.add_argument(
        "--format",
        default="torchscript",
        choices=["dorado", "torchscript"],
        help="Export format. Default: torchscript",
    )

    subparser.set_defaults(func=run_model_export)


def run_model_export(args):
    from remora.model_util import (
        continue_from_checkpoint,
        load_torchscript_model,
        export_model_dorado,
        export_model_torchscript,
    )

    LOGGER.info("Loading model")
    try:
        model, ckpt = load_torchscript_model(args.checkpoint_path)
        LOGGER.info("Loaded a torchscript model")
    except RuntimeError:
        ckpt, model = continue_from_checkpoint(
            args.checkpoint_path, args.model_path
        )
        LOGGER.info("Loaded model from checkpoint")

    if args.format == "dorado":
        LOGGER.info("Exporting model to dorado format")
        export_model_dorado(ckpt, model, args.output_path)
    elif args.format == "torchscript":
        LOGGER.info("Exporting model to TorchScript format")
        export_model_torchscript(ckpt, model, args.output_path)
    else:
        raise RemoraError(f"Invalid export format: {args.format}")


def register_model_list_pretrained(parser):
    subparser = parser.add_parser(
        "list_pretrained",
        description="List pre-trained modified base models.",
        help="List pre-trained modified base models.",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument("--pore", help="specify pore type")
    subparser.add_argument(
        "--basecall-model-type",
        help="specify the basecaller model type (e.g., fast, hac or sup)",
    )
    subparser.add_argument(
        "--basecall-model-version", help="specify the version of the basecaller"
    )
    subparser.add_argument(
        "--modified-bases",
        nargs="+",
        help="specify the modified base models you are interested in",
    )
    subparser.add_argument(
        "--remora-model-type",
        help="specify the motif or context that the remora model has been "
        "trained on",
    )
    subparser.add_argument(
        "--remora-model-version", help="specify the remora model version"
    )
    subparser.set_defaults(func=run_list_pretrained)


def run_list_pretrained(args):
    from remora.model_util import get_pretrained_models
    from tabulate import tabulate

    models, header = get_pretrained_models(
        args.pore,
        args.basecall_model_type,
        args.basecall_model_version,
        args.modified_bases,
        args.remora_model_type,
        args.remora_model_version,
    )
    LOGGER.info(
        "Remora pretrained modified base models:\n"
        + tabulate(models, headers=header, showindex=False)
    )


def register_model_download(parser):
    subparser = parser.add_parser(
        "download",
        description="Download pre-trained modified base models.",
        help="Download pre-trained modified base models.",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument("--pore", help="specify pore type")
    subparser.add_argument(
        "--basecall-model-type",
        help="specify the basecaller model type (e.g., fast, hac or sup)",
    )
    subparser.add_argument(
        "--basecall-model-version", help="specify the version of the basecaller"
    )
    subparser.add_argument(
        "--modified-bases",
        nargs="+",
        help="specify the modified base models you are interested in",
    )
    subparser.add_argument(
        "--remora-model-type",
        help="specify the motif or context that the remora model has been "
        "trained on",
    )
    subparser.add_argument(
        "--remora-model-version", help="specify the remora model version"
    )
    subparser.set_defaults(func=run_download)


def run_download(args):
    from remora.model_util import get_pretrained_models
    from remora.download import ModelDownload
    import pkg_resources

    models, header = get_pretrained_models(
        args.pore,
        args.basecall_model_type,
        args.basecall_model_version,
        args.modified_bases,
        args.remora_model_type,
        args.remora_model_version,
    )
    path = pkg_resources.resource_filename(
        "remora",
        constants.MODEL_DATA_DIR_NAME,
    )
    out_path = Path(path)
    out_path.mkdir(parents=True, exist_ok=True)
    model_dl = ModelDownload(out_path)
    for model_url in models["Remora_Model_URL"]:
        if model_url != "":
            model_dl.download(model_url)


################
# remora infer #
################


def register_infer(parser):
    subparser = parser.add_parser(
        "infer",
        description="Perform Remora model inference",
        help="Perform Remora model inference",
        formatter_class=SubcommandHelpFormatter,
    )
    ssubparser = subparser.add_subparsers(title="infer commands")
    # Since `infer` has several sub-commands, print help as default
    subparser.set_defaults(func=lambda x: subparser.print_help())
    # Register infer sub commands
    register_infer_from_pod5_and_bam(ssubparser)
    register_infer_duplex_from_pod5_and_bam(ssubparser)


def register_infer_from_pod5_and_bam(parser):
    subparser = parser.add_parser(
        "from_pod5_and_bam",
        description="Infer modified bases from pod5 and bam inputs",
        help="Run inference on pod5s and alignments",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "pod5",
        help="POD5 file corresponding to bam file.",
    )
    subparser.add_argument(
        "in_bam",
        help="BAM file containing mv tags.",
    )

    out_grp = subparser.add_argument_group("Output Arguments")
    out_grp.add_argument(
        "--out-bam",
        help="Output BAM path.",
    )
    out_grp.add_argument(
        "--log-filename",
        help="Log filename. Default: Don't output log file.",
    )

    mdl_grp = subparser.add_argument_group("Model Arguments")
    mdl_grp.add_argument(
        "--model",
        help="Path to a pretrained model in torchscript format.",
    )
    mdl_grp.add_argument(
        "--pore",
        help="Choose the type of pore the Remora model has been trained on "
        "(e.g. dna_r10.4_e8.1)",
    )
    mdl_grp.add_argument(
        "--basecall-model-type",
        help="Choose the basecaller model type (choose from fast, hac or sup)",
    )
    mdl_grp.add_argument(
        "--basecall-model-version",
        help="Choose a specific basecaller version",
    )
    mdl_grp.add_argument(
        "--modified-bases",
        nargs="+",
        help="Long name of the modified bases to call (e.g., 5mc, 5hmc).",
    )
    mdl_grp.add_argument(
        "--remora-model-type",
        help="Choose the specific motif of the model you want to load. "
        "If None, load CG model.",
    )
    mdl_grp.add_argument(
        "--remora-model-version",
        type=int,
        help="Choose the remora model version. If None, use latest.",
    )

    data_grp = subparser.add_argument_group("Data Arguments")
    data_grp.add_argument(
        "--num-reads",
        default=None,
        type=int,
        help="Number of reads.",
    )
    data_grp.add_argument(
        "--reference-anchored",
        action="store_true",
        help="Infer per-read modified bases against reference bases instead "
        "of basecalls.",
    )

    comp_grp = subparser.add_argument_group("Compute Arguments")
    comp_grp.add_argument(
        "--device",
        type=int,
        help="ID of GPU that is used for inference. Default: CPU only",
    )
    comp_grp.add_argument(
        "--num-extract-alignment-workers",
        type=int,
        default=1,
        help="Number of signal extraction workers. Default: %(default)d",
    )
    comp_grp.add_argument(
        "--num-prepare-batch-workers",
        type=int,
        default=1,
        help="Number of batch preparation workers. Default: %(default)d",
    )
    comp_grp.add_argument(
        "--num-infer-workers",
        type=int,
        default=1,
        help="Number of chunk extraction workers. If performing signal "
        "refinement this should be increased. Default: %(default)d",
    )
    comp_grp.add_argument(
        "--batch-size",
        default=constants.DEFAULT_BATCH_SIZE,
        type=int,
        help="Number of input units per batch. Default: %(default)d",
    )

    subparser.set_defaults(func=run_infer_from_pod5_and_bam)


def register_infer_duplex_from_pod5_and_bam(parser):
    duplex_delim_flag = "--duplex-delim"
    subparser = parser.add_parser(
        "duplex_from_pod5_and_bam",
        description="Infer modified bases on duplex reads from pod5 and bam "
        "inputs",
        help="Run inference on pod5s simplex reads and duplex alignments with "
        "duplex pairs",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "pod5",
        help="POD5 file corresponding to bam file.",
    )
    subparser.add_argument(
        "simplex_bam",
        help="Base called BAM file containing mv tags.",
    )
    subparser.add_argument(
        "duplex_bam",
        help="BAM file containing duplex base called sequences (and optional "
        "reference mappings). Record names may either be the template read_id "
        "or template<delim>complement. The value of <delim> can be set with "
        f"{duplex_delim_flag}.",
    )
    subparser.add_argument(
        "duplex_read_pairs",
        help="Whitespace separated plain text file containing read ID pairs, no"
        "header.",
    )
    subparser.add_argument(
        duplex_delim_flag,
        help="deliminator string between template and complement read "
        "ids in the duplex BAM",
        default=";",
    )

    out_grp = subparser.add_argument_group("Output Arguments")
    out_grp.add_argument(
        "--out-bam",
        help="Output BAM path.",
    )
    out_grp.add_argument(
        "--log-filename",
        help="Log filename. Default: Don't output log file.",
    )

    mdl_grp = subparser.add_argument_group("Model Arguments")
    mdl_grp.add_argument(
        "--model",
        help="Path to a pretrained model in torchscript format.",
    )
    mdl_grp.add_argument(
        "--pore",
        help="Choose the type of pore the Remora model has been trained on "
        "(e.g. dna_r10.4_e8.1)",
    )
    mdl_grp.add_argument(
        "--basecall-model-type",
        help="Choose the basecaller model type (choose from fast, hac or sup)",
    )
    mdl_grp.add_argument(
        "--basecall-model-version",
        help="Choose a specific basecaller version",
    )
    mdl_grp.add_argument(
        "--modified-bases",
        nargs="+",
        help="Long name of the modified bases to call (e.g., 5mc, 5hmc).",
    )
    mdl_grp.add_argument(
        "--remora-model-type",
        help="Choose the specific motif of the model you want to load. "
        "If None, load CG model.",
    )
    mdl_grp.add_argument(
        "--remora-model-version",
        type=int,
        help="Choose the remora model version. If None, use latest.",
    )

    data_grp = subparser.add_argument_group("Data Arguments")
    data_grp.add_argument(
        "--num-reads",
        default=None,
        type=int,
        help="Number of reads.",
    )

    comp_grp = subparser.add_argument_group("Compute Arguments")
    comp_grp.add_argument(
        "--device",
        type=int,
        help="ID of GPU that is used for inference. Default: CPU only",
    )
    comp_grp.add_argument(
        "--num-extract-alignment-workers",
        type=int,
        default=1,
        help="Number of IO extraction workers. Default: %(default)d",
    )
    comp_grp.add_argument(
        "--num-duplex-prep-workers",
        type=int,
        default=1,
        help="Number of duplex prep workers (tends to bottleneck). Default:"
        "%(default)d",
    )
    comp_grp.add_argument(
        "--num-infer-workers",
        type=int,
        default=1,
        help="Number of chunk extraction workers. If performing signal "
        "refinement this should be increased. Default: %(default)d",
    )

    subparser.set_defaults(func=run_infer_from_pod5_and_bam_duplex)


def _unpack_model_kw_args(args) -> dict:
    if args.model and not os.path.exists(args.model):
        raise ValueError(f"didn't find model file at {args.model}")

    model_kwargs = {
        "model_filename": args.model,
        "pore": args.pore,
        "basecall_model_type": args.basecall_model_type,
        "basecall_model_version": args.basecall_model_version,
        "modified_bases": args.modified_bases,
        "remora_model_type": args.remora_model_type,
        "remora_model_version": args.remora_model_version,
        "device": args.device,
    }
    return model_kwargs


def run_infer_from_pod5_and_bam(args):
    from remora.model_util import load_model
    from remora.inference import infer_from_pod5_and_bam

    if args.log_filename is not None:
        log.init_logger(args.log_filename)
    # test that model can be loaded in parent process
    model_kwargs = _unpack_model_kw_args(args)
    model, model_metadata = load_model(**model_kwargs, quiet=False)
    infer_from_pod5_and_bam(
        pod5_path=args.pod5,
        in_bam_path=args.in_bam,
        model=model,
        model_metadata=model_metadata,
        out_bam_path=args.out_bam,
        num_reads=args.num_reads,
        num_extract_alignment_workers=args.num_extract_alignment_workers,
        num_prep_batch_workers=args.num_prepare_batch_workers,
        num_infer_workers=args.num_infer_workers,
        batch_size=args.batch_size,
        ref_anchored=args.reference_anchored,
    )


def run_infer_from_pod5_and_bam_duplex(args):
    from remora.model_util import load_model
    from remora.inference import infer_duplex

    if args.log_filename is not None:
        log.init_logger(args.log_filename)
    model_kwargs = _unpack_model_kw_args(args)
    model, model_metadata = load_model(**model_kwargs, quiet=False)

    if not os.path.exists(args.pod5):
        raise ValueError(f"didn't find pod5 at {args.pod5}")
    if not os.path.exists(args.simplex_bam):
        raise ValueError(f"didn't find simplex bam at {args.simplex_bam}")
    if not os.path.exists(args.duplex_bam):
        raise ValueError(f"didn't find duplex bam at {args.duplex_bam}")
    if not os.path.exists(args.duplex_read_pairs):
        raise ValueError(
            f"didn't find duplex read pairs at {args.duplex_read_pairs}"
        )

    infer_duplex(
        simplex_pod5_path=args.pod5,
        simplex_bam_path=args.simplex_bam,
        duplex_bam_path=args.duplex_bam,
        pairs_path=args.duplex_read_pairs,
        model=model,
        model_metadata=model_metadata,
        out_bam=args.out_bam,
        num_extract_alignment_threads=args.num_extract_alignment_workers,
        num_duplex_prep_workers=args.num_duplex_prep_workers,
        num_infer_threads=args.num_infer_workers,
        num_reads=args.num_reads,
        duplex_deliminator=args.duplex_delim,
    )


###################
# remora validate #
###################


def register_validate(parser):
    subparser = parser.add_parser(
        "validate",
        description="Validate modified base predictions",
        help="Validate modified base predictions",
        formatter_class=SubcommandHelpFormatter,
    )
    ssubparser = subparser.add_subparsers(title="validation commands")
    # Since `validate` has several sub-commands, print help as default
    subparser.set_defaults(func=lambda x: subparser.print_help())
    # Register validate sub commands
    register_validate_from_modbams(ssubparser)
    register_validate_from_remora_dataset(ssubparser)


def register_validate_from_modbams(parser):
    subparser = parser.add_parser(
        "from_modbams",
        description="Validation with ground truth samples",
        help="Validation with ground truth samples",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "--bam-and-bed",
        required=True,
        nargs=2,
        metavar=("BAM", "GROUND_TRUTH_BED"),
        action="append",
        help="""Argument accepts 2 values. The first value is the BAM file path
        with modified base tags. The second is a bed file with ground truth
        reference positions. The name field in the ground truth bed file should
        be the single letter code for a modified base or the corresponding
        canonical base. This argument can be provided more than once for
        multiple samples.""",
    )
    subparser.add_argument(
        "--full-results-filename", help="Output per-read calls to TSV file."
    )
    subparser.add_argument(
        "--name",
        default="sample",
        help="Name of this sample/comparison. Useful when tabulating "
        "several runs.",
    )
    subparser.add_argument(
        "--pct-filt",
        type=float,
        default=10.0,
        help="Filter a specified percentage (or less given ties) of calls.",
    )
    subparser.add_argument(
        "--allow-unbalanced",
        action="store_true",
        help="Allow classes to be unbalanced for metric computation.",
    )
    subparser.add_argument(
        "--max-sites-per-read",
        type=int,
        help="Maxiumum number of sites to extract from a single read. "
        "Default: %(default)s",
    )
    subparser.add_argument(
        "--seed",
        type=int,
        help="Seed value. Default: Random seed",
    )
    subparser.add_argument(
        "--extra-bases",
        help="Extra canoncial or modified base single letter codes not in "
        "the ground truth bed files which should be added to the accepted "
        "alphabet. For example, to run a sample with canonical ground truth "
        "(C) and 5mC and 5hmC calls (m and h) modified base calls this "
        "argument would be `--extra-bases mh`",
    )
    subparser.add_argument(
        "--log-filename",
        help="Log filename. (default: Don't output log file)",
    )

    subparser.set_defaults(func=run_validate_modbams)


def run_validate_modbams(args):
    from remora.validate import validate_modbams

    if args.log_filename is not None:
        log.init_logger(args.log_filename)
    validate_modbams(
        bams_and_beds=args.bam_and_bed,
        full_results_path=args.full_results_filename,
        name=args.name,
        pct_filt=args.pct_filt,
        allow_unbalanced=args.allow_unbalanced,
        seed=args.seed,
        extra_bases=args.extra_bases,
        max_sites_per_read=args.max_sites_per_read,
    )


def register_validate_from_remora_dataset(parser):
    subparser = parser.add_parser(
        "from_remora_dataset",
        description="Run validation on external Remora dataset",
        help="Validate on Remora dataset",
        formatter_class=SubcommandHelpFormatter,
    )
    subparser.add_argument(
        "remora_dataset_path",
        help="Remora training dataset",
    )

    mdl_grp = subparser.add_argument_group("Model Arguments")
    mdl_grp.add_argument(
        "--model",
        help="Path to a pretrained model.",
    )
    mdl_grp.add_argument(
        "--pore",
        help="Choose the type of pore the Remora model has been trained on "
        "(e.g. dna_r10.4_e8.1)",
    )
    mdl_grp.add_argument(
        "--basecall-model-type",
        help="Choose the basecaller model type (choose from fast, hac or sup)",
    )
    mdl_grp.add_argument(
        "--basecall-model-version",
        help="Choose a specific basecaller version",
    )
    mdl_grp.add_argument(
        "--modified-bases",
        nargs="+",
        help="Long name of the modified bases to call (e.g., 5mc, 5hmc).",
    )
    mdl_grp.add_argument(
        "--remora-model-type",
        help="Choose the specific motif of the model you want to load. "
        "If None, load CG model.",
    )
    mdl_grp.add_argument(
        "--remora-model-version",
        type=int,
        help="Choose the remora model version. If None, use latest.",
    )

    out_grp = subparser.add_argument_group("Output Arguments")
    out_grp.add_argument(
        "--out-file",
        help="Output path for the validation result file.",
    )
    out_grp.add_argument(
        "--full-results-filename", help="Output per-read calls to TSV file."
    )

    val_grp = subparser.add_argument_group("Validation Arguments")
    val_grp.add_argument(
        "--pct-filt",
        type=float,
        default=10.0,
        help="Filter a specified percentage (or less given ties) of calls.",
    )

    comp_grp = subparser.add_argument_group("Compute Arguments")
    comp_grp.add_argument(
        "--batch-size",
        default=constants.DEFAULT_BATCH_SIZE,
        type=int,
        help="Number of input units per batch. Default: %(default)d",
    )
    comp_grp.add_argument(
        "--device",
        type=int,
        help="ID of GPU that is used for inference. Default: CPU",
    )

    subparser.set_defaults(func=run_validate_from_remora_dataset)


def run_validate_from_remora_dataset(args):
    import torch

    from remora.model_util import load_model
    from remora.validate import ValidationLogger
    from remora.data_chunks import RemoraDataset

    LOGGER.info("Loading dataset from Remora file")
    dataset = RemoraDataset.load_from_file(
        args.remora_dataset_path,
        batch_size=args.batch_size,
        shuffle_on_iter=False,
        drop_last=False,
    )

    LOGGER.info("Loading model")
    model, model_metadata = load_model(
        args.model,
        pore=args.pore,
        basecall_model_type=args.basecall_model_type,
        basecall_model_version=args.basecall_model_version,
        modified_bases=args.modified_bases,
        remora_model_type=args.remora_model_type,
        remora_model_version=args.remora_model_version,
        device=args.device,
    )

    dataset.trim_kmer_context_bases(model_metadata["kmer_context_bases"])
    dataset.trim_chunk_context(model_metadata["chunk_context"])
    LOGGER.info(f"Loaded dataset summary:\n{dataset.summary}")

    if args.out_file is None:
        out_fp = sys.stdout
    else:
        out_fp = open(args.out_file, "w", buffering=1)
        atexit.register(out_fp.close)
    if args.full_results_filename is None:
        full_results_fp = None
    else:
        full_results_fp = open(args.full_results_filename, "w", buffering=1)
        atexit.register(full_results_fp.close)

    LOGGER.info("Running validation")
    val_fp = ValidationLogger(out_fp, full_results_fp)
    val_fp.validate_model(
        model,
        model_metadata["mod_bases"],
        torch.nn.CrossEntropyLoss(),
        dataset,
        args.pct_filt / 100,
        display_progress_bar=True,
    )
