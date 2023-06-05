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
        help="Output Remora training dataset file.",
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
        help="""Extract training chunks centered on a defined motif. Argument
        takes 2 values representing 1) sequence motif and 2) focus position
        within the motif. For example to restrict to CpG sites use "
        "--motif CG 0". Default: Any context ("N 0")""",
    )
    data_grp.add_argument(
        "--focus-reference-positions",
        help="""BED file containing reference positions around which to extract
        training chunks.""",
    )
    data_grp.add_argument(
        "--chunk-context",
        default=constants.DEFAULT_CHUNK_CONTEXT,
        type=int,
        nargs=2,
        metavar=("NUM_BEFORE", "NUM_AFTER"),
        help="""Number of context signal points to select around the central
        position.""",
    )
    data_grp.add_argument(
        "--min-samples-per-base",
        type=int,
        default=constants.DEFAULT_MIN_SAMPLES_PER_BASE,
        help="""Minimum number of samples per base. This sets the size of the
        ragged arrays of chunk sequences.""",
    )
    data_grp.add_argument(
        "--kmer-context-bases",
        nargs=2,
        default=constants.DEFAULT_KMER_CONTEXT_BASES,
        type=int,
        metavar=("BASES_BEFORE", "BASES_AFTER"),
        help="""Definition of k-mer (derived from the reference) passed into
        the model along with each signal position.""",
    )
    data_grp.add_argument(
        "--max-chunks-per-read",
        type=int,
        default=15,
        help="Maxiumum number of chunks to extract from a single read.",
    )
    data_grp.add_argument(
        "--base-start-justify",
        action="store_true",
        help="""Justify extracted chunk against the start of the base of
        interest. Default justifies chunk to middle of signal of the base
        of interest.""",
    )
    data_grp.add_argument(
        "--offset",
        default=0,
        type=int,
        help="Offset selected chunk position by a number of bases.",
    )
    data_grp.add_argument(
        "--num-reads",
        type=int,
        help="Number of reads.",
    )
    data_grp.add_argument(
        "--basecall-anchor",
        action="store_true",
        help="""Make dataset from basecall sequence instead of aligned
        reference sequence""",
    )
    data_grp.add_argument(
        "--reverse-signal",
        action="store_true",
        help="""Is nanopore signal 3' to 5' orientation? Primarily for direct
        RNA""",
    )

    refine_grp = subparser.add_argument_group("Signal Mapping Refine Arguments")
    refine_grp.add_argument(
        "--refine-kmer-level-table",
        help="""Tab-delimited file containing no header and two fields:
        1. string k-mer sequence and 2. float expected normalized level.
        All k-mers must be the same length and all combinations of the bases
        'ACGT' must be present in the file.""",
    )
    refine_grp.add_argument(
        "--refine-rough-rescale",
        action="store_true",
        help="""Apply a rough rescaling using quantiles of signal+move table
        and levels.""",
    )
    refine_grp.add_argument(
        "--refine-scale-iters",
        default=-1,
        type=int,
        help="""Number of iterations of signal mapping refinement and signal
        re-scaling to perform. Set to 0 in order to perform signal mapping
        refinement, but skip re-scaling. Set to -1 (default) to skip signal
        mapping (potentially using levels for rough rescaling).""",
    )
    refine_grp.add_argument(
        "--refine-half-bandwidth",
        default=constants.DEFAULT_REFINE_HBW,
        type=int,
        help="""Half bandwidth around signal mapping over which to search for
        "new path.""",
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
        help="""Short dwell penalty refiner parameters. Dwells shorter than
        LIMIT will be penalized a value of `WEIGHT * (dwell - TARGET)^2`.""",
    )

    label_grp = subparser.add_argument_group("Label Arguments")
    label_grp.add_argument(
        "--mod-base",
        nargs=2,
        metavar=("SINGLE_LETTER_CODE", "MOD_BASE"),
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
        help="Number of signal extraction workers.",
    )
    comp_grp.add_argument(
        "--num-extract-chunks-workers",
        type=int,
        default=1,
        help="""Number of chunk extraction workers. If performing signal
        refinement this should be increased.""",
    )

    subparser.set_defaults(func=run_dataset_prepare)


def run_dataset_prepare(args):
    from remora.util import Motif
    from remora.io import parse_bed
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
        basecall_anchor=args.basecall_anchor,
        rev_sig=args.reverse_signal,
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
        help="Basename for output datasets.",
    )
    subparser.add_argument(
        "--val-prop",
        type=float,
        help="""The proportion of data to be split into validation set, where
        val-prop in [0,0.5).""",
    )
    subparser.add_argument(
        "--val-num",
        type=int,
        help="Number of validation chunks to select.",
    )
    subparser.add_argument(
        "--unstratified",
        action="store_true",
        help="""For --val-prop split, perform unstratified splitting. Default
        will perform split stratified over labels.""",
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
        metavar=("PATH", "NUM_CHUNKS"),
        action="append",
        help="""1) Remora training dataset path and 2) max number of chunks
        to extract from this dataset. Second argument can be "all" to use all
        chunks from a dataset""",
    )
    subparser.add_argument(
        "--output-dataset",
        required=True,
        help="Output path for dataset",
    )
    subparser.add_argument(
        "--balance",
        action="store_true",
        help="Automatically balance classes after merging",
    )
    subparser.add_argument(
        "--log-filename",
        help="Log filename. (default: Don't output log file)",
    )
    subparser.set_defaults(func=run_dataset_merge)


def run_dataset_merge(args):
    from remora.data_chunks import merge_datasets

    if args.log_filename is not None:
        log.init_logger(args.log_filename)

    input_datasets = [
        (ds_path, None if num_chunks == "all" else int(num_chunks))
        for ds_path, num_chunks in args.input_dataset
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
        help="Proportion of the dataset to be used as validation.",
    )
    data_grp.add_argument(
        "--batch-size",
        default=constants.DEFAULT_BATCH_SIZE,
        type=int,
        help="Number of samples per batch.",
    )
    data_grp.add_argument(
        "--chunk-context",
        type=int,
        nargs=2,
        metavar=("NUM_BEFORE", "NUM_AFTER"),
        help="""Override chunk context from data prep. Number of context signal
        points to select around the central position.""",
    )
    data_grp.add_argument(
        "--kmer-context-bases",
        nargs=2,
        type=int,
        metavar=("BASES_BEFORE", "BASES_AFTER"),
        help="""Override kmer context bases from data prep. Definition of
        k-mer (derived from the reference) passed into the model along with
        each signal position.""",
    )
    data_grp.add_argument(
        "--ext-val",
        nargs="+",
        help="Path(s) to the external validation Remora datasets.",
    )
    data_grp.add_argument(
        "--ext-val-names",
        nargs="+",
        help="""Names for external datasets. If provided must match length of
        [--ext-val] argument""",
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
        help="Path to the output files.",
    )
    out_grp.add_argument(
        "--save-freq",
        default=10,
        type=int,
        help="After how many epochs to save the model.",
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
        help="Model layer size.",
    )

    train_grp = subparser.add_argument_group("Training Arguments")
    train_grp.add_argument(
        "--epochs",
        default=constants.DEFAULT_EPOCHS,
        type=int,
        help="Number of training epochs.",
    )
    train_grp.add_argument(
        "--optimizer",
        default=constants.OPTIMIZERS[0],
        choices=constants.OPTIMIZERS,
        help="Optimizer setting.",
    )
    train_grp.add_argument(
        "--scheduler",
        default=None,
        help="Scheduler setting.",
    )
    train_grp.add_argument(
        "--lr",
        default=constants.DEFAULT_LR,
        type=float,
        help="Learning rate setting.",
    )
    train_grp.add_argument(
        "--weight-decay",
        default=constants.DEFAULT_WEIGHT_DECAY,
        type=float,
        help="Weight decay setting.",
    )
    train_grp.add_argument(
        "--early-stopping",
        default=5,
        type=int,
        help="""Stops training after a number of epochs without improvement.
        If set to 0 no stopping is done.""",
    )
    train_grp.add_argument(
        "--seed",
        type=int,
        help="Seed value.",
    )
    train_grp.add_argument(
        "--filter-fraction",
        default=constants.DEFAULT_FILT_FRAC,
        type=float,
        help="""Fraction of predictions to filter in validation reporting.
        Un-filtered validation metrics will always be reported as well.""",
    )
    train_grp.add_argument(
        "--lr-sched-kwargs",
        nargs=3,
        action="append",
        metavar=("NAME", "VALUE", "TYPE"),
    )
    train_grp.add_argument(
        "--batch-label-weights",
        type=float,
        nargs="+",
        help="Select batch labels with specified weights",
    )
    train_grp.add_argument(
        "--high-conf-incorrect-thr-frac",
        nargs=2,
        type=float,
        metavar=("THRESHOLD", "FRACTION"),
        help="1.) Threshold value of what to consider a high confidence "
        " predicition. Based on the softmax output. \n"
        "2.) Fraction (of the batch size) of highly confident incorrect "
        "predictions to filter during training. Filtering up to this value, "
        "but might be lower.",
    )
    train_grp.add_argument(
        "--finetune-path",
        help="Path to the torch checkpoint for the model to be fine tuned.",
    )
    train_grp.add_argument(
        "--freeze-num-layers",
        default=0,
        type=int,
        help="Number of layers to be frozen for finetuning.",
    )

    comp_grp = subparser.add_argument_group("Compute Arguments")
    comp_grp.add_argument(
        "--device",
        help="Device for neural network processing. See torch.device.",
    )

    subparser.set_defaults(func=run_model_train)


def run_model_train(args):
    from remora.util import parse_device
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
        parse_device(args.device),
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
        args.ext_val_names,
        args.lr_sched_kwargs,
        args.balance,
        args.batch_label_weights,
        args.high_conf_incorrect_thr_frac,
        args.finetune_path,
        args.freeze_num_layers,
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
        help="Path or directory to save the model.",
    )
    subparser.add_argument(
        "--model-path",
        help="Path to a model architecture. Default: Use path from checkpoint.",
    )
    subparser.add_argument(
        "--format",
        default="dorado",
        choices=["dorado", "torchscript"],
        help="Export format.",
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
        help="Specify the basecaller model type (e.g., fast, hac or sup)",
    )
    subparser.add_argument(
        "--basecall-model-version", help="Specify basecaller model version"
    )
    subparser.add_argument(
        "--modified-bases",
        nargs="+",
        help="Specify the modified base(s)",
    )
    subparser.add_argument(
        "--remora-model-type",
        help="Specify model motif (sequence context)",
    )
    subparser.add_argument(
        "--remora-model-version", help="Specify Remora model version"
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
        help="Specify the basecaller model type (e.g., fast, hac or sup)",
    )
    subparser.add_argument(
        "--basecall-model-version", help="Specify basecaller model version"
    )
    subparser.add_argument(
        "--modified-bases",
        nargs="+",
        help="Specify the modified base(s)",
    )
    subparser.add_argument(
        "--remora-model-type",
        help="Specify model motif (sequence context)",
    )
    subparser.add_argument(
        "--remora-model-version", help="Specify Remora model version"
    )
    subparser.set_defaults(func=run_download)


def run_download(args):
    import pkg_resources

    from remora.download import ModelDownload
    from remora.model_util import get_pretrained_models

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
    out_grp.add_argument("--out-bam", help="Output BAM path.", required=True)
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
        help="""Choose the specific motif of the model you want to load.
        If None, load CG model.""",
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
        help="""Infer per-read modified bases against reference bases instead
        of basecalls.""",
    )

    comp_grp = subparser.add_argument_group("Compute Arguments")
    comp_grp.add_argument(
        "--device",
        type=int,
        help="ID of GPU that is used for inference. Default: CPU only",
    )
    comp_grp.add_argument(
        "--queue-max",
        type=int,
        default=1_000,
        help="Maximum number of reads to store in each multiprocessing queue.",
    )
    comp_grp.add_argument(
        "--num-extract-alignment-workers",
        type=int,
        default=1,
        help="Number of signal extraction workers.",
    )
    comp_grp.add_argument(
        "--num-prepare-batch-workers",
        type=int,
        default=1,
        help="Number of batch preparation workers.",
    )
    comp_grp.add_argument(
        "--batch-size",
        default=constants.DEFAULT_BATCH_SIZE,
        type=int,
        help="Number of input units per batch.",
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
        help=f"""BAM file containing duplex base called sequences (and optional
        reference mappings). Record names may either be the template read_id
        or template<delim>complement. The value of <delim> can be set with
        {duplex_delim_flag}.""",
    )
    subparser.add_argument(
        "duplex_read_pairs",
        help="""Whitespace separated plain text file containing read ID pairs,
        no header.""",
    )
    subparser.add_argument(
        duplex_delim_flag,
        default=";",
        help="""Deliminator string between template and complement read
        ids in the duplex BAM""",
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
        help="""Choose the type of pore the Remora model has been trained on
        (e.g. dna_r10.4_e8.1)""",
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
        help="Number of IO extraction workers.",
    )
    comp_grp.add_argument(
        "--num-duplex-prep-workers",
        type=int,
        default=1,
        help="Number of duplex prep workers (tends to bottleneck).",
    )
    comp_grp.add_argument(
        "--num-infer-workers",
        type=int,
        default=1,
        help="""Number of chunk extraction workers. If performing signal
        refinement this should be increased.""",
    )

    subparser.set_defaults(func=run_infer_from_pod5_and_bam_duplex)


def _unpack_model_kw_args(args) -> dict:
    from remora.util import parse_device

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
        "device": parse_device(args.device),
    }
    return model_kwargs


def run_infer_from_pod5_and_bam(args):
    import torch

    from remora.model_util import load_model
    from remora.inference import infer_from_pod5_and_bam

    if args.log_filename is not None:
        log.init_logger(args.log_filename)
    # test that model can be loaded in parent process
    model_kwargs = _unpack_model_kw_args(args)
    model, model_metadata = load_model(
        **model_kwargs, quiet=False, eval_only=True
    )
    torch.set_grad_enabled(False)
    infer_from_pod5_and_bam(
        pod5_path=args.pod5,
        in_bam_path=args.in_bam,
        model=model,
        model_metadata=model_metadata,
        out_bam_path=args.out_bam,
        num_reads=args.num_reads,
        queue_max=args.queue_max,
        num_extract_alignment_workers=args.num_extract_alignment_workers,
        num_prep_batch_workers=args.num_prepare_batch_workers,
        batch_size=args.batch_size,
        ref_anchored=args.reference_anchored,
    )


def run_infer_from_pod5_and_bam_duplex(args):
    import torch

    from remora.model_util import load_model
    from remora.inference import infer_duplex

    if args.log_filename is not None:
        log.init_logger(args.log_filename)
    model_kwargs = _unpack_model_kw_args(args)
    model, model_metadata = load_model(
        **model_kwargs, quiet=False, eval_only=True
    )
    torch.set_grad_enabled(False)

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
        help="""Name of this sample/comparison. Useful when tabulating
        several runs.""",
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
        help="Maxiumum number of sites to extract from a single read.",
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
        help="Number of input units per batch.",
    )
    comp_grp.add_argument(
        "--device",
        type=int,
        help="ID of GPU that is used for inference. Default: CPU",
    )

    subparser.set_defaults(func=run_validate_from_remora_dataset)


def run_validate_from_remora_dataset(args):
    import torch

    from remora.util import parse_device
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
        device=parse_device(args.device),
        eval_only=True,
    )
    torch.set_grad_enabled(False)

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
    )


##################
# remora analyze #
##################


def register_analyze(parser):
    subparser = parser.add_parser(
        "analyze",
        description="Analyze nanopore data including raw signal",
        help="Analyze nanopore data including raw signal",
        formatter_class=SubcommandHelpFormatter,
    )
    ssubparser = subparser.add_subparsers(title="Analyze commands")
    # register_estimate_kmer_levels(ssubparser)
    # Since `plot` has several sub-commands, print help as default
    subparser.set_defaults(func=lambda x: subparser.print_help())
    # Register analyze sub commands
    register_analyze_plot(ssubparser)


def register_analyze_plot(parser):
    subparser = parser.add_parser(
        "plot",
        description="Plot nanopore data including raw signal",
        help="Plot nanopore data including raw signal",
        formatter_class=SubcommandHelpFormatter,
    )
    ssubparser = subparser.add_subparsers(title="Plot commands")
    # Since `plot` has several sub-commands, print help as default
    subparser.set_defaults(func=lambda x: subparser.print_help())
    # Register plot sub commands
    register_plot_ref_region(ssubparser)


def register_plot_ref_region(parser):
    subparser = parser.add_parser(
        "ref_region",
        description="Plot signal at reference region",
        help="Plot signal at reference region",
        formatter_class=SubcommandHelpFormatter,
    )

    in_grp = subparser.add_argument_group("Input Arguments")
    in_grp.add_argument(
        "--pod5-and-bam",
        required=True,
        nargs=2,
        metavar=("POD5", "BAM"),
        action="append",
        help="""POD5 signal path and BAM file path. BAM file must be mapped,
        sorted and indexed and contain move table and MD tags. Multiple
        samples can be supplied and will be plotted in different colors""",
    )
    in_grp.add_argument(
        "--ref-regions",
        required=True,
        help="""Reference region(s) to plot specified in BED format. Each line
        in this input file will produce one page in the output PDF.""",
    )
    in_grp.add_argument(
        "--highlight-ranges",
        help="""BED file containing regions to highlight""",
    )
    in_grp.add_argument(
        "--highlight-color",
        default="orange",
        help="""Color or highlighted regions""",
    )

    refine_grp = subparser.add_argument_group("Signal Mapping Refine Arguments")
    refine_grp.add_argument(
        "--refine-kmer-level-table",
        help="""Tab-delimited file containing no header and two fields:
        1. string k-mer sequence and 2. float expected normalized level.
        All k-mers must be the same length and all combinations of the bases
        'ACGT' must be present in the file.""",
    )
    refine_grp.add_argument(
        "--refine-rough-rescale",
        action="store_true",
        help="""Apply a rough rescaling using quantiles of signal+move table
        and levels.""",
    )
    refine_grp.add_argument(
        "--refine-scale-iters",
        default=0,
        type=int,
        help="""Number of iterations of signal mapping refinement and signal
        re-scaling to perform. Set to 0 (default) in order to perform signal
        mapping refinement, but skip re-scaling. Set to -1 to skip signal
        mapping (potentially using levels for rough rescaling).""",
    )
    refine_grp.add_argument(
        "--refine-half-bandwidth",
        default=constants.DEFAULT_REFINE_HBW,
        type=int,
        help="""Half bandwidth around signal mapping over which to search for
        new path.""",
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
        help="""Short dwell penalty refiner parameters. Dwells shorter than
        LIMIT will be penalized a value of `WEIGHT * (dwell - TARGET)^2`.""",
    )

    plt_grp = subparser.add_argument_group("Plot Arguments")
    plt_grp.add_argument(
        "--figsize",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=constants.DEFAULT_PLOT_FIG_SIZE,
        help="Figure size",
    )
    plt_grp.add_argument(
        "--ylim",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        help="Signal plotting limits",
    )

    out_grp = subparser.add_argument_group("Output Arguments")
    out_grp.add_argument(
        "--plots-filename",
        default="remora_raw_signal_plot.pdf",
        help="Output plots PDF file location.",
    )
    out_grp.add_argument(
        "--log-filename",
        help="Log filename. Default: Don't output log file.",
    )

    subparser.set_defaults(func=run_plot_ref_region)


def run_plot_ref_region(args):
    import pod5
    import pysam
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from remora import log, io, refine_signal_map

    if args.log_filename is not None:
        log.init_logger(args.log_filename)
    pod5_paths, bc_paths = zip(*args.pod5_and_bam)
    bam_fhs = [pysam.AlignmentFile(bc_path) for bc_path in bc_paths]
    pod5_fhs = [pod5.Reader(pod5_path) for pod5_path in pod5_paths]
    sig_map_refiner = refine_signal_map.SigMapRefiner(
        kmer_model_filename=args.refine_kmer_level_table,
        do_rough_rescale=args.refine_rough_rescale,
        scale_iters=args.refine_scale_iters,
        algo=args.refine_algo,
        half_bandwidth=args.refine_half_bandwidth,
        sd_params=args.refine_short_dwell_parameters,
        do_fix_guage=True,
    )
    highlight_ranges = None
    if args.highlight_ranges is not None:
        highlight_ranges = io.parse_bed(args.highlight_ranges)

    with PdfPages(args.plots_filename) as pdf_fh:
        for ref_reg in io.parse_bed_lines(args.ref_regions):
            reg_highlight_ranges = None
            if highlight_ranges is not None:
                try:
                    reg_highlight_ranges = [
                        (pos, pos + 1, args.highlight_color)
                        for pos in highlight_ranges[
                            (ref_reg.ctg, ref_reg.strand)
                        ]
                        if ref_reg.start <= pos < ref_reg.end
                    ]
                except KeyError:
                    LOGGER.debug(f"No highlight values for region {ref_reg}")
                    pass
            fig, ax = plt.subplots(figsize=args.figsize)
            io.plot_signal_at_ref_region(
                ref_reg,
                zip(pod5_fhs, bam_fhs),
                sig_map_refiner,
                fig_ax=(fig, ax),
                ylim=args.ylim,
                highlight_ranges=reg_highlight_ranges,
            )
            pdf_fh.savefig(fig, bbox_inches="tight")


def register_estimate_kmer_levels(parser):
    subparser = parser.add_parser(
        "estimate_kmer_levels",
        description="Estimate k-mer level table",
        help="Estimate k-mer level table",
        formatter_class=SubcommandHelpFormatter,
    )

    in_grp = subparser.add_argument_group("Input Arguments")
    in_grp.add_argument(
        "--pod5-and-bam",
        required=True,
        nargs=2,
        metavar=("POD5", "BAM"),
        action="append",
        help="""POD5 signal path and BAM file path. BAM file must be mapped,
        sorted and indexed and contain move table and MD tags. Multiple
        samples can be supplied and will be aggregated after site level
        extraction""",
    )

    refine_grp = subparser.add_argument_group("Signal Mapping Refine Arguments")
    refine_grp.add_argument(
        "--refine-kmer-level-table",
        help="""Tab-delimited file containing no header and two fields:
        1. string k-mer sequence and 2. float expected normalized level.
        All k-mers must be the same length and all combinations of the bases
        'ACGT' must be present in the file.""",
    )
    refine_grp.add_argument(
        "--refine-rough-rescale",
        action="store_true",
        help="""Apply a rough rescaling using quantiles of signal+move table
        and levels.""",
    )
    refine_grp.add_argument(
        "--refine-scale-iters",
        default=0,
        type=int,
        help="""Number of iterations of signal mapping refinement and signal
        re-scaling to perform. Set to 0 (default) in order to perform signal
        mapping refinement (aka resquiggle), but skip fine re-scaling. Set to
        -1 to skip signal mapping (potentially using levels for rough
        rescaling).""",
    )
    refine_grp.add_argument(
        "--refine-half-bandwidth",
        default=constants.DEFAULT_REFINE_HBW,
        type=int,
        help="""Half bandwidth around signal mapping over which to search for
        new path.""",
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
        help="""Short dwell penalty refiner parameters. Dwells shorter than
        LIMIT will be penalized a value of `WEIGHT * (dwell - TARGET)^2`.""",
    )

    data_grp = subparser.add_argument_group("Data Arguments")
    data_grp.add_argument(
        "--min-coverage",
        type=int,
        default=10,
        help="Miniumum coverage to include a site.",
    )
    data_grp.add_argument(
        "--kmer-context-bases",
        nargs=2,
        default=(2, 2),
        type=int,
        metavar=("BASES_BEFORE", "BASES_AFTER"),
        help="""Definition of k-mer by the number of bases before and after the
        assigned signal position""",
    )

    out_grp = subparser.add_argument_group("Output Arguments")
    out_grp.add_argument(
        "--levels-filename",
        default="remora_kmer_levels.txt",
        help="Output file for kmer levels.",
    )
    out_grp.add_argument(
        "--log-filename",
        help="Log filename. Default: Don't output log file.",
    )

    comp_grp = subparser.add_argument_group("Compute Arguments")
    comp_grp.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers.",
    )
    comp_grp.add_argument(
        "--chunk-width",
        type=int,
        default=1_000,
        help="""Width of reference region to process at one time. Should be
        smaller for very high coverage.""",
    )
    comp_grp.add_argument(
        "--max-chunk-coverage",
        type=int,
        default=100,
        help="Maxiumum mean chunk coverage for each region.",
    )

    subparser.set_defaults(func=run_estimate_kmer_levels)


def run_estimate_kmer_levels(args):
    from itertools import product

    import pysam
    import numpy as np

    from remora import log, io, refine_signal_map

    if args.log_filename is not None:
        log.init_logger(args.log_filename)
    # open first to avoid long process without write access
    out_fh = open(args.levels_filename, "w")

    sig_map_refiner = refine_signal_map.SigMapRefiner(
        kmer_model_filename=args.refine_kmer_level_table,
        do_rough_rescale=args.refine_rough_rescale,
        scale_iters=args.refine_scale_iters,
        algo=args.refine_algo,
        half_bandwidth=args.refine_half_bandwidth,
        sd_params=args.refine_short_dwell_parameters,
        do_fix_guage=True,
    )
    if not sig_map_refiner.is_loaded or sig_map_refiner.scale_iters < 0:
        LOGGER.warning(
            "It is highly recommended to apply signal mapping refinement in "
            "order to output a valid kmer level table."
        )

    kmer_len = sum(args.kmer_context_bases) + 1
    all_kmer_levels = dict(
        ("".join(bs), []) for bs in product("ACGT", repeat=kmer_len)
    )
    for pod5_path, bam_path in args.pod5_and_bam:
        try:
            with pysam.AlignmentFile(bam_path) as bam_fh:
                _ = bam_fh.fetch(bam_fh.header.references[0], 0, 1)
        except ValueError:
            LOGGER.warning(
                "Cannot estimate levels from BAM file without mappings or index"
            )
            continue
        LOGGER.info(f"Extracting levels from {pod5_path} and {bam_path}")
        for kmer, levels in io.get_site_kmer_levels(
            pod5_path,
            bam_path,
            sig_map_refiner,
            args.kmer_context_bases,
            min_cov=args.min_coverage,
            chunk_len=args.chunk_width,
            max_chunk_cov=args.max_chunk_coverage,
            num_workers=args.num_workers,
        ).items():
            all_kmer_levels[kmer].append(levels)
    LOGGER.info("Aggregating and outputting levels")
    been_warned = False
    for kmer, levels in sorted(all_kmer_levels.items()):
        levels = np.concatenate(levels)
        if levels.size == 0:
            if not been_warned:
                LOGGER.warning("Some k-mers not observed.")
                been_warned = True
            out_fh.write(f"{kmer}\tnan\n")
        else:
            out_fh.write(f"{kmer}\t{np.median(levels)}\n")
    out_fh.close()
