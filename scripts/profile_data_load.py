import argparse

import numpy as np
from taiyaki.mapped_signal_files import MappedSignalReader

from remora import log, constants
from remora.data_chunks import RemoraDataset
from remora.util import Motif

LOGGER = log.get_logger()


def iter_batches(dl_trn, num_profile_batches):
    for epoch_i in range(num_profile_batches):
        for batch_i, (inputs, labels, read_data) in enumerate(dl_trn):
            LOGGER.debug(f"{epoch_i} {batch_i} {inputs[0].shape}")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Profile training data extraction",
    )
    data_grp = parser.add_argument_group("Data Arguments")
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
        "--focus-offset",
        default=constants.DEFAULT_FOCUS_OFFSET,
        type=int,
        help="Offset into stored chunks to be predicted. Default: %(default)d",
    )
    data_grp.add_argument(
        "--chunk-context",
        default=constants.DEFAULT_CONTEXT_CHUNKS,
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
        "--val-prop",
        default=0.01,
        type=float,
        help="Proportion of the dataset to be used as validation. "
        "Default: %(default)f",
    )
    data_grp.add_argument(
        "--kmer-context-bases",
        nargs=2,
        default=constants.DEFAULT_CONTEXT_BASES,
        type=int,
        help="Definition of k-mer (derived from the reference) passed into "
        "the model along with each signal position. Default: %(default)s",
    )
    data_grp.add_argument(
        "--num-profile-batches",
        default=10,
        type=int,
        help="Number of batches to iterate over for profiling. "
        "Default: %(default)d",
    )

    mdl_grp = parser.add_argument_group("Model Arguments")
    mdl_grp.add_argument(
        "--base-pred",
        action="store_true",
        help="Train to predict bases and not mods.",
    )

    out_grp = parser.add_argument_group("Output Arguments")
    out_grp.add_argument(
        "--output-basename",
        default="remora_data_loading",
        help="Path basename to output profiling results. Default: %(default)s",
    )
    out_grp.add_argument(
        "--log-filename",
        help="Log filename. Default: No log file",
    )

    return parser


def main(args):
    log.init_logger(args.log_filename)

    import cProfile

    input_msf = MappedSignalReader(args.dataset_path)
    alphabet_info = input_msf.get_alphabet_information()
    alphabet, collapse_alphabet = (
        alphabet_info.alphabet,
        alphabet_info.collapse_alphabet,
    )
    label_conv = np.arange(len(alphabet))
    motif = Motif(*args.motif)
    num_reads = len(input_msf.get_read_ids())
    # initialize empty dataset with pre-allocated memory
    dataset = RemoraDataset.allocate_empty_chunks(
        num_chunks=args.max_chunks_per_read * num_reads,
        chunk_context=args.chunk_context,
        kmer_context_bases=args.kmer_context_bases,
        base_pred=args.base_pred,
        mod_bases=args.mod_bases,
        motif=motif.to_tuple(),
    )
    LOGGER.info("Profiling Remora dataset extraction")
    cProfile.runctx(
        """
        """,
        globals(),
        locals(),
        filename=f"{args.output_basename}_remora_convert.prof",
    )

    LOGGER.info("Profiling data iteration")
    cProfile.runctx(
        "iter_batches(dl_trn, args.num_profile_batches)",
        globals(),
        locals(),
        filename=f"{args.output_basename}_iter_batches.prof",
    )


if __name__ == "__main__":
    main(get_parser().parse_args())
