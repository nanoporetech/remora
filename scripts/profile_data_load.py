import argparse

from taiyaki.mapped_signal_files import MappedSignalReader
import torch
from tqdm import tqdm

from remora import log, constants, encoded_kmers
from remora.data_chunks import RemoraDataset
from remora.prepare_train_data import fill_dataset
from remora.util import Motif, get_can_converter

LOGGER = log.get_logger()


def iter_batches(dataset, num_profile_batches):
    bb, ab = dataset.kmer_context_bases
    for epoch_i in tqdm(range(num_profile_batches), smoothing=0, unit="epoch"):
        for batch_i, (
            (sigs, seqs, seq_maps, seq_lens),
            labels,
            read_data,
        ) in enumerate(dataset):
            enc_kmers = torch.from_numpy(
                encoded_kmers.compute_encoded_kmer_batch(
                    bb, ab, seqs, seq_maps, seq_lens
                )
            )
            LOGGER.debug(f"{epoch_i} {batch_i} {enc_kmers.shape}")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Profile training data extraction",
    )
    data_grp = parser.add_argument_group("Data Arguments")
    data_grp.add_argument(
        "taiyaki_mapped_signal",
        help="Taiyaki mapped signal file",
    )
    data_grp.add_argument(
        "--num-chunks",
        type=int,
        help="Total number of chunks loaded for training. "
        "Default: All chunks loaded",
    )
    data_grp.add_argument(
        "--chunk-context",
        default=constants.DEFAULT_CHUNK_CONTEXT,
        type=int,
        nargs=2,
        help="Number of context signal points or bases to select around the "
        "central position. Default: %(default)s",
    )
    data_grp.add_argument(
        "--batch-size",
        default=constants.DEFAULT_BATCH_SIZE,
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
        default=constants.DEFAULT_KMER_CONTEXT_BASES,
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
    data_grp.add_argument(
        "--max-chunks-per-read",
        type=int,
        default=10,
        help="Maxiumum number of chunks to extract from a single read. "
        "Default: %(default)s",
    )
    data_grp.add_argument(
        "--max-seq-length",
        type=int,
        default=constants.DEFAULT_MAX_SEQ_LEN,
        help="Maxiumum bases from a chunk Should be adjusted accordingly with "
        "--chunk-context. Default: %(default)s",
    )

    mdl_grp = parser.add_argument_group("Model Arguments")
    mdl_grp.add_argument(
        "--mod-bases",
        help="Single letter codes for modified bases to predict. Must "
        "provide either this or specify --base-pred.",
    )
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

    input_msf = MappedSignalReader(args.taiyaki_mapped_signal)
    alphabet_info = input_msf.get_alphabet_information()
    alphabet, collapse_alphabet = (
        alphabet_info.alphabet,
        alphabet_info.collapse_alphabet,
    )
    can_conv = get_can_converter(
        alphabet_info.alphabet, alphabet_info.collapse_alphabet
    )
    label_conv = can_conv
    motif = Motif(*args.motif)
    num_reads = len(input_msf.get_read_ids())
    # initialize empty dataset with pre-allocated memory
    dataset = RemoraDataset.allocate_empty_chunks(
        num_chunks=args.max_chunks_per_read * num_reads,
        chunk_context=args.chunk_context,
        max_seq_len=args.max_seq_length,
        kmer_context_bases=args.kmer_context_bases,
        base_pred=args.base_pred,
        mod_bases=args.mod_bases,
        motif=motif.to_tuple(),
    )
    LOGGER.info("Profiling Remora dataset conversion")
    save_fn = f"{args.output_basename}_remora_mapped_signal.npz"
    # appease flake8
    fill_dataset
    cProfile.runctx(
        """fill_dataset(
            input_msf,
            dataset,
            num_reads,
            can_conv,
            can_conv,
            args.max_chunks_per_read,
        )
dataset.shuffle()
dataset.save(save_fn)
        """,
        globals(),
        locals(),
        filename=f"{args.output_basename}_remora_convert.prof",
    )

    LOGGER.info("Profiling data iteration")
    cProfile.runctx(
        "iter_batches(dataset, args.num_profile_batches)",
        globals(),
        locals(),
        filename=f"{args.output_basename}_iter_batches.prof",
    )


if __name__ == "__main__":
    main(get_parser().parse_args())
