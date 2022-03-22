import argparse
import multiprocessing as mp

import torch
from tqdm import tqdm

from remora.util import Motif, queue_iter
from remora.data_chunks import RemoraDataset
from remora.refine_signal_map import SigMapRefiner
from remora import log, constants, encoded_kmers, RemoraError
from remora.prepare_train_data import (
    check_alphabet,
    fill_reads_q,
    extract_chunks_worker,
)

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


def fill_dataset(dataset, chunks_q, save_fn):
    for read_chunks in queue_iter(chunks_q):
        for chunk in read_chunks:
            try:
                dataset.add_chunk(chunk)
            except RemoraError:
                pass
    dataset.clip_chunks()
    dataset.shuffle()
    dataset.save(save_fn)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Profile training data extraction",
    )

    parser.add_argument(
        "input_reads",
        help="Taiyaki mapped signal or RemoraReads pickle file.",
    )

    data_grp = parser.add_argument_group("Data Arguments")
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
        "--min-samples-per-base",
        type=int,
        default=constants.DEFAULT_MIN_SAMPLES_PER_BASE,
        help="Minimum number of samples per base. This sets the size of the "
        "ragged arrays of chunk sequences. Default: %(default)s",
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
        "--max-reads",
        type=int,
        help="Maxiumum number of reads to process.",
    )

    refine_grp = parser.add_argument_group("Signal Mapping Refine Arguments")
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

    mdl_grp = parser.add_argument_group("Model Arguments")
    mdl_grp.add_argument(
        "--mod-base",
        nargs=2,
        action="append",
        metavar=("SINGLE_LETTER_CODE", "MOD_BASE"),
        default=None,
        help="If provided input is RemoraReads pickle, modified bases must "
        "be provided. Exmaple: `--mod-base m 5mC --mod-base h 5hmC`",
    )
    mdl_grp.add_argument(
        "--mod-base-control",
        action="store_true",
        help="Is this a modified bases control sample?",
    )
    mdl_grp.add_argument(
        "--base-pred",
        action="store_true",
        help="Train to predict bases (SNPs) and not mods.",
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

    LOGGER.info("Filling reads queue")
    mod_bases, mod_long_names, num_reads = check_alphabet(
        args.input_reads, args.base_pred, args.mod_base_control, args.mod_base
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
    reads_q = mp.Queue()
    fill_reads_q(reads_q, args.input_reads, max_reads=args.max_reads)

    LOGGER.info("Profiling chunk extraction")
    chunks_q = mp.Queue()
    cProfile.runctx(
        """extract_chunks_worker(
        reads_q,
        chunks_q,
        None,
        sig_map_refiner,
        args.max_chunks_per_read,
        args.chunk_context,
        args.kmer_context_bases,
        args.base_pred,
        args.base_start_justify,
        args.offset,
        )
        """,
        globals(),
        locals(),
        filename=f"{args.output_basename}_remora_extract_chunks.prof",
    )

    LOGGER.info("Profiling dataset prep")
    if args.max_reads is not None:
        num_reads = max(num_reads, args.max_reads)
    # initialize empty dataset with pre-allocated memory
    dataset = RemoraDataset.allocate_empty_chunks(
        num_chunks=args.max_chunks_per_read * num_reads,
        chunk_context=args.chunk_context,
        kmer_context_bases=args.kmer_context_bases,
        min_samps_per_base=args.min_samples_per_base,
        base_pred=args.base_pred,
        mod_bases=mod_bases,
        mod_long_names=mod_long_names,
        motifs=[
            Motif(*args.motif).to_tuple(),
        ],
        sig_map_refiner=sig_map_refiner,
        base_start_justify=args.base_start_justify,
        offset=args.offset,
    )
    save_fn = f"{args.output_basename}_remora_mapped_signal.npz"
    cProfile.runctx(
        "fill_dataset(dataset, chunks_q, save_fn)",
        globals(),
        locals(),
        filename=f"{args.output_basename}_remora_save_dataset.prof",
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
