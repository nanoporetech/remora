import argparse
import atexit

import numpy as np
from tqdm import tqdm
from taiyaki.mapped_signal_files import MappedSignalReader, BatchHDF5Writer


def extract_canonical_dataset(
    input_msf, output_msf, context_bases, max_chunks_per_read
):
    for read in tqdm(input_msf, smoothing=0):
        if read.Reference.size <= context_bases * 2:
            continue
        read_dict = read.get_read_dictionary()
        valid_size = read.Reference.size - (context_bases * 2)
        for center_loc in np.random.choice(
            np.arange(context_bases, read.Reference.size - context_bases),
            size=min(max_chunks_per_read, valid_size),
            replace=False,
        ):
            chunk_dict = read_dict.copy()
            # trim signal and correct Ref_to_signal mapping
            ref_st = center_loc - context_bases
            ref_en = center_loc + context_bases + 1
            sig_st = read.Ref_to_signal[ref_st]
            sig_en = read.Ref_to_signal[ref_en]
            chunk_dict["read_id"] = f"{read.read_id}:::pos{center_loc}"
            chunk_dict["Dacs"] = read.Dacs[sig_st:sig_en]
            chunk_dict["Ref_to_signal"] = (
                read.Ref_to_signal[ref_st:ref_en] - sig_st
            )
            chunk_dict["Reference"] = read.Reference[
                center_loc - context_bases : center_loc + context_bases + 1
            ]
            output_msf.write_read(chunk_dict)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract modified base model training dataset",
    )
    parser.add_argument(
        "mapped_signal_file",
        help="Taiyaki mapped signal file.",
    )
    parser.add_argument(
        "--output-mapped-signal-file",
        default="remora_canonical_base_training_dataset.hdf5",
        help="Output Taiyaki mapped signal file. Default: %(default)s",
    )
    parser.add_argument(
        "--context-bases",
        type=int,
        default=50,
        help="Modified base. Default: %(default)s",
    )
    parser.add_argument(
        "--max-chunks-per-read",
        type=int,
        default=10,
        help="Maxiumum number of chunks to extract from a single read. "
        "Default: %(default)s",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100000,
        help="Number of chunks per batch in output file. "
        "Default: %(default)s",
    )

    return parser


def main(args):
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


if __name__ == "__main__":
    main(get_parser().parse_args())
