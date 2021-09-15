import argparse
import atexit

import numpy as np
from tqdm import tqdm
from taiyaki.mapped_signal_files import MappedSignalReader, MappedSignalWriter


def extract_canonical_dataset(input_msf, output_msf, context_bases):
    for read in tqdm(input_msf, smoothing=0):
        if read.Reference.size <= context_bases * 2:
            continue
        center_loc = np.random.randint(
            context_bases, read.Reference.size - context_bases, 1
        )[0]
        read_dict = read.get_read_dictionary()
        # trim signal and correct Ref_to_signal mapping
        ref_st = center_loc - context_bases
        ref_en = center_loc + context_bases + 1
        sig_st = read_dict["Ref_to_signal"][ref_st]
        sig_en = read_dict["Ref_to_signal"][ref_en]
        read_dict["Dacs"] = read_dict["Dacs"][sig_st:sig_en]
        read_dict["Ref_to_signal"] = read.Ref_to_signal[ref_st:ref_en] - sig_st
        read_dict["Reference"] = read.Reference[
            center_loc - context_bases : center_loc + context_bases + 1
        ]
        output_msf.write_read(read_dict)


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

    return parser


def main(args):
    input_msf = MappedSignalReader(args.mapped_signal_file)
    atexit.register(input_msf.close)
    output_msf = MappedSignalWriter(
        args.output_mapped_signal_file, input_msf.get_alphabet_information()
    )
    atexit.register(output_msf.close)
    extract_canonical_dataset(
        input_msf,
        output_msf,
        args.context_bases,
    )


if __name__ == "__main__":
    main(get_parser().parse_args())
