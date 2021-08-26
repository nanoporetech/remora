import numpy as np
from tqdm import tqdm

from taiyaki.mapped_signal_files import MappedSignalReader, MappedSignalWriter


MOD_BASE = "a"
CAN_MOTIF = "GATC"
MOTIF_OFFSET = 1
CONTEXT_BASES = 20


# input_msf = MappedSignalReader("merged_map_sig.hdf5")
def extract_motif_dataset(mod_base, can_motif, motif_offset, context_bases):

    mod_motif = (
        can_motif[:motif_offset] + mod_base + can_motif[motif_offset + 1 :]
    )

    input_msf = MappedSignalReader(
        "/media/groups/res_algo/active/mstoiber/mods_wo_basecalling/merged_map_sig.hdf5"
    )
    alphabet_info = input_msf.get_alphabet_information()
    output_msf = MappedSignalWriter("toy_training_data.hdf5", alphabet_info)

    int_mod_base = alphabet_info.alphabet.find(mod_base)
    int_can_motif = np.array(
        [alphabet_info.alphabet.find(b) for b in can_motif]
    )
    int_mod_motif = np.array(
        [alphabet_info.alphabet.find(b) for b in mod_motif]
    )

    for read in tqdm(input_msf, smoothing=0):
        focus_pos = None
        int_motif = (
            int_mod_motif
            if (read.Reference == int_mod_base).sum() > 0
            else int_can_motif
        )

        for read in tqdm(input_msf, smoothing=0):
            focus_pos = None
            int_motif = (
                int_mod_motif
                if (read.Reference == int_mod_base).sum() > 0
                else int_can_motif
            )
            # extract region around first modified base
            for offset in range(
                context_bases,
                read.Reference.shape[0] - len(can_motif) - context_bases,
            ):
                if np.array_equal(
                    read.Reference[offset : offset + len(can_motif)], int_motif
                ):
                    focus_pos = offset + motif_offset
                    break
            if focus_pos is None:
                continue
            read_dict = read.get_read_dictionary()
            read_dict["Ref_to_signal"] = read.Ref_to_signal[
                focus_pos - context_bases : focus_pos + context_bases + 1
            ]
            read_dict["Reference"] = read.Reference[
                focus_pos - context_bases : focus_pos + context_bases + 1
            ]
            output_msf.write_read(read_dict)

        input_msf.close()
        output_msf.close()
