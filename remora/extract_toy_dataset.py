import numpy as np
from tqdm import tqdm

from taiyaki.mapped_signal_files import MappedSignalReader, MappedSignalWriter


MOD_BASE = "a"
CAN_MOTIF = "GATC"
MOTIF_OFFSET = 1
MOD_MOTIF = CAN_MOTIF[:MOTIF_OFFSET] + MOD_BASE + CAN_MOTIF[MOTIF_OFFSET + 1 :]
CONTEXT_BASES = 20


input_msf = MappedSignalReader("merged_map_sig.hdf5")
alphabet_info = input_msf.get_alphabet_information()
output_msf = MappedSignalWriter("toy_training_data.hdf5", alphabet_info)

int_mod_base = alphabet_info.alphabet.find(MOD_BASE)
int_can_motif = np.array([alphabet_info.alphabet.find(b) for b in CAN_MOTIF])
int_mod_motif = np.array([alphabet_info.alphabet.find(b) for b in MOD_MOTIF])

for read in tqdm(input_msf, smoothing=0):
    focus_pos = None
    int_motif = (
        int_mod_motif
        if sum(read.Reference == int_mod_base) > 0
        else int_can_motif
    )
    # extract region around first modified base
    for offset in range(
        CONTEXT_BASES, read.Reference.shape[0] - len(CAN_MOTIF) - CONTEXT_BASES
    ):
        if np.array_equal(
            read.Reference[offset : offset + len(CAN_MOTIF)], int_motif
        ):
            focus_pos = offset + MOTIF_OFFSET
            break
    if focus_pos is None:
        continue
    read_dict = read.get_read_dictionary()
    read_dict["Ref_to_signal"] = read.Ref_to_signal[
        focus_pos - CONTEXT_BASES : focus_pos + CONTEXT_BASES + 1
    ]
    read_dict["Reference"] = read.Reference[
        focus_pos - CONTEXT_BASES : focus_pos + CONTEXT_BASES + 1
    ]
    output_msf.write_read(read_dict)

input_msf.close()
output_msf.close()
