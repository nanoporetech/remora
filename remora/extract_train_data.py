from taiyaki.mapped_signal_files import MappedSignalReader

MOD_OFFSET = 20


mod_training_msf = MappedSignalReader("toy_training_data.hdf5")
alphabet_info = mod_training_msf.get_alphabet_information()

for read in mod_training_msf:
    sig = read.get_current(read.get_mapped_dacs_region())
    ref = "".join(alphabet_info.collapse_alphabet[b] for b in read.Reference)
    base_locs = read.Ref_to_signal - read.Ref_to_signal[0]
    is_mod = read.Reference[MOD_OFFSET] == 1
    # TODO: write methods and robust API to train prediction model (for is_mod)
    # from sig and ref.
    # ref is fixed length (MOD_OFFSET * 2 + 1)
    # signal in this case is assigned by megalodon (so essentially the coarse
    # mapping from tombo2)
    # The exact mapping from reference bases to signal is found in base_locs.
    # base_locs need not be used in prediction, but may be used if desired.
