from taiyaki.mapped_signal_files import MappedSignalReader

import pdb


def sample_chunks_bybase(
    read_data_path,
    number_to_sample,
    bases_below,
    bases_above,
    mod_offset,
    mod,
    standardise=True,
    select_strands_randomly=True,
    first_strand_index=0,
):

    """Extract the signal corresponding to the modbase and any number of surrounding bases
    before or after the mod.

    Args:
        read_data: list of signal_mapping.SignalMapping objects
        number_to_sample: target number of data elements to return
        bases_below: number of bases to include from before the central base
        bases_above: number of bases to include from after the central base
        mod_offset: position of the modbase
        standardise: returns standardised signals, otherwise unscaled
        select_strands_randomly: chooses a random read at each iteration
        first_strand_index: When select_strands_randomly==False, start reading
            from this position
    """
    read_data = MappedSignalReader(read_data_path)

    n_reads = len(read_data.get_read_ids())

    alphabet_info = read_data.get_alphabet_information()

    if (
        number_to_sample is None
        or number_to_sample == 0
        or number_to_sample > n_reads
    ):
        number_to_sample = n_reads

    if bases_below is None:
        bases_below = 0

    if bases_above is None:
        bases_above = 0

    if not isinstance(number_to_sample, int):
        raise ValueError("number_to_sample must be an integer")

    if not isinstance(bases_below, int):
        raise ValueError("bases_below must be an integer")

    if not isinstance(bases_above, int):
        raise ValueError("bases_above must be an integer")

    if not isinstance(first_strand_index, int):
        raise ValueError("first_strand_index must be an integer")

    extracted_signal = []
    labels = []
    refs = []
    base_locations = []
    read_ids = []
    positions = []
    count = 0

    for read in read_data:

        sig = read.get_current(read.get_mapped_dacs_region())
        ref = "".join(
            alphabet_info.collapse_alphabet[b] for b in read.Reference
        )
        base_locs = read.Ref_to_signal - read.Ref_to_signal[0]
        is_mod = read.Reference[mod_offset] == alphabet_info.alphabet.find(mod)

        signalPoint1 = mod_offset - bases_below
        signalPoint2 = mod_offset + bases_above
        if signalPoint1 <= 0:
            signalPoint1 = 0
            print("cannot select from negative bases, starting from first base")
        if signalPoint2 >= len(base_locs):
            signalPoint2 = len(base_locs)
            print(
                "cannot select from bases after end of sequence, using last base"
            )

        extracted_signal.append(
            sig[base_locs[signalPoint1] : base_locs[signalPoint2]]
        )
        labels.append(is_mod)
        refs.append(ref)
        base_locations.append(base_locs)
        read_ids.append(read.read_id)
        positions.append(read.Ref_to_signal[mod_offset])

        count += 1
        if count >= number_to_sample:
            break

    return extracted_signal, labels, refs, base_locations, read_ids, positions


def sample_chunks_bychunksize(
    read_data_path,
    number_to_sample,
    chunk_size_below,
    chunk_size_above,
    mod_offset,
    mod,
    standardise=True,
    select_strands_randomly=True,
    first_strand_index=0,
):

    """Extract the signal corresponding to the modbase and any number of surrounding bases
    before or after the mod.

    Args:
        read_data: list of signal_mapping.SignalMapping objects
        number_to_sample: target number of data elements to return
        chunk_size_below: number of signal points to include from before the central base
        chunk_size_above: number of signal points to include from after the central base
        mod_offset: position of the modbase
        standardise: returns standardised signals, otherwise unscaled
        select_strands_randomly: chooses a random read at each iteration
        first_strand_index: When select_strands_randomly==False, start reading
            from this position
    """

    read_data = MappedSignalReader(read_data_path)

    n_reads = len(read_data.get_read_ids())

    alphabet_info = read_data.get_alphabet_information()

    if (
        number_to_sample is None
        or number_to_sample == 0
        or number_to_sample > n_reads
    ):
        number_to_sample = n_reads

    if chunk_size_below is None:
        chunk_size_below = 0

    if chunk_size_above is None:
        chunk_size_above = 0

    if not isinstance(number_to_sample, int):
        raise ValueError("number_to_sample must be an integer")

    if not isinstance(chunk_size_below, int):
        raise ValueError("chunk_size_below must be an integer")

    if not isinstance(chunk_size_above, int):
        raise ValueError("chunk_size_above must be an integer")

    if not isinstance(first_strand_index, int):
        raise ValueError("first_strand_index must be an integer")

    extracted_signal = []
    labels = []
    refs = []
    base_locations = []
    read_ids = []
    positions = []
    count = 0

    for read in read_data:

        sig = read.get_current(read.get_mapped_dacs_region())
        ref = "".join(
            alphabet_info.collapse_alphabet[b] for b in read.Reference
        )
        base_locs = read.Ref_to_signal - read.Ref_to_signal[0]
        is_mod = read.Reference[mod_offset] == alphabet_info.alphabet.find(mod)

        signalPoint1 = base_locs[mod_offset] - chunk_size_below
        signalPoint2 = base_locs[mod_offset] + chunk_size_above
        if signalPoint1 < 0:
            print("cannot select from negative bases, skipping")
            continue
        if signalPoint2 > base_locs[-1]:
            print("cannot select from bases after end of sequence, skipping")
            continue

        extracted_signal.append(sig[signalPoint1:signalPoint2])
        labels.append(is_mod)
        refs.append(ref)
        base_locations.append(base_locs)
        read_ids.append(read.read_id)
        positions.append(read.Ref_to_signal[mod_offset])

        count += 1
        if count >= number_to_sample:
            break

    return extracted_signal, labels, refs, base_locations, read_ids, positions
