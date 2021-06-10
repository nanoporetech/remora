
def sample_chunks(read_data, number_to_sample, bases_below, bases_above,MOD_OFFSET,
                    standardise=True, select_strands_randomly=True,
                    first_strand_index=0):

    """
    Args:
        read_data: list of signal_mapping.SignalMapping objects
        number_to_sample: target number of data elements to return
        bases_below: number of bases to include from before the central base
        bases_above: number of bases to include from after the central base
        standardise: returns standardised signals, otherwise unscaled
        select_strands_randomly: chooses a random read at each iteration
        first_strand_index: When select_strands_randomly==False, start reading
            from this position
    """
    n_reads = len(read_data.get_read_ids())

    alphabet_info = read_data.get_alphabet_information()

    if number_to_sample is None or number_to_sample == 0 or number_to_sample > n_reads:
        number_to_sample = n_reads

    if bases_below is None:
        bases_below = 0

    if bases_above is None:
        bases_above = 0


    if not isinstance(number_to_sample, int):
        raise ValueError('number_to_sample must be an integer')

    if not isinstance(bases_below, int):
        raise ValueError('bases_below must be an integer')

    if not isinstance(bases_above, int):
        raise ValueError('bases_above must be an integer')

    if not isinstance(first_strand_index, int):
        raise ValueError('first_strand_index must be an integer')


    extracted_signal = []
    labels = []
    refs = []
    base_locations = []
    count = 0

    for read in read_data:


        sig = read.get_current(read.get_mapped_dacs_region())
        ref = ''.join(alphabet_info.collapse_alphabet[b] for b in read.Reference)
        base_locs = read.Ref_to_signal - read.Ref_to_signal[0]
        is_mod = read.Reference[MOD_OFFSET] == 1

        signalPoint1 = MOD_OFFSET - bases_below
        signalPoint2 = MOD_OFFSET + bases_above
        if signalPoint1 <= 0:
            signalPoint1 = 0
            print('cannot select from negative bases, starting from first base')
        if signalPoint2 >= len(base_locs):
            signalPoint2 = len(base_locs)
            print('cannot select from bases after end of sequence, using last base')


        extracted_signal.append(sig[base_locs[signalPoint1]:base_locs[signalPoint2]])
        labels.append(is_mod)
        refs.append(ref)
        base_locations.append(base_locs)

        count+=1
        if count >= number_to_sample:
            break

    return extracted_signal, labels, refs, base_locations
