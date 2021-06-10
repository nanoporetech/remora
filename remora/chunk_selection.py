
def sample_chunks(read_data, number_to_sample, num_surrounding_bases,MOD_OFFSET,
                    standardise=True, select_strands_randomly=True,
                    first_strand_index=0):

    """
    Args:
        read_data: list of signal_mapping.SignalMapping objects
        number_to_sample: target number of data elements to return
        num_surrounding_bases: include signal from bases around the central base;
            this is symmetrical so a value of n will select n bases before and
            after the potentially modified base
        standardise: returns standardised signals, otherwise unscaled
        select_strands_randomly: chooses a random read at each iteration
        first_strand_index: When select_strands_randomly==False, start reading
            from this position
    """
    n_reads = len(read_data.get_read_ids())

    alphabet_info = read_data.get_alphabet_information()

    if number_to_sample is None or number_to_sample == 0 or number_to_sample > n_reads:
        number_to_sample = n_reads

    if num_surrounding_bases is None:
        num_surrounding_bases = 0


    if not isinstance(number_to_sample, int):
        raise ValueError('number_to_sample must be an integer')

    if not isinstance(num_surrounding_bases, int):
        raise ValueError('num_surrounding_bases must be an integer')

    if not isinstance(first_strand_index, int):
        raise ValueError('first_strand_index must be an integer')


    extracted_signal = []
    count = 0

    for read in read_data:


        sig = read.get_current(read.get_mapped_dacs_region())
        ref = ''.join(alphabet_info.collapse_alphabet[b] for b in read.Reference)
        base_locs = read.Ref_to_signal - read.Ref_to_signal[0]

        signalPoint1 = MOD_OFFSET - num_surrounding_bases
        signalPoint2 = MOD_OFFSET + num_surrounding_bases
        if signalPoint1 <= 0:
            signalPoint1 = 0
            print('cannot select from negative bases, starting from first base')
        if signalPoint2 >= len(base_locs):
            signalPoint2 = len(base_locs)
            print('cannot select from bases after end of sequence, using last base')


        extracted_signal.append(sig[base_locs[signalPoint1]:base_locs[signalPoint2]])

        count+=1
        if count >= number_to_sample:
            break

    return extracted_signal
