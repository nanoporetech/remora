import sys
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from taiyaki.mapped_signal_files import MappedSignalReader, MappedSignalWriter
from taiyaki.alphabet import AlphabetInfo


def get_threshold(mod_scores_path, percentile):
    all_mod_scores = defaultdict(list)
    with open(mod_scores_path) as fp:
        header = fp.readline().split()
        for line in fp:
            fields = dict(zip(header, line.split()))
            all_mod_scores[fields["read_id"]].append(
                (
                    int(fields["pos"]),
                    fields["strand"],
                    np.exp(float(fields["mod_log_prob"])),
                )
            )

    probs = [
        site_data[2]
        for read_data in all_mod_scores.values()
        for site_data in read_data
    ]
    threshold = np.percentile(probs, 100 - percentile)

    return dict(all_mod_scores), threshold


def update_alphabet(alphabet_info, new_base):
    new_can_base, new_mod_base, new_mln = new_base
    if new_mod_base in alphabet_info.mod_bases:
        return alphabet_info, np.arange(alphabet_info.nbase, dtype=np.int16)

    mod_bases = alphabet_info.mod_bases + new_mod_base
    can_equiv_bases = ""
    mod_long_names = []
    for mod_base in alphabet_info.mod_bases:
        can_equiv_bases += mod_base.translate(alphabet_info.translation_table)
        mod_long_names.append(alphabet_info.mod_name_conv[mod_base])
    can_equiv_bases += new_can_base
    mod_long_names.append(new_mln)
    new_alphabet_info = AlphabetInfo(
        alphabet_info.can_bases + mod_bases,
        alphabet_info.can_bases + can_equiv_bases,
        mod_long_names,
        do_reorder=True,
    )

    alphabet_conv = np.zeros(alphabet_info.nbase, dtype=np.int16) - 1
    for base_code, base in enumerate(alphabet_info.alphabet):
        alphabet_conv[base_code] = new_alphabet_info.alphabet.index(base)
    return new_alphabet_info, alphabet_conv


def main(args):
    # Load in 5hmC calls
    sys.stderr.write("reading input signal mappings\n")
    input_msf = MappedSignalReader(args.input_msf)

    # copying merged alphabet to 5hmC mapped signal file
    sys.stderr.write("Constructung new alphabet\n")
    old_alphabet_info = input_msf.get_alphabet_information()
    new_alphabet_info, alphabet_conv = update_alphabet(
        old_alphabet_info, args.new_base
    )

    # Create output file
    sys.stderr.write("creating output file\n")
    output_msf = MappedSignalWriter(args.output_msf, new_alphabet_info)

    # Find the probability at which the threshold is set
    sys.stderr.write("reading in mod base calls\n")
    all_reads_mods, threshold = get_threshold(
        args.input_per_read_mod_scores, args.percentile
    )

    # loading in the 5mC mapping summary text file from the mega/remora run
    map_summs = {}
    with open(args.input_mapping_summary) as fp:
        header = fp.readline().split()
        for line in fp:
            fields = dict(zip(header, line.split()))
            map_summs[fields["read_id"]] = {
                "start": int(fields["start"]),
                "end": int(fields["end"]),
            }

    num_reads = 0
    fail_reasons = defaultdict(int)
    sys.stderr.write("starting markups\n")
    for read in tqdm(input_msf):
        num_reads += 1
        map_summ = map_summs[read.read_id]
        try:
            # If using different megalodon runs mods may not exist
            read_mods = all_reads_mods[read.read_id]
        except KeyError:
            fail_reasons["missing read"] += 1
            continue

        poss, strands, mod_probs = zip(*read_mods)
        strand = strands[0]
        if not all(strand == pos_strand for pos_strand in strands):
            fail_reasons["Diff strands found {sorted(set(strands))}"] += 1
            continue
        if strand not in "+-":
            fail_reasons[f"Bad strand {strand}"] += 1
            continue

        # find positions to mark up with new mod
        markup_pos = np.array(poss)[np.array(mod_probs) > threshold]

        # update the reference encoding to new alphabet
        read.Reference = alphabet_conv[read.Reference]
        if strand == "+":
            markup_pos -= map_summ["start"]
        else:
            markup_pos = map_summ["end"] - markup_pos - 1

        if markup_pos.size == 0:
            output_msf.write_read(read.get_read_dictionary())
            continue

        if markup_pos.max() > read.Reference.size:
            fail_reasons[f"Past read end strand:{strand}"] += 1
            continue
        if markup_pos.min() < 0:
            fail_reasons[f"Before read start strand:{strand}"] += 1
            continue

        # check that a old_base is called here. If not we skip the read.
        if args.old_base is not None and np.any(
            read.Reference[markup_pos]
            != new_alphabet_info.alphabet.find(args.old_base)
        ):
            fail_reasons[
                "Bad markup pos "
                f"{sorted(set(read.Reference[markup_pos]))} "
                f"strand:{strand}"
            ] += 1
            continue

        # Update the threshold breakers with an mC label not hmC
        read.Reference[markup_pos] = new_alphabet_info.alphabet.find(
            args.new_base[1]
        )
        # write the read to output file
        output_msf.write_read(read.get_read_dictionary())
        fail_reasons["Success"] += 1

    fail_reads_str = "\n".join(
        (
            f"{fn: <8d}\t{fs}"
            for fn, fs in sorted(
                (fail_num, fail_str)
                for fail_str, fail_num in fail_reasons.items()
            )[::-1]
        )
    )
    sys.stderr.write(f"Failed read\n{fail_reads_str}\n")

    input_msf.close()
    output_msf.close()


def get_parser():
    parser = argparse.ArgumentParser(
        description="Add markup of a modified base to signal mappings. "
        "A specified proportion of the most confident calls will be assigned "
        "the new base in the output signal mappings. All input files should "
        "be generated from the same megalodon command."
    )
    parser.add_argument("input_msf", help="Input mapped signal file path")
    parser.add_argument(
        "input_per_read_mod_scores", help="Input per-read mod scores text file"
    )
    parser.add_argument(
        "input_mapping_summary", help="Input mapping summary file"
    )
    parser.add_argument("output_msf", help="Output mapped signal file path")
    parser.add_argument(
        "--new-base",
        nargs=3,
        default=("C", "m", "5mC"),
        help="New base to be added. Three arguments should be 1) canonical "
        "base, 2) modified base single letter code 3) modified base long "
        "name. Default: %(default)s",
    )
    parser.add_argument(
        "--old-base", help="Check that new bases only overwrite this base."
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=28,
        help="Set upper percentile cutoff for markup. Default: %(default)f",
    )
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
