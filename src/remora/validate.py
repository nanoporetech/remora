import atexit
from collections import defaultdict

import pysam
import numpy as np
from tqdm import tqdm


def parse_mods(fn, regs, mod_b, is_mod, full_fp):
    probs = []
    with pysam.AlignmentFile(fn, check_sq=False) as bam:
        for read in tqdm(bam, smoothing=0):
            if read.modified_bases is None:
                continue
            if regs is not None and read.reference_name not in regs:
                continue
            # note read.modified_bases stores positions in forward strand
            # query sequence coordinates
            mod_pos_probs = [
                pos_prob
                for (_, _, mod_i), mod_values in read.modified_bases.items()
                for pos_prob in mod_values
                if mod_i == mod_b
            ]
            if len(mod_pos_probs) == 0:
                continue
            q_to_r = dict(read.get_aligned_pairs(matches_only=True))
            for q_pos, m_prob in mod_pos_probs:
                try:
                    r_pos = q_to_r[q_pos]
                except KeyError:
                    continue
                if regs is not None and r_pos not in regs[read.reference_name]:
                    continue
                if full_fp is not None:
                    full_fp.write(
                        f"{read.query_name}\t{q_pos}\t{m_prob}\t"
                        f"{read.reference_name}\t{r_pos}\t{is_mod}\n"
                    )
                probs.append(m_prob)
    return np.array(probs)


def calc_metrics(can_probs, mod_probs, low_thresh, high_thresh):
    tn = np.sum(can_probs <= low_thresh)
    fn = np.sum(mod_probs <= low_thresh)
    tp = np.sum(mod_probs >= high_thresh)
    fp = np.sum(can_probs >= high_thresh)
    err_rate = (fp + fn) / (tn + fn + tp + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tot_calls = can_probs.size + mod_probs.size
    num_filt = tot_calls - (tn + fn + tp + fp)
    frac_filt = num_filt / tot_calls
    metrics_str = (
        f"{err_rate:.6f}\t{fpr:.6f}\t{fnr:.6f}\t{frac_filt:.6f}\t{tot_calls}"
    )
    return err_rate, fpr, fnr, frac_filt, metrics_str


def validate_from_modbams(args):
    if args.regions_bed is None:
        regs = None
    else:
        regs = defaultdict(set)
        with open(args.regions_bed) as regs_fp:
            for line in regs_fp:
                ctg, st, en = line.split()[:3]
                regs[ctg].update(range(int(st), int(en)))
    full_fp = None
    if args.full_output_filename is not None:
        full_fp = open(args.full_output_filename, "w")
        atexit.register(full_fp.close)
        full_fp.write(
            "query_name\tquery_pos\tmod_prob\tref_name\tref_pos\tis_mod\n"
        )
    can_probs = np.concatenate(
        [
            parse_mods(can_fn, regs, args.mod_base, False, full_fp)
            for can_fn in args.can_bams
        ]
    )
    mod_probs = np.concatenate(
        [
            parse_mods(mod_fn, regs, args.mod_base, True, full_fp)
            for mod_fn in args.mod_bams
        ]
    )

    if not args.allow_unbalanced:
        if can_probs.size > mod_probs.size:
            np.random.shuffle(can_probs)
            can_probs = can_probs[: mod_probs.size]
        else:
            np.random.shuffle(mod_probs)
            mod_probs = mod_probs[: can_probs.size]

    all_probs = np.concatenate([can_probs, mod_probs])
    probs_cs = np.cumsum(np.bincount(all_probs)) / all_probs.size

    if args.fixed_thresh is not None:
        lt = int(args.fixed_thresh[0] * 255)
        ht = int(args.fixed_thresh[1] * 255)
        if lt >= ht:
            ht = lt + 1
        print(
            f"{calc_metrics(can_probs, mod_probs, lt, ht)[-1]}\t"
            f"{lt}_{ht}\t{args.name}"
        )
        return

    can_cs = np.cumsum(np.bincount(can_probs)) / can_probs.size
    mod_cs = np.cumsum(np.bincount(mod_probs)) / mod_probs.size
    pct_threshs = []
    for lt in range(256):
        lcs = probs_cs[lt]
        try:
            ht = next(
                i
                for i, cs in enumerate(probs_cs)
                if cs - lcs >= args.pct_filt / 100
            )
        except StopIteration:
            continue
        pct_threshs.append((lt, ht))
    # find thresold with min error rate
    lt, ht = sorted(
        [(mod_cs[lt] - can_cs[ht], lt, ht) for lt, ht in pct_threshs]
    )[0][1:]
    if lt >= ht:
        ht = lt + 1
    print(
        f"{calc_metrics(can_probs, mod_probs, lt, ht)[-1]}\t{lt}_{ht}\t"
        f"{args.name}"
    )

    """ Alternative methods to find fixed percent of reads to filter

    # find threshold with same upper and lower probability threshold
    lt, ht = sorted([
        (np.abs(lt - (255 - ht)), lt, ht) for lt, ht in pct_threshs
    ])[0][1:]
    print (
        f"{calc_metrics(can_probs, mod_probs, lt, ht)[-1]}\t{lt}_{ht}\t"
        f"{args.name}"
    )

    # find threshold balancing fpr and fnr
    lt, ht = sorted([
        (np.abs(mod_cs[lt] - (1 - can_cs[ht])), lt, ht)
        for lt, ht in pct_threshs
    ])[0][1:]
    print (
        f"{calc_metrics(can_probs, mod_probs, lt, ht)[-1]}\t{lt}_{ht}\t"
        f"{args.name}"
    )

    # find thresholds centered on CDF
    lower_prob = 0.5 - (args.pct_filt / 200)
    upper_prob = 0.5 + (args.pct_filt / 200)
    lt = np.argmin(np.abs(probs_cs - lower_prob))
    ht = np.argmin(np.abs(probs_cs - upper_prob))
    print (
        f"{calc_metrics(can_probs, mod_probs, lt, ht)[-1]}\t{lt}_{ht}\t"
        f"{args.name}"
    )
    """
