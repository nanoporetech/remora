import argparse
import multiprocessing as mp

import numpy as np

from remora import constants
from remora.util import queue_iter
from remora.refine_signal_map import SigMapRefiner
from remora.prepare_train_data import fill_reads_q

import matplotlib

if True:
    # Agg appears to be the most robust backend when only saving plots.
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

SEQ_COLS = {
    0: "#00CC00",
    1: "#0000CC",
    2: "#FFB300",
    3: "#CC0000",
    -1: "#000000",
}


def plot_read(
    read,
    sig_map_refiner,
    all_scores=None,
    traceback=None,
    seq_band=None,
    base_offsets=None,
    b_st=None,
    b_en=None,
    s_st=None,
    s_en=None,
    scale_scores=False,
):
    levels = sig_map_refiner.extract_levels(read.int_seq)
    if b_st is None:
        b_st = 0
    if b_en is None:
        b_en = read.int_seq.size
    if s_st is None:
        s_st = (
            read.seq_to_sig_map[b_st] if seq_band is None else seq_band[0, b_st]
        )
    if s_en is None:
        s_en = (
            read.seq_to_sig_map[b_en + 1]
            if seq_band is None
            else seq_band[1, b_en - 1]
        )
    print(f"{read.read_id} bases: {b_st}-{b_en} signal: {s_st}-{s_en}")

    # plot only the signal and levels
    if all_scores is None:
        plt.figure(figsize=(20, 5))
        _ = plt.plot(np.arange(s_en - s_st), read.sig[s_st:s_en])
        for b_idx, (bi_st, bi_en) in enumerate(
            zip(read.seq_to_sig_map[:-1], read.seq_to_sig_map[1:])
        ):
            if not (s_st <= bi_st <= s_en):
                continue
            plt.axvline(bi_st - s_st, color="k", linewidth=0.3)
            plt.plot(
                (bi_st - s_st, max(bi_st - s_st + 1, bi_en - s_st)),
                (levels[b_idx], levels[b_idx]),
                color="r",
            )
            plt.text(x=bi_st - s_st, y=1.5, s=read.str_seq[b_idx])
            plt.text(x=bi_st - s_st, y=-1.5, s=b_idx)
        return

    p_data = np.full((b_en - b_st, s_en - s_st - 1), np.NAN, dtype=float)
    tb_data = np.full((b_en - b_st, s_en - s_st - 1), np.NAN, dtype=float)
    for p_idx, (band_ofst, (bi_st, bi_en)) in enumerate(
        zip(base_offsets[b_st:b_en], seq_band[:, b_st:b_en].T)
    ):
        for band_pos in range(bi_en - bi_st):
            p_sig_idx = band_pos + bi_st - s_st
            if not (0 < p_sig_idx < s_en - s_st):
                continue
            p_data[p_idx, band_pos + bi_st - s_st - 1] = all_scores[
                band_ofst + band_pos
            ]
            tb_data[p_idx, band_pos + bi_st - s_st - 1] = traceback[
                band_ofst + band_pos
            ]
    if scale_scores:
        p_data = np.log10(p_data / np.nanmin(p_data, axis=0))
    fig, ax = plt.subplots(
        3,
        2,
        sharex="col",
        gridspec_kw={"width_ratios": [100, 5]},
        figsize=(20, 10),
    )
    ax[0, 1].remove()
    _ = ax[0, 0].plot(np.arange(s_en - s_st), read.sig[s_st:s_en])
    for b_idx, (bi_st, bi_en) in enumerate(
        zip(read.seq_to_sig_map[:-1], read.seq_to_sig_map[1:])
    ):
        if not (s_st <= bi_st <= s_en):
            continue
        ax[0, 0].axvline(bi_st - s_st, color="k", linewidth=0.3)
        ax[0, 0].plot(
            (bi_st - s_st, max(bi_st - s_st + 1, bi_en - s_st)),
            (levels[b_idx], levels[b_idx]),
            color="r",
        )
        ax[0, 0].text(x=bi_st - s_st, y=1.5, s=read.str_seq[b_idx])
        ax[0, 0].text(x=bi_st - s_st, y=-1.5, s=b_idx, rotation=-30)
    pp_y = np.repeat(np.arange(b_en - b_st), 2) + 0.5
    pp_x = np.empty_like(pp_y)
    pp_x[::2] = read.seq_to_sig_map[b_st:b_en] - 0.5 - s_st
    pp_x[1::2] = read.seq_to_sig_map[b_st + 1 : b_en + 1] - 1.5 - s_st
    _ = sns.heatmap(p_data, ax=ax[1, 0], cbar_ax=ax[1, 1], cmap="magma_r")
    _ = ax[1, 0].plot(pp_x, pp_y, color="k")
    _ = sns.heatmap(tb_data, ax=ax[2, 0], cbar_ax=ax[2, 1], cmap="viridis_r")
    _ = ax[2, 0].plot(pp_x, pp_y, color="k")
    return p_data, tb_data


def main(args):
    sig_map_refiner = SigMapRefiner(
        kmer_model_filename=args.refine_kmer_level_table,
        do_rough_rescale=args.refine_rough_rescale,
        scale_iters=args.refine_scale_iters,
        algo=args.refine_algo,
        half_bandwidth=args.refine_half_bandwidth,
        sd_params=args.refine_short_dwell_parameters,
        do_fix_guage=True,
    )
    reads_q = mp.Queue()
    filler_p = mp.Process(
        target=fill_reads_q,
        args=(reads_q, args.input_reads),
        daemon=True,
        name="ReadQueueFiller",
    )
    filler_p.start()

    pdf_fp = PdfPages(args.out_pdf)
    reads_iter = queue_iter(reads_q)
    for _ in range(args.num_reads):
        read = next(reads_iter)
        read.refine_signal_mapping(sig_map_refiner)
        plot_read(
            read,
            sig_map_refiner,
            b_st=None if args.seq_range is None else args.seq_range[0],
            b_en=None if args.seq_range is None else args.seq_range[1],
            s_st=None if args.sig_range is None else args.sig_range[0],
            s_en=None if args.sig_range is None else args.sig_range[1],
        )
        pdf_fp.savefig(bbox_inches="tight")
        plt.close()
    pdf_fp.close()


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_reads",
        help="Taiyaki mapped signal or RemoraReads pickle file.",
    )

    parser.add_argument(
        "--out-pdf",
        default="remora_signal_mapping.pdf",
        help="Plots output path",
    )
    parser.add_argument(
        "--sig-range", type=int, nargs=2, help="Signal point range to plot"
    )
    parser.add_argument(
        "--seq-range", type=int, nargs=2, help="Sequence range to plot"
    )
    parser.add_argument(
        "--num-reads", type=int, default=1, help="Number of reads to plot"
    )

    refine_grp = parser.add_argument_group("Signal Mapping Refine Arguments")
    refine_grp.add_argument(
        "--refine-kmer-level-table",
        required=True,
        help="Tab-delimited file containing no header and two fields: "
        "1. string k-mer sequence and 2. float expected normalized level. "
        "All k-mers must be the same length and all combinations of the bases "
        "'ACGT' must be present in the file.",
    )
    refine_grp.add_argument(
        "--refine-rough-rescale",
        action="store_true",
        help="Apply a rough rescaling using quantiles of signal+move table "
        "and levels.",
    )
    refine_grp.add_argument(
        "--refine-scale-iters",
        default=constants.DEFAULT_REFINE_SCALE_ITERS,
        type=int,
        help="Number of iterations of signal mapping refinement and signal "
        "re-scaling to perform. Set to 0 (default) in order to perform signal "
        "mapping refinement, but skip re-scaling. Set to -1 to skip signal "
        "mapping (potentially using levels for rough rescaling).",
    )
    refine_grp.add_argument(
        "--refine-half-bandwidth",
        default=constants.DEFAULT_REFINE_HBW,
        type=int,
        help="Half bandwidth around signal mapping over which to search for "
        "new path.",
    )
    refine_grp.add_argument(
        "--refine-algo",
        default=constants.DEFAULT_REFINE_ALGO,
        choices=constants.REFINE_ALGOS,
        help="Refinement algorithm to apply (if kmer level table is provided).",
    )
    refine_grp.add_argument(
        "--refine-short-dwell-parameters",
        default=constants.DEFAULT_REFINE_SHORT_DWELL_PARAMS,
        type=float,
        nargs=3,
        metavar=("TARGET", "LIMIT", "WEIGHT"),
        help="Short dwell penalty refiner parameters. Dwells shorter than "
        "LIMIT will be penalized a value of `WEIGHT * (dwell - TARGET)^2`. "
        "Default: %(default)s",
    )

    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
