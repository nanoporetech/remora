import numpy as np

DEFAULT_START_TRIM = 1
DEFAULT_END_TRIM = 1


def clip_sig(sig, seq_to_sig):
    sig = sig[seq_to_sig[0] : seq_to_sig[-1]]
    return sig, seq_to_sig - seq_to_sig[0]


def cumsum0(sig):
    sig_cs = np.empty(sig.size + 1)
    sig_cs[0] = 0
    sig_cs[1:] = np.cumsum(sig)
    return sig_cs


def compute_cumsum_and_mean(sig, seq_to_sig, dwells):
    sig_cs = cumsum0(sig)
    with np.errstate(divide="ignore", invalid="ignore"):
        base_means = np.diff(sig_cs[seq_to_sig]) / dwells
        base_means[np.isinf(base_means)] = np.nan
    return base_means


def compute_trim_cumsum_and_mean(
    sig,
    seq_to_sig,
    dwells,
    st_trim=DEFAULT_START_TRIM,
    en_trim=DEFAULT_END_TRIM,
):
    sig_cs = cumsum0(sig)
    trim_sts = np.minimum(sig.size, seq_to_sig[:-1] + st_trim)
    trim_ens = np.maximum(0, seq_to_sig[1:] - en_trim)
    trim_sums = sig_cs[trim_ens] - sig_cs[trim_sts]
    trim_dwells = np.maximum(0, dwells - st_trim - en_trim)
    with np.errstate(divide="ignore", invalid="ignore"):
        trim_means = trim_sums / trim_dwells
        trim_means[np.isinf(trim_means)] = np.nan
    return trim_means


def compute_dwell(sig, seq_to_sig, **kwargs):
    return {"dwell": np.diff(seq_to_sig).astype(np.float32)}


def compute_dwell_mean(sig, seq_to_sig, **kwargs):
    dwells = compute_dwell(sig, seq_to_sig, **kwargs)["dwell"]
    sig, seq_to_sig = clip_sig(sig, seq_to_sig)
    base_means = compute_cumsum_and_mean(sig, seq_to_sig, dwells)
    return {"dwell": dwells, "mean": base_means}


def compute_dwell_mean_sd(sig, seq_to_sig, **kwargs):
    dwells = compute_dwell(sig, seq_to_sig, **kwargs)["dwell"]
    sig, seq_to_sig = clip_sig(sig, seq_to_sig)
    base_means = compute_cumsum_and_mean(sig, seq_to_sig, dwells)
    ss_cs = cumsum0(np.square(sig))
    with np.errstate(divide="ignore", invalid="ignore"):
        base_sds = np.sqrt(
            np.maximum(
                np.diff(ss_cs[seq_to_sig]) / dwells - np.square(base_means),
                0,
            )
        )
        base_sds[np.isinf(base_sds)] = np.nan
    return {"dwell": dwells, "mean": base_means, "sd": base_sds}


def compute_trimmean(sig, seq_to_sig, **kwargs):
    st_trim = kwargs.get("start_trim", DEFAULT_START_TRIM)
    en_trim = kwargs.get("end_trim", DEFAULT_END_TRIM)

    dwells = compute_dwell(sig, seq_to_sig)["dwell"]
    sig, seq_to_sig = clip_sig(sig, seq_to_sig)
    trim_means = compute_trim_cumsum_and_mean(
        sig, seq_to_sig, dwells, st_trim, en_trim
    )
    return {"dwells": dwells, "trimmean": trim_means}


def compute_trimmean_trimsd(sig, seq_to_sig, **kwargs):
    st_trim = kwargs.get("start_trim", DEFAULT_START_TRIM)
    en_trim = kwargs.get("end_trim", DEFAULT_END_TRIM)

    dwells = compute_dwell(sig, seq_to_sig)["dwell"]
    sig, seq_to_sig = clip_sig(sig, seq_to_sig)
    trim_means = compute_trim_cumsum_and_mean(
        sig, seq_to_sig, dwells, st_trim, en_trim
    )
    ss_cs = cumsum0(np.square(sig))
    trim_dwells = np.maximum(0, dwells - st_trim - en_trim)

    trim_sts = np.minimum(sig.size, seq_to_sig[:-1] + st_trim)
    trim_ens = np.maximum(0, seq_to_sig[1:] - en_trim)
    trim_ss_sums = ss_cs[trim_ens] - ss_cs[trim_sts]
    trim_dwells = np.maximum(0, dwells - st_trim - en_trim)
    with np.errstate(divide="ignore", invalid="ignore"):
        trim_sds = np.sqrt(
            np.maximum(
                0,
                (trim_ss_sums / trim_dwells) - np.square(trim_means),
            )
        )
        trim_sds[np.isinf(trim_sds)] = np.nan
    return {"dwell": dwells, "trimmean": trim_means, "trimsd": trim_sds}


METRIC_FUNCS = {
    "dwell": compute_dwell,
    "dwell_mean": compute_dwell_mean,
    "dwell_mean_sd": compute_dwell_mean_sd,
    "dwell_trimmean": compute_trimmean,
    "dwell_trimmean_trimsd": compute_trimmean_trimsd,
}
