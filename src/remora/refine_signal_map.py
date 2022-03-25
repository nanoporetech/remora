from itertools import product
from dataclasses import dataclass

import numpy as np
from scipy import stats

from remora import RemoraError, log
from remora.constants import (
    DEFAULT_REFINE_HBW,
    DEFAULT_REFINE_SHORT_DWELL_PARAMS,
    DEFAULT_REFINE_ALGO,
    DEFAULT_REFINE_BAND_MIN_STEP,
    REFINE_ALGO_DWELL_PEN_NAME,
    DEFAULT_REFINE_SCALE_ITERS,
)
from remora.refine_signal_map_core import (
    seq_banded_dp,
    extract_levels,
    adjust_seq_band,
)

LOGGER = log.get_logger()


###############
# Short dwell #
###############


def compute_dwell_pen_array(target, limit, weight):
    if limit > target:
        LOGGER.warning(
            f"Requested short dwell limit ({limit}) is greater than target "
            f"dwell ({target}). Setting limit to target."
        )
        limit = target
    return weight * np.square(np.arange(limit, dtype=np.float32) - target)


DEFAULT_REFINE_SHORT_DWELL_PEN = compute_dwell_pen_array(
    *DEFAULT_REFINE_SHORT_DWELL_PARAMS
)


#################
# Re-scale Core #
#################


def rescale_lstsq(dacs, levels, shift, scale):
    norm_sig = (dacs - shift) / scale
    shift_est, scale_est = np.linalg.lstsq(
        np.column_stack([np.ones_like(norm_sig), norm_sig]),
        levels,
        rcond=None,
    )[0]
    new_shift = shift - (scale * shift_est / scale_est)
    new_scale = scale / scale_est
    return new_shift, new_scale


def rough_rescale_lstsq(dacs, levels, shift, scale, quants):
    norm_sig = (dacs - shift) / scale
    norm_qs = np.quantile(norm_sig, quants)
    shift_est, scale_est = np.linalg.lstsq(
        np.column_stack([np.ones_like(norm_qs), norm_qs]),
        np.quantile(levels, quants),
        rcond=None,
    )[0]
    new_shift = shift - (scale * shift_est / scale_est)
    new_scale = scale / scale_est
    return new_shift, new_scale


##########################
# Signal Mapping Refiner #
##########################


def index_from_kmer(kmer, alphabet="ACGT"):
    """Encode string k-mer as integer via len(alphabet)-bit encoding.

    Args:
        kmer (str): kmer string
        alphabet (str): bases used. Default: ACGT

    Returns:
        int: bit encoded kmer

    Example:
        index_from_kmer('AAA', 'ACG')               returns 0
        index_from_kmer('CAAAAAAAA', 'ACGTVWXYX')   returns 65536
    """
    return sum(
        alphabet.find(base) * (len(alphabet) ** kmer_pos)
        for kmer_pos, base in enumerate(kmer[::-1])
    )


@dataclass
class SigMapRefiner:
    """Object to perform scaling and signal mapping refinement before chunk
    extraction.

    Args:
        TODO
    """

    kmer_model_filename: str = None
    do_rough_rescale: bool = True
    scale_iters: int = DEFAULT_REFINE_SCALE_ITERS
    algo: str = DEFAULT_REFINE_ALGO
    half_bandwidth: int = DEFAULT_REFINE_HBW
    sd_params: tuple = None
    do_fix_guage: bool = False

    sd_arr: np.ndarray = DEFAULT_REFINE_SHORT_DWELL_PEN
    _levels_array: np.ndarray = None
    str_kmer_levels: dict = None
    kmer_len: int = None
    center_idx: int = -1
    is_loaded: bool = False

    def __repr__(self):
        if not self.is_loaded:
            return "No Remora signal refine/map settings loaded"
        r_str = (
            f"Loaded {self.kmer_len}-mer table with {self.center_idx + 1} "
            "central position."
        )
        if self.do_rough_rescale:
            r_str += " Rough re-scaling will be executed."
        if self.scale_iters > 0:
            r_str += (
                f" {self.scale_iters} rounds of signal mapping refinement "
                "followed by precise re-scaling will be executed."
            )
        if self.scale_iters >= 0:
            r_str += (
                " Signal mapping refinement will be executed using the "
                f"{self.algo} refinement method (band half width: "
                f"{self.half_bandwidth})."
            )
            if self.algo == REFINE_ALGO_DWELL_PEN_NAME:
                r_str += f" Short dwell penalty array set to {self.sd_params}."
        return r_str

    def load_kmer_table(self):
        self.str_kmer_levels = {}
        with open(self.kmer_model_filename) as kmer_fp:
            self.kmer_len = len(kmer_fp.readline().split()[0])
            kmer_fp.seek(0)
            for line in kmer_fp:
                kmer, level = line.split()
                kmer = kmer.upper()
                if kmer in self.str_kmer_levels:
                    raise RemoraError(
                        f"K-mer found twice in levels file '{kmer}'."
                    )
                if self.kmer_len != len(kmer):
                    raise RemoraError(
                        f"K-mer lengths not all equal '{len(kmer)} != "
                        f"{self.kmer_len}' for {kmer}."
                    )
                try:
                    self.str_kmer_levels[kmer] = float(level)
                    if np.isnan(self.str_kmer_levels[kmer]):
                        self.str_kmer_levels[kmer] = 0
                except ValueError:
                    raise RemoraError(
                        f"Could not convert level to float '{level}'"
                    )
        if len(self.str_kmer_levels) != 4**self.kmer_len:
            raise RemoraError(
                "K-mer table contains fewer entries "
                f"({len(self.str_kmer_levels)}) than expected "
                f"({4 ** self.kmer_len})"
            )

    def determine_dominant_pos(self):
        if self.str_kmer_levels is None:
            return
        sorted_kmers = sorted(
            (level, kmer) for kmer, level in self.str_kmer_levels.items()
        )
        kmer_idx_stats = []
        kmer_summ = ""
        for kmer_idx in range(self.kmer_len):
            kmer_idx_pos = []
            for base in "ACGT":
                kmer_idx_pos.append(
                    [
                        levels_idx
                        for levels_idx, (_, kmer) in enumerate(sorted_kmers)
                        if kmer[kmer_idx] == base
                    ]
                )
            # compute Kruskal-Wallis H-test statistics for non-random ordering
            # of groups, indicating the dominant position within the k-mer
            kmer_idx_stats.append(stats.kruskal(*kmer_idx_pos)[0])
            kmer_summ += f"\t{kmer_idx}\t{kmer_idx_stats[-1]:10.2f}\n"
        self.center_idx = np.argmax(kmer_idx_stats)
        LOGGER.debug(f"K-mer index stats:\n{kmer_summ}")
        LOGGER.debug(f"Choosen central position: {self.center_idx}")

    def __post_init__(self):
        if self._levels_array is not None:
            self.is_loaded = True
            self.kmer_len = int(np.log(self._levels_array.size) / np.log(4))
            assert 4**self.kmer_len == self._levels_array.size
        elif self.kmer_model_filename is not None:
            self.load_kmer_table()
            self.is_loaded = True
            self.determine_dominant_pos()
            if self.do_fix_guage:
                self.fix_gauge()
        if self.sd_params is not None:
            self.sd_arr = compute_dwell_pen_array(*self.sd_params)
        if (
            self.is_loaded
            and self.scale_iters >= 0
            and self.algo == REFINE_ALGO_DWELL_PEN_NAME
        ):
            LOGGER.info(f"Refine short dwell penalty array: {self.sd_arr}")

    def extract_levels(self, int_seq):
        return extract_levels(
            int_seq.astype(np.int32),
            self.levels_array,
            self.kmer_len,
            self.center_idx,
        )

    def fix_gauge(self):
        med = np.median(self.levels_array)
        # note factor to scale MAD to SD
        mad = np.median(np.absolute(self.levels_array - med)) * 1.4826
        self._levels_array = (self.levels_array - med) / mad
        self.str_kmer_levels = {}
        for kmer in product(*["ACGT"] * self.kmer_len):
            self.str_kmer_levels["".join(kmer)] = self._levels_array[
                index_from_kmer(kmer)
            ]

    @property
    def levels_array(self):
        if self._levels_array is None:
            if self.str_kmer_levels is None:
                return
            self._levels_array = np.empty(4**self.kmer_len, dtype=np.float32)
            for kmer, level in self.str_kmer_levels.items():
                self._levels_array[index_from_kmer(kmer)] = level
        return self._levels_array

    def rough_rescale(
        self,
        shift,
        scale,
        seq_to_sig_map,
        int_seq,
        dacs,
        quants=np.arange(0.05, 1, 0.05),
        clip_bases=10,
        use_base_center=True,
    ):
        """Estimate new scaling parameters base on quantiles of levels and
        quantiles of central signal point in each base.

        Args:
            levels (np.array): Estimated reference levels for each base
            quants (np.array): Quantiles to use for rough estimates
            clip_bases (int): Number of bases to clip from either end before
                computing rough rescale estimates.
            use_base_center (bool): Use the central signal point from each base.
                If False, quantiles will be computed from the full signal.
        """
        levels = self.extract_levels(int_seq)
        if use_base_center:
            optim_dacs = dacs[(seq_to_sig_map[:-1] + seq_to_sig_map[1:]) // 2]
            if clip_bases > 0 and levels.size > clip_bases * 2:
                levels = levels[clip_bases:-clip_bases]
                optim_dacs = optim_dacs[clip_bases:-clip_bases]
        else:
            optim_dacs = dacs[seq_to_sig_map[0] : seq_to_sig_map[-1]]
            optim_dacs = (optim_dacs - shift) / scale
        return rough_rescale_lstsq(optim_dacs, levels, shift, scale, quants)

    def rescale(
        self,
        levels,
        dacs,
        shift,
        scale,
        seq_to_sig_map,
        dwell_filter_pctls=(10, 90),
        min_abs_level=0.2,
        edge_filter_bases=10,
        min_levels=10,
    ):
        """Estimate new scaling parameters base on current signal mapping to
        estimated levels.

        Args:
            levels (np.array): Estimated reference levels for each base
            dacs (np.ndarray): Unnormalized DAC signal
            shift (float): Shift from dac to normalized signal. via formula:
                norm = (dac - shift) / scale
            scale (float): Scale from dac to normalized signal
            seq_to_sig_map (np.ndarray): Position within signal array assigned to
                each base in seq
            dwell_filter_pctls (tuple): Lower and upper percentile values to
                filter short and stalled bases from estimation
            min_abs_level (float): Minimum (absolute values) level to include
                in computaiton
            edge_filter_bases (int): Number of bases at the edge of a read to
                remove from optimization computation
            min_levels (int): Minimum number of levels to perform re-scaling
        """
        with np.errstate(invalid="ignore"):
            dacs_cumsum = np.empty(dacs.size + 1)
            dacs_cumsum[0] = 0
            dacs_cumsum[1:] = np.cumsum(dacs)
            dwells = np.diff(seq_to_sig_map)
            dac_means = np.diff(dacs_cumsum[seq_to_sig_map]) / dwells

        # filter bases base on lower and upper percentile of dwell to remove
        # regions of poor signal assignment
        # estimate dwell limits (these limits are treated as exclusive
        # boundaries for filtering as the lower boundary may be 1)
        dwells = np.diff(seq_to_sig_map)
        dwell_min, dwell_max = np.percentile(dwells, dwell_filter_pctls)
        edge_filter = np.full(dwells.size, True, dtype=np.bool)
        if edge_filter_bases > 0:
            edge_filter[:edge_filter_bases] = False
            edge_filter[-edge_filter_bases:] = False
        valid_bases = np.logical_and.reduce(
            (
                dwells > dwell_min,
                dwells < dwell_max,
                np.abs(levels - np.mean(levels)) > min_abs_level,
                np.logical_not(np.isnan(dac_means)),
                edge_filter,
            )
        )
        filt_levels = levels[valid_bases]
        filt_dacs = dac_means[valid_bases]
        if filt_levels.size < min_levels:
            raise RemoraError("Too few positions")

        return rescale_lstsq(filt_dacs, filt_levels, shift, scale)

    def refine_sig_map(self, shift, scale, seq_to_sig_map, int_seq, dacs):
        levels = self.extract_levels(int_seq)
        dacs = dacs[seq_to_sig_map[0] : seq_to_sig_map[-1]]
        sig_st = seq_to_sig_map[0]
        seq_to_sig_map = seq_to_sig_map - sig_st
        # 0 indicates 1 round of sig map refine w/o rescaling
        for _ in range(max(1, self.scale_iters)):
            seq_to_sig_map, _, _, _, _ = refine_signal_mapping(
                (dacs - shift) / scale,
                seq_to_sig_map,
                levels,
                self.half_bandwidth,
                self.algo,
                self.sd_arr,
            )
            # 0 specifies no re-scaling
            if self.scale_iters > 0:
                try:
                    shift, scale = self.rescale(
                        levels, dacs, shift, scale, seq_to_sig_map
                    )
                except RemoraError as e:
                    LOGGER.debug(f"Remora rescaling error: {self.read_id} {e}")
                    break
        return seq_to_sig_map + sig_st, shift, scale

    def get_save_kwargs(self):
        return {
            "refine_kmer_levels": self._levels_array,
            "refine_kmer_center_idx": self.center_idx,
            "refine_do_rough_rescale": int(self.do_rough_rescale),
            "refine_scale_iters": self.scale_iters,
            "refine_algo": self.algo,
            "refine_half_bandwidth": self.half_bandwidth,
            "refine_sd_arr": self.sd_arr,
        }

    @classmethod
    def load_from_np_savez(cls, data):
        return cls(
            _levels_array=data["refine_kmer_levels"],
            center_idx=int(data["refine_kmer_center_idx"].item()),
            do_rough_rescale=bool(int(data["refine_do_rough_rescale"].item())),
            scale_iters=int(data["refine_scale_iters"].item()),
            algo=str(data["refine_algo"].item()),
            half_bandwidth=int(data["refine_half_bandwidth"].item()),
            sd_arr=data["refine_sd_arr"],
        )


###########
# Banding #
###########


def compute_sig_band(bps, levels, bhw=DEFAULT_REFINE_HBW, is_banded=True):
    """Compute band over which to explore possible paths. Band is represented
    in sequence/level coordinates at each signal position.

    Args:
        bps (np.ndarray): Integer array containing breakpoints
        levels (np.ndarray): float array containing expected signal levels. May
            contain np.NAN values. Band will be constructed to maintain path
            through NAN regions.
        bhw (int): band half width
        is_banded (bool): Should bhw be applied over full path. If False only
            NAN levels restrict the computed band.

    Returns:
        int32 np.ndarray with shape (2, sig_len = bps[-1] - bps[0]). The first
        row contains the lower band boundaries in sequence coordinates and the
        second row contains the upper boundaries in sequence coordinates.
    """
    if is_banded and bhw is None:
        raise RemoraError("Cannot compute band with half width of None.")
    seq_len = levels.size
    if bps.size - 1 != seq_len:
        raise RemoraError("Breakpoints must be one longer than levels.")
    sig_len = bps[-1] - bps[0]
    seq_indices = np.repeat(np.arange(seq_len), np.diff(bps))

    # Calculate bands
    # The 1st row consists of the start indices (inc) and the 2nd row
    # consists of the end indices (exc) of the valid rows for each col.
    band = np.empty((2, sig_len), dtype=np.int32)
    if is_banded:
        # Use specific band if instructed
        band[0, :] = np.maximum(seq_indices - bhw, 0)
        band[1, :] = np.minimum(seq_indices + bhw + 1, seq_len)
    else:
        # otherwise specify entire input matrix
        band[0, :] = 0
        band[1, :] = seq_len

    # Modify bands based on invalid levels
    nan_mask = np.in1d(seq_indices, np.nonzero(np.isnan(levels)))
    nan_sig_indices = np.where(nan_mask)[0]
    nan_seq_indices = seq_indices[nan_mask]
    band[0, nan_sig_indices] = nan_seq_indices
    band[1, nan_sig_indices] = nan_seq_indices + 1
    # Modify bands close to invalid levels so monotonically increasing
    band[0, :] = np.maximum.accumulate(band[0, :])
    band[1, :] = np.minimum.accumulate(band[1, ::-1])[::-1]

    return band


def validate_band(band, sig_len=None, seq_len=None, is_sig_band=True):
    """Validate that band is valid and agrees with input data.

    Args:
        band (np.array): int32 array with shape (2, sig_len or seq_len). The
            first row contains the lower band boundaries and the second row
            contains the upper boundaries.
        sig_len (int): Length of signal associated with band
        seq_len (int): Length of sequence/levels associated with band
        is_sig_band (bool): Does the provided band specify sequence/level
            positions for each signal position? If not it is assumed that the
            band contains signal positions for each sequence/level position.

    Raises:
        RemoraError if any portion of the band is determined to be invalid.
    """
    # first coordinate 0, last coordinate signal length
    if band[0, 0] != 0:
        raise RemoraError("Band does not start with 0 coordinate.")

    # ends all greater than starts
    if np.diff(band, axis=0)[0].min() <= 0:
        raise RemoraError("Band contains 0-length region")
    # monotonic start and end postions
    if np.diff(band[0]).min() < 0:
        raise RemoraError(
            "Band start positions are not monotonically increasing"
        )
    if np.diff(band[1]).min() < 0:
        raise RemoraError("Band end positions are not monotonically increasing")

    # if provided check that start and end coordinates agree with signal and
    # levels.
    if is_sig_band:
        if sig_len is not None and band.shape[1] != sig_len:
            LOGGER.debug(f"Invalid sig_band length: {band.shape[1]} {sig_len}")
            raise RemoraError("Invalid sig_band length")
        if seq_len is not None and band[1, -1] != seq_len:
            LOGGER.debug(
                f"Invalid sig_band end coordinate: {band[1, -1]} {seq_len}"
            )
            raise RemoraError("Invalid sig_band end coordinate")
    else:
        if sig_len is not None and band[1, -1] != sig_len:
            LOGGER.debug(
                f"Invalid seq_band end coordinate: {band[1, -1]} {sig_len}"
            )
            raise RemoraError("Invalid seq_band end coordinate")
        if seq_len is not None and band.shape[1] != seq_len:
            LOGGER.debug(f"Invalid sig_band length: {band.shape[1]} {seq_len}")
            raise RemoraError("Invalid sig_band length")


def convert_to_seq_band(sig_band):
    """Convert band with sig_len entries containing upper and lower band
    boundaries in base coordinates to a seq_len entries contraining upper and
    lower band boundaries in signal space.

    Args:
        sig_band (np.array): int32 array with shape (2, sig_len). The first row
            contains the lower band boundaries in sequence coordinates and the
            second row contains the upper boundaries in sequence coordinates.

    Returns:
        int32 np.ndarray with shape (2, seq_len = sig_band[1, -1]). The first
        row contains the lower band boundaries in signal coordinates and the
        second row contains the upper boundaries in signal coordinates.
    """
    sig_len = sig_band.shape[1]
    seq_len = sig_band[1, -1]
    seq_band = np.zeros((2, seq_len), dtype=np.int32)
    seq_band[1, :] = sig_len

    # upper signal coordinates define lower sequence boundaries
    lower_sig_pos = np.nonzero(np.ediff1d(sig_band[1, :], to_begin=0))[0]
    lower_base_pos = sig_band[1, lower_sig_pos - 1]
    seq_band[0, lower_base_pos] = lower_sig_pos
    seq_band[0, :] = np.maximum.accumulate(seq_band[0, :])

    upper_sig_pos = np.nonzero(np.ediff1d(sig_band[0, :], to_begin=0))[0]
    upper_base_pos = sig_band[0, upper_sig_pos]
    seq_band[1, upper_base_pos - 1] = upper_sig_pos
    seq_band[1, :] = np.minimum.accumulate(seq_band[1, ::-1])[::-1]

    return seq_band


######################
# Signal Map Refiner #
######################


def refine_signal_mapping(
    signal,
    seq_to_sig_map,
    levels,
    band_half_width=DEFAULT_REFINE_HBW,
    refine_algo=DEFAULT_REFINE_ALGO,
    short_dwell_pen=DEFAULT_REFINE_SHORT_DWELL_PEN,
):
    """Refine input signal mapping to minimize difference difference between
    signal and levels.

    Args:
        signal (np.ndarray): Float32 array containing normalized signal values.
        seq_to_sig_map (np.ndarray): Int32 array with locations within
            signal for each base represented by levels. `seq_to_sig_map.size`
            should equal `levels.size + 1` in order to include the end of the
            last base. Values should be monotonically increasing (ties allowed).
        levels (np.ndarray): Float32 array containing estimated levels.
        band_half_width (int): Half bandwidth

    Returns:
        2-tuple containing:
            1. Signal mapping
            2. Mapping score
    """
    # Note that there is no guarantee the seq_to_sig_map go right to the end
    # of the signal provided so may need to trim signal
    signal = signal[seq_to_sig_map[0] : seq_to_sig_map[-1]]
    sig_map_start = 0
    if seq_to_sig_map[0] != 0:
        sig_map_start = seq_to_sig_map[0]
        seq_to_sig_map = np.copy(seq_to_sig_map)
        seq_to_sig_map -= seq_to_sig_map[0]

    # TODO compute seq band directly with anti-diagonal band
    sig_band = compute_sig_band(
        seq_to_sig_map,
        levels,
        bhw=band_half_width,
    )
    seq_band = convert_to_seq_band(sig_band)
    adjust_seq_band(seq_band, min_step=DEFAULT_REFINE_BAND_MIN_STEP)
    validate_band(
        seq_band,
        sig_len=signal.shape[0],
        seq_len=levels.shape[0],
        is_sig_band=False,
    )

    # Change nans to zeros so nans don't break things
    temp_levels = np.copy(levels)
    temp_levels[np.isnan(levels)] = 0
    all_scores, path, traceback, base_offsets = seq_banded_dp(
        signal.astype(np.float32),
        temp_levels.astype(np.float32),
        seq_band,
        short_dwell_pen,
        refine_algo,
    )
    return path + sig_map_start, all_scores, traceback, seq_band, base_offsets
