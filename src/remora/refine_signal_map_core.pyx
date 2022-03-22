# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np

from remora import RemoraError
from remora.constants import REFINE_ALGO_VIT_NAME, REFINE_ALGO_DWELL_PEN_NAME

cimport cython
from libc.math cimport HUGE_VALF

cdef float LARGE_SCORE = 100

# function pointer for core by-base scoring functions
ctypedef void (*core_func_ptr)(
    float[::1],        # curr_scores
    int[::1],          # curr_tb
    const float[::1],  # prev_scores
    const float,       # curr_level
    const float[::1],  # curr_signal
    int,               # band_start_diff
    const float[::1],  # dwell_penalty
)


###########
# Banding #
###########

def adjust_seq_band(int[:, ::1] seq_band, int min_step=1):
    """Adjust band boundaries to disallow invalid paths. Namely, make sure each
    band start and end is at least one greater than the previous.

    Note input band will be updated in place

    Args:
        seq_band (np.ndarray): int32 np.ndarray with shape
            (2, seq_len = levels.shape[0]). The first row contains the lower
            band boundaries in signal coordinates and the second row contains
            the upper boundaries in signal coordinates. The first lower bound
            should be 0 and last upper bound should be signal.shape[0].
       min_step (int): Minimum step between one base and the next to enforce
            in band adjustment.
    """
    cdef int band_min = seq_band[0, 0]
    cdef int seq_pos
    # fix starts to make sure each start is at least min_step less than prev
    for seq_pos in range(seq_band.shape[1] - 2, -1, -1):
        if seq_band[0, seq_pos] > seq_band[0, seq_pos + 1] - min_step:
            seq_band[0, seq_pos] = seq_band[0, seq_pos + 1] - min_step
    # proceed through beginning of band ensuring only valid positions
    seq_band[0, 0] = band_min
    seq_pos = 1
    while seq_band[0, seq_pos] <= seq_band[0, seq_pos - 1]:
        seq_band[0, seq_pos] = seq_band[0, seq_pos - 1] + 1
        seq_pos += 1

    cdef int band_max = seq_band[1, seq_band.shape[1] - 1]
    # fix ends to make sure each end is at least min_step more than prev
    for seq_pos in range(1, seq_band.shape[1]):
        if seq_band[1, seq_pos] < seq_band[1, seq_pos - 1] + min_step:
            seq_band[1, seq_pos] = seq_band[1, seq_pos - 1] + min_step
    # proceed through end of band ensuring only valid positions
    seq_band[1, seq_band.shape[1] - 1] = band_max
    seq_pos = seq_band.shape[1] - 2
    while seq_band[1, seq_pos] >= seq_band[1, seq_pos + 1]:
        seq_band[1, seq_pos] = seq_band[1, seq_pos + 1] - 1
        seq_pos -= 1


####################
# Level Extraction #
####################

cdef inline int index_from_int_kmer(
    const int[::1] int_kmer,
    int kmer_len
):
    cdef int idx = 0
    cdef int kmer_pos
    for kmer_pos in range(kmer_len):
        idx += int_kmer[kmer_len - kmer_pos - 1] * (4 ** kmer_pos)
    return idx


def extract_levels(
    const int[::1] int_seq,
    const float[::1] int_kmer_levels,
    int kmer_len,
    int center_idx
):
    levels = np.zeros(int_seq.shape[0], dtype=np.float32)
    cdef int pos
    for pos in range(int_seq.shape[0] - kmer_len):
        levels[pos + center_idx] = int_kmer_levels[
            index_from_int_kmer(int_seq[pos : pos + kmer_len], kmer_len)
        ]
    return levels


######################
# Mapping Refinement #
######################

cdef inline float score(float s, float l):
    """Find squared difference between sample and level.

    Args:
        s (float): sample
        l (float): level
    """
    cdef float tmp = s - l
    return tmp * tmp


cdef void banded_traceback(
    int[::1] path,
    const int[:, ::1] seq_band,
    const int[::1] base_offsets,
    const int[::1] traceback,
):
    """Perform traceback to determine path from a forward pass.

    Args:
        path (1D int array): To be populated by this function. Path will
            contain start position for each base within the signal array.
        seq_band (const 2D int array): Contains band boundaries for each base in
            signal coordinates.
        base_offsets (const 1D int array): Offset to the beginning of each base
            within the ragged traceback array.
        traceback (const 1D int array): Ragged array representing each position
            within the band. Each element contains the number of signal points
            backwards until the first point assigned to this base.
    """
    cdef int sig_lookup_pos, next_sig_offset, base_idx
    # set start to 0 and end to sig_len
    path[0] = 0
    path[path.shape[0] - 1] = seq_band[1, seq_band.shape[1] - 1]
    for base_idx in range(path.shape[0] - 2, 0, -1):
        # signal position to lookup for this traceback step
        sig_lookup_pos = path[base_idx + 1] - 1
        next_sig_offset = traceback[
            base_offsets[base_idx] + sig_lookup_pos - seq_band[0, base_idx]
        ]
        # record position in which base_idx starts
        path[base_idx] = sig_lookup_pos - next_sig_offset

cdef void banded_forward_dwell_penalty_step(
    float[::1] curr_scores,
    int[::1] curr_tb,
    float[::1] prev_scores,
    const float curr_level,
    const float[::1] curr_signal,
    int band_start_diff,
    const float[::1] dwell_penalty,
):
    """Process one base using Viterbi path scoring with squared error between
    signal and levels plus a penalty for short dwells (number of consecutive
    stay states).

    Args:
        curr_scores (1D float array): To be populated by this function. Elements
            will be forward scores at each position
        curr_tb (1D int array): To be populated by this function. Elements will
            be the number of signal points backwards until the first point
            assigned to this base
        prev_scores (const 1D float array): Forward scores for the previous base
        curr_level (float): Expected signal level for the current base
        curr_signal (const 1D float array): Signal values for the current
            base's band
        band_start_diff (int): Difference in starting coordinates between
            current and previous base
        dwell_penalty (const float 1D array): Penalties for short dwells
    """
    cdef int band_pos, dwell_idx
    cdef float running_pos_score, pos_score

    # compute un-penalized band position scores for lookup after dwell_penalty
    # range is searched
    cdef float[::1] unpen_scores = np.empty_like(curr_scores)
    cdef int[::1] unpen_tb = np.empty_like(curr_tb)
    banded_forward_vit_step(
        unpen_scores,
        unpen_tb,
        prev_scores,
        curr_level,
        curr_signal,
        band_start_diff,
        dwell_penalty,
    )

    # loop over signal positions within this base band
    for band_pos in range(curr_scores.shape[0]):
        # if past the end of the prev band stay until the end
        if (
            band_pos + band_start_diff - prev_scores.shape[0]
            >= dwell_penalty.shape[0]
        ):
            curr_scores[band_pos] = (
                curr_scores[band_pos - 1]
                + score(curr_level, curr_signal[band_pos])
            )
            curr_tb[band_pos] = curr_tb[band_pos - 1] + 1
            continue
        # set invalid score and traceback if no valid transitions
        curr_scores[band_pos] = (
            LARGE_SCORE + prev_scores[prev_scores.shape[0] - 1]
        )
        curr_tb[band_pos] = -1
        if band_pos == 0 and band_start_diff == 0:
            continue
        running_pos_score = 0
        for dwell_idx in range(dwell_penalty.shape[0]):
            # beginning of curr or prev band reached
            if dwell_idx > band_pos or (
                band_start_diff == 0 and band_pos == dwell_idx
            ):
                break
            # update running pos score back from band_pos
            running_pos_score += score(
                curr_level, curr_signal[band_pos - dwell_idx]
            )
            # position past the end of the prev band
            if (
                band_pos - dwell_idx - 1 + band_start_diff
                >= prev_scores.shape[0]
            ):
                continue
            # compute penalized score for this band position at this dwell
            pos_score = (
                prev_scores[band_pos - dwell_idx - 1 + band_start_diff]
                + running_pos_score
                + dwell_penalty[dwell_idx]
            )
            # if this is a new min for this band pos, then update score/tb
            if pos_score < curr_scores[band_pos]:
                curr_scores[band_pos] = pos_score
                curr_tb[band_pos] = dwell_idx
        # if possible compute unpenalized score with dwell greater than
        # dwell_penalty.shape[0] using unpen_scores/unpen_tb
        if band_pos >= dwell_penalty.shape[0]:
            pos_score = (
                unpen_scores[band_pos - dwell_penalty.shape[0]]
                + running_pos_score
            )
            if pos_score < curr_scores[band_pos]:
                curr_scores[band_pos] = pos_score
                curr_tb[band_pos] = (
                    unpen_tb[band_pos - dwell_penalty.shape[0]]
                    + dwell_penalty.shape[0]
                )


cdef void banded_forward_vit_step(
    float[::1] curr_scores,
    int[::1] curr_tb,
    const float[::1] prev_scores,
    const float curr_level,
    const float[::1] curr_signal,
    int band_start_diff,
    const float[::1] sdp,
):
    """Process one base using standard Viterbi path scoring with squared error
    between signal and levels.

    Args:
        curr_scores (1D float array): To be populated by this function. Elements
            will be forward scores at each position
        curr_tb (1D int array): To be populated by this function. Elements will
            be the number of signal points backwards until the first point
            assigned to this base
        prev_scores (const 1D float array): Forward scores for the previous base
        curr_level (float): Expected signal level for the current base
        curr_signal (const 1D float array): Signal values for the current
            base's band
        band_start_diff (int): Difference in starting coordinates between
            current and previous base
        sdp (unused): Unused parameter for this core method
    """
    cdef int band_pos
    cdef float base_score, move_score, stay_score
    # compute start position in band
    if band_start_diff == 0:
        # if this is a "stay" band start, set invalid score and traceback
        curr_scores[0] = LARGE_SCORE + prev_scores[prev_scores.shape[0] - 1]
        curr_tb[0] = -1
    else:
        # else compute move score for start of base band
        base_score = score(curr_level, curr_signal[0])
        curr_scores[0] = prev_scores[band_start_diff - 1] + base_score
        curr_tb[0] = 0
        # clip prev_scores to start at same position as curr_scores
        prev_scores = prev_scores[band_start_diff:]
    # if base bands are the same
    if prev_scores.shape[0] == curr_scores.shape[0]:
        prev_scores = prev_scores[:prev_scores.shape[0] - 1]

    # compute scores where curr and prev base overlap
    for band_pos in range(1, prev_scores.shape[0] + 1):
        base_score = score(curr_level, curr_signal[band_pos])
        move_score = prev_scores[band_pos - 1] + base_score
        stay_score = curr_scores[band_pos - 1] + base_score
        if move_score < stay_score:
            curr_scores[band_pos] = move_score
            curr_tb[band_pos] = 0
        else:
            curr_scores[band_pos] = stay_score
            curr_tb[band_pos] = curr_tb[band_pos - 1] + 1

    # stay through rest of the band
    for band_pos in range(prev_scores.shape[0] + 1, curr_scores.shape[0]):
        base_score = score(curr_level, curr_signal[band_pos])
        stay_score = curr_scores[band_pos - 1] + base_score
        curr_scores[band_pos] = stay_score
        curr_tb[band_pos] = curr_tb[band_pos - 1] + 1


cdef void banded_forward_dp(
    float[::1] all_scores,
    int[::1] traceback,
    const float[::1] signal,
    const float[::1] levels,
    const int[:, ::1] seq_band,
    const int[::1] base_offsets,
    const float[::1] short_dwell_penalty,
    core_method,
):
    """Perform banded forward dynamic programming.

    Args:
        all_scores (1D int array): To be populated by this function. Ragged
            array representing each position within the band. Each element
            contains the forward score to that position within the band.
        traceback (1D int array): To be populated by this function. Ragged
            array representing each position within the band. Each element
            contains the number of signal points backwards until the first
            point assigned to this base.
        signal (1D float array): Normalized signal values.
        levels (1D float array): Expected signal levels for each base.
        seq_band (2D int array): Contains band boundaries for each base in
            signal coordinates.
        base_offsets (1D int array): Offset to the beginning of each base
        short_dwell_penalty (1D float array): Penalty values applied when
            core_method == "dwell_penalty".
        core_method (str): Core method to use for forward pass path scoring.
            "Viterbi" and "dwell_penalty" are implemented.
    """
    cdef core_func_ptr core_method_func
    if core_method == REFINE_ALGO_VIT_NAME:
        core_method_func = banded_forward_vit_step
    elif core_method == REFINE_ALGO_DWELL_PEN_NAME:
        core_method_func = banded_forward_dwell_penalty_step
    else:
        raise RemoraError(
            f"Invalid core signal mapping refine method: {core_method}"
        )

    cdef int sig_pos
    cdef int prev_bw, prev_offset, prev_band_st
    cdef int curr_bw, curr_offset, curr_band_st, curr_band_en

    # compute first base forward scores
    curr_bw = seq_band[1, 0]
    # spoof previous scores to force stays through first base
    prev_scores = np.full(curr_bw, HUGE_VALF, dtype=np.float32)
    prev_scores[0] = 0
    core_method_func(
        all_scores[:curr_bw],
        traceback[:curr_bw],
        prev_scores,
        levels[0],
        signal[:curr_bw],
        1,
        short_dwell_penalty,
    )
    prev_bw = curr_bw
    prev_band_st = prev_offset = 0

    # compute forward scores for all bases
    for base_idx in range(1, levels.shape[0]):
        curr_band_st = seq_band[0, base_idx]
        curr_band_en = seq_band[1, base_idx]
        curr_bw = curr_band_en - curr_band_st
        curr_offset = base_offsets[base_idx]
        # compute all scores and traceback for this base
        core_method_func(
            all_scores[curr_offset:curr_offset + curr_bw],
            traceback[curr_offset:curr_offset + curr_bw],
            all_scores[prev_offset:prev_offset + prev_bw],
            levels[base_idx],
            signal[curr_band_st:curr_band_en],
            curr_band_st - prev_band_st,
            short_dwell_penalty,
        )
        prev_band_st = curr_band_st
        prev_bw = curr_bw
        prev_offset = curr_offset


def seq_banded_dp(
    const float[::1] signal,
    const float[::1] levels,
    const int[:, ::1] seq_band,
    const float[::1] short_dwell_penalty,
    core_method=REFINE_ALGO_VIT_NAME,
):
    """Decode the path assignment between the input signal and levels.

    Note that paths are restricted to the input seq band.

    Args:
        signal (np.ndarray): Float32 array containing normalized signal values.
        levels (np.ndarray): Float32 array containing estimated levels.
        seq_band (np.ndarray): int32 np.ndarray with shape
            (2, seq_len = levels.shape[0]). The first row contains the lower
            band boundaries in signal coordinates and the second row contains
            the upper boundaries in signal coordinates. The first lower bound
            should be 0 and last upper bound should be signal.shape[0].
        short_dwell_penalty (np.ndarray): Float32 array with penalty values for
            short dwells. Length of the array defines the length of dwells that
            are penalized.
        core_method (str): Perform decoding with one of the following options:
            - "Viterbi": Standard Viterbi forward pass
            - "dwell_penalty": Viterbi forward pass using short_dwell_penalty

    Returns:
        Tuple containing:
            1)  Float32 1D numpy array representing each position within
                the band. Signal positions within each base are stored
                contiguously. Each element contains the forward score to that
                position within the band.
            2) Int32 np.ndarray of length = level.shape[0] + 1. The first value
               will be zero, and the remaining values are the start locations
               of each level within the signal array.
    """
    # Prepare for banded forwards pass followed by traceback
    base_offsets = np.empty(seq_band.shape[1] + 1, dtype=np.int32)
    base_offsets[0] = 0
    base_offsets[1:] = np.cumsum(np.diff(seq_band, axis=0)[0])
    band_len = base_offsets[base_offsets.shape[0] - 1]

    # all_scores array holds current score at every sequence by signal position
    # within the band
    all_scores = np.empty(band_len, dtype=np.float32)
    # traceback contains one value for each position in the band and contains
    # the number of stays from this position until the next move back to the
    # previous base/level.
    traceback = np.empty(band_len, dtype=np.int32)
    # Execute forward pass filling all_scores and traceback
    banded_forward_dp(
        all_scores,
        traceback,
        signal,
        levels,
        seq_band,
        base_offsets,
        short_dwell_penalty,
        core_method,
    )
    # path is primary return value as described in the return type
    path = np.empty(levels.shape[0] + 1, dtype=np.int32)
    # perform traceback and full path
    banded_traceback(path, seq_band, base_offsets, traceback)
    return all_scores, path, traceback, base_offsets


def forward_dp(
    const float[::1] signal,
    const float[::1] levels,
    const int[:, ::1] seq_band,
    const float[::1] short_dwell_penalty,
    core_method=REFINE_ALGO_VIT_NAME,
):
    base_offsets = np.empty(seq_band.shape[1] + 1, dtype=np.int32)
    base_offsets[0] = 0
    base_offsets[1:] = np.cumsum(np.diff(seq_band, axis=0)[0])
    band_len = base_offsets[base_offsets.shape[0] - 1]
    all_scores = np.empty(band_len, dtype=np.float32)
    traceback = np.empty(band_len, dtype=np.int32)
    banded_forward_dp(
        all_scores,
        traceback,
        signal,
        levels,
        seq_band,
        base_offsets,
        short_dwell_penalty,
        core_method,
    )
    return all_scores, traceback, base_offsets
