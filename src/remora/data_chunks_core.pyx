#cython: language_level=3

import numpy as np

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def trim_sb_chunk_context_core(
    int stored_cc_before,
    int stored_cc_after,
    int cc_before,
    int cc_after,
    int total_seq_context,
    signed char[:, ::1] seqs,
    short[:, ::1] seq_mappings,
    short[::1] seq_lens,
):
    cdef cc_width = cc_before + cc_after
    cdef int chunk_idx
    cdef short seq_len, st_clip
    # determine sequence clipping coords from seq_to_sig_map array
    if stored_cc_before - cc_before > 0:
        for chunk_idx in range(seq_lens.size):
            st_clip = 0
            while seq_mappings[chunk_idx, st_clip + 1] <= 0:
                st_clip += 1
            seq_len = seq_lens[chunk_idx]
            seq_mappings[chunk_idx][: seq_len + 1 - st_clip] = seq_mappings[
                chunk_idx][st_clip : seq_len + 1]
            seqs[chunk_idx][: seq_len + total_seq_context - st_clip] = seqs[
                chunk_idx][st_clip : seq_len + total_seq_context]
            seq_lens[chunk_idx] = seq_len - st_clip
            seq_mappings[chunk_idx, 0] = 0
    if stored_cc_after != cc_after:
        for chunk_idx in range(seq_lens.size):
            while seq_mappings[chunk_idx, seq_lens[chunk_idx] - 1] >= cc_width:
                seq_lens[chunk_idx] -= 1
            seq_mappings[chunk_idx, seq_lens[chunk_idx]] = cc_width
