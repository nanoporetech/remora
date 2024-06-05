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
    short[::1] seq_clip_coords,
):
    cdef short cc_width = cc_before + cc_after
    cdef int chunk_idx, pos_idx
    cdef int num_chunks = seq_lens.size
    cdef short seq_len, st_clip
    cdef short[::1] csm
    cdef signed char[::1] cs
    # determine sequence clipping coords from seq_to_sig_map array
    if stored_cc_before > cc_before:
        for chunk_idx in range(num_chunks):
            st_clip = 0
            while seq_mappings[chunk_idx, st_clip + 1] <= 0:
                st_clip += 1
            seq_len = seq_lens[chunk_idx]
            csm = seq_mappings[chunk_idx]
            for pos_idx in range(seq_len + 1 - st_clip):
                csm[pos_idx] = csm[st_clip + pos_idx]
            seq_clip_coords[chunk_idx] = st_clip
            cs = seqs[chunk_idx]
            for pos_idx in range(seq_len + total_seq_context - st_clip):
                cs[pos_idx] = cs[pos_idx + st_clip]
            seq_lens[chunk_idx] = seq_len - st_clip
            seq_mappings[chunk_idx, 0] = 0
    if stored_cc_after > cc_after:
        for chunk_idx in range(num_chunks):
            while seq_mappings[chunk_idx, seq_lens[chunk_idx] - 1] >= cc_width:
                seq_lens[chunk_idx] -= 1
            seq_mappings[chunk_idx, seq_lens[chunk_idx]] = cc_width
