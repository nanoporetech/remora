#cython: language_level=3

import numpy as np

cimport cython

cdef int ENCODING_LEN = 4


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_encoded_kmer_batch(
    int before_context_bases,
    int after_context_bases,
    const signed char[:, ::1] seqs,
    const short[:, ::1] seq_mappings,
    const short[::1] seq_lens,
):
    cdef int nchunks = seq_lens.shape[0]

    # initialize output array
    cdef int sig_len = seq_mappings[0, seq_lens[0]]
    cdef int kmer_len = before_context_bases + after_context_bases + 1
    cdef int enc_kmer_len = ENCODING_LEN * kmer_len
    out_arr = np.zeros((seq_lens.shape[0], enc_kmer_len, sig_len), np.float32)
    cdef float[:, :, ::1] out_mv = out_arr

    # loop over chunks, kmer_pos and mappings to fill output array
    cdef int chunk_idx, seq_len, kmer_pos, enc_offset
    cdef int seq_pos, base, base_st, base_en, sig_pos
    for chunk_idx in range(nchunks):
        seq_len = seq_lens[chunk_idx]
        for kmer_pos in range(kmer_len):
            enc_offset = ENCODING_LEN * kmer_pos
            for seq_pos in range(seq_len):
                base = seqs[chunk_idx, seq_pos + kmer_pos]
                if base == -1:
                    continue
                base_st = seq_mappings[chunk_idx, seq_pos]
                base_en = seq_mappings[chunk_idx, seq_pos + 1]
                for sig_pos in range(base_st, base_en):
                    out_mv[chunk_idx, enc_offset + base, sig_pos] = 1.0
    return out_arr
