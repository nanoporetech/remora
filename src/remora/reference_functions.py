import numpy as np
import bisect

from remora import constants


class referenceEncoder:
    def __init__(
        self,
        chunk_context,
        fixed_seq_len_chunks,
        context_bases=constants.DEFAULT_CONTEXT_BASES,
    ):
        self.alphabet = {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
            "N": [0, 0, 0, 0],
        }

        self.chunk_context = chunk_context
        self.fixed_seq_len_chunks = fixed_seq_len_chunks
        self.before_bases, self.after_bases = context_bases
        self.kmer_len = sum(context_bases) + 1

    def reference_encoding_by_base(
        self, sig_len, references, base_locs, focus_offset
    ):
        extracted_bl = base_locs[
            focus_offset
            - self.chunk_context[0] : focus_offset
            + self.chunk_context[1]
            + 1
        ]
        extracted_ref = references[
            focus_offset
            - self.chunk_context[0] : focus_offset
            + self.chunk_context[1]
            + 1
        ]
        encoding = np.zeros((self.kmer_len * (len(self.alphabet) - 1), sig_len))
        for i in range(len(extracted_bl) - 1):
            code = []
            for k in range(i - self.before_bases, i + self.after_bases + 1):
                if k < 0 or k >= len(references):
                    code += self.alphabet["N"]
                else:
                    code += self.alphabet[extracted_ref[k]]

            gap = extracted_bl[i + 1] - extracted_bl[i]
            encoding[
                :,
                extracted_bl[i]
                - extracted_bl[0] : extracted_bl[i + 1]
                - extracted_bl[0],
            ] = np.tile(np.array(code), (gap, 1)).T

        return encoding

    def reference_encoding_by_chunk(
        self, sig_len, references, base_locs, focus_offset
    ):

        index_below = (
            bisect.bisect(
                base_locs,
                base_locs[focus_offset] - self.chunk_context[0],
            )
            - 1
        )
        index_above = (
            bisect.bisect(
                base_locs,
                base_locs[focus_offset] + self.chunk_context[1],
            )
            - 1
        )
        encoding = np.zeros((self.kmer_len * (len(self.alphabet) - 1), sig_len))
        origin = base_locs[focus_offset] - self.chunk_context[0]

        prev = curr = origin
        while True:
            prev = curr
            curr = base_locs[np.argmax(base_locs > prev)]
            if curr > base_locs[focus_offset] + self.chunk_context[1]:
                curr = base_locs[focus_offset] + self.chunk_context[1]

            gap = curr - prev
            code = []
            for k in range(
                index_below - self.before_bases,
                index_below + self.after_bases + 1,
            ):
                if k < 0 or k >= len(references):
                    code += self.alphabet["N"]
                else:
                    code += self.alphabet[references[k]]

            encoding[:, prev - origin : curr - origin] = np.tile(
                np.array(code), (gap, 1)
            ).T
            index_below += 1

            if curr > base_locs[index_above] or gap == 0:
                break
        return encoding

    def get_reference_encoding(
        self, sig_len, references, base_locs, focus_offset
    ):

        if self.fixed_seq_len_chunks:
            encodings = self.reference_encoding_by_base(
                sig_len, references, base_locs, focus_offset
            )

        else:
            encodings = self.reference_encoding_by_chunk(
                sig_len, references, base_locs, focus_offset
            )

        return encodings
