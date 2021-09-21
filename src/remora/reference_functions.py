import numpy as np
import bisect


class referenceEncoder:
    def __init__(
        self, focus_offset, chunk_context, fixed_seq_len_chunks, kmer_size=3
    ):

        self.alphabet = {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
            "N": [0, 0, 0, 0],
        }

        self.focus_offset = focus_offset
        self.chunk_context = chunk_context
        self.fixed_seq_len_chunks = fixed_seq_len_chunks
        self.kmer_size = kmer_size

    def reference_encoding(self, signals, references, base_locs):

        encoding = np.zeros(
            (self.kmer_size * (len(self.alphabet) - 1), base_locs[-1])
        )
        for i in range(len(base_locs) - 1):
            code = []
            for k in range(
                i - self.kmer_size // 2, i + self.kmer_size // 2 + 1
            ):
                if k < 0 or k >= len(ref):
                    code += self.alphabet["N"]
                else:
                    code += self.alphabet[references[k]]

            gap = base_locs[i + 1] - base_locs[i]
            encoding[:, base_locs[i] : base_locs[i + 1]] = np.tile(
                np.array(code), (gap, 1)
            ).T

        return encoding

    def reference_encoding_bybase(self, signals, references, base_locs):
        extracted_bl = base_locs[
            self.focus_offset
            - self.chunk_context[0] : self.focus_offset
            + self.chunk_context[1]
            + 1
        ]
        extracted_ref = references[
            self.focus_offset
            - self.chunk_context[0] : self.focus_offset
            + self.chunk_context[1]
            + 1
        ]
        encoding = np.zeros(
            (self.kmer_size * (len(self.alphabet) - 1), len(signals))
        )
        for i in range(len(extracted_bl) - 1):
            code = []
            for k in range(
                i - self.kmer_size // 2, i + self.kmer_size // 2 + 1
            ):
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

    def reference_encoding_bychunk(self, signals, references, base_locs):

        index_below = (
            bisect.bisect(
                base_locs,
                base_locs[self.focus_offset] - self.chunk_context[0],
            )
            - 1
        )
        index_above = (
            bisect.bisect(
                base_locs,
                base_locs[self.focus_offset] + self.chunk_context[1],
            )
            - 1
        )
        encoding = np.zeros(
            (self.kmer_size * (len(self.alphabet) - 1), len(signals))
        )
        origin = base_locs[self.focus_offset] - self.chunk_context[0]

        counter = 0

        while True:
            if counter == 0:
                curr = origin
            else:
                curr = next

            next = base_locs[np.argmax(base_locs > curr)]
            if next > base_locs[self.focus_offset] + self.chunk_context[1]:
                next = base_locs[self.focus_offset] + self.chunk_context[1]

            gap = next - curr

            code = []
            for k in range(
                index_below - self.kmer_size // 2,
                index_below + self.kmer_size // 2 + 1,
            ):
                if k < 0 or k >= len(references):
                    code += self.alphabet["N"]
                else:
                    code += self.alphabet[references[k]]

            encoding[:, curr - origin : next - origin] = np.tile(
                np.array(code), (gap, 1)
            ).T
            index_below += 1
            counter += 1

            if next > base_locs[index_above] or gap == 0:
                break
        return encoding

    def get_reference_encoding(self, signals, references, base_locs):

        if self.fixed_seq_len_chunks:
            encodings = self.reference_encoding_bybase(
                signals, references, base_locs
            )

        else:
            encodings = self.reference_encoding_bychunk(
                signals, references, base_locs
            )

        return encodings
