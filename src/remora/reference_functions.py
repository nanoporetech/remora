import numpy as np
import bisect


class referenceEncoder:
    def __init__(self, focus_offset, chunk_context, fixed_seq_len_chunks):

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

    def reference_encoding(self, signals, references, base_locs, kmer_size=3):

        encodings = []
        for sig, ref, bl in zip(signals, references, base_locs):
            encoding = np.zeros((kmer_size * (len(self.alphabet) - 1), bl[-1]))
            for i in range(len(bl) - 1):
                code = []
                for k in range(i - kmer_size // 2, i + kmer_size // 2 + 1):
                    if k < 0 or k >= len(ref):
                        code += self.alphabet["N"]
                    else:
                        code += self.alphabet[ref[k]]

                gap = bl[i + 1] - bl[i]
                encoding[:, bl[i] : bl[i + 1]] = np.tile(
                    np.array(code), (gap, 1)
                ).T
            encodings.append(encoding)

        return encodings

    def reference_encoding_bybase(
        self, signals, references, base_locs, kmer_size=3
    ):
        encodings = []
        for sig, ref, bl in zip(signals, references, base_locs):
            if len(sig) == 0:
                continue
            extracted_bl = bl[
                self.focus_offset
                - self.chunk_context[0] : self.focus_offset
                + self.chunk_context[1]
                + 1
            ]
            extracted_ref = ref[
                self.focus_offset
                - self.chunk_context[0] : self.focus_offset
                + self.chunk_context[1]
                + 1
            ]
            encoding = np.zeros(
                (kmer_size * (len(self.alphabet) - 1), len(sig))
            )
            for i in range(len(extracted_bl) - 1):
                code = []
                for k in range(i - kmer_size // 2, i + kmer_size // 2 + 1):
                    if k < 0 or k >= len(ref):
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
            encodings.append(encoding)

        return encodings

    def reference_encoding_bychunk(
        self, signals, references, base_locs, kmer_size=3
    ):
        encodings = []
        for sig, ref, bl in zip(signals, references, base_locs):
            index_below = (
                bisect.bisect(
                    bl,
                    bl[self.focus_offset] - self.chunk_context[0],
                )
                - 1
            )
            index_above = (
                bisect.bisect(
                    bl,
                    bl[self.focus_offset] + self.chunk_context[1],
                )
                - 1
            )
            encoding = np.zeros(
                (kmer_size * (len(self.alphabet) - 1), len(sig))
            )
            origin = bl[self.focus_offset] - self.chunk_context[0]

            counter = 0

            while True:
                if counter == 0:
                    curr = origin
                else:
                    curr = next
                next = bl[np.argmax(bl > curr)]
                if next > bl[self.focus_offset] + self.chunk_context[1]:
                    next = bl[self.focus_offset] + self.chunk_context[1]

                gap = next - curr

                code = []
                for k in range(
                    index_below - kmer_size // 2,
                    index_below + kmer_size // 2 + 1,
                ):
                    if k < 0 or k >= len(ref):
                        code += self.alphabet["N"]
                    else:
                        code += self.alphabet[ref[k]]

                encoding[:, curr - origin : next - origin] = np.tile(
                    np.array(code), (gap, 1)
                ).T

                index_below += 1
                counter += 1

                if next > bl[index_above] or gap == 0:
                    break
            encodings.append(encoding)
        return encodings

    def get_reference_encoding(
        self, signals, references, base_locs, kmer_size=3
    ):

        if self.fixed_seq_len_chunks:
            encodings = self.reference_encoding_bybase(
                signals, references, base_locs, kmer_size
            )

        else:
            encodings = self.reference_encoding_bychunk(
                signals, references, base_locs, kmer_size
            )

        return encodings
