import argparse
from collections import defaultdict

import pysam
import numpy as np
from tqdm import tqdm

from remora.util import format_mm_ml_tags
from remora.inference import mods_tags_to_str


def collapse_label(read, alphabet, valid_indices, new_alphabet):
    q_mod_probs = defaultdict(dict)
    for (
        _,
        mod_strand,
        mod_name,
    ), mod_values in read.modified_bases_forward.items():
        assert mod_strand == 0, "Duplex mods not supported"
        for pos, prob in mod_values:
            q_mod_probs[pos][mod_name] = (prob + 0.5) / 256
    q_mod_probs_collapse = {}
    for q_pos, pos_probs in q_mod_probs.items():
        pos_probs = np.array(
            [1 - sum(pos_probs.values())]
            + [pos_probs.get(mod_name, 0) for mod_name in alphabet[1:]]
        )
        # re-normalize and remove can prob
        q_mod_probs_collapse[q_pos] = (
            pos_probs[valid_indices] / np.sum(pos_probs[valid_indices])
        )[1:]
    probs = np.stack([pos_probs for pos_probs in q_mod_probs_collapse.values()])
    mod_tags = mods_tags_to_str(
        format_mm_ml_tags(
            seq=read.get_forward_sequence(),
            poss=q_mod_probs_collapse.keys(),
            probs=probs,
            mod_bases=new_alphabet[1:],
            can_base=alphabet[0],
        )
    )
    read = read.to_dict()
    read["tags"] = [
        tag
        for tag in read["tags"]
        if not (tag.startswith("MM") or tag.startswith("ML"))
    ]
    read["tags"].extend(mod_tags)
    return read


def main(args):
    new_alphabet = "".join(
        [b for b in args.mod_alphabet if b not in args.labels_to_remove]
    )
    valid_indices = np.array(
        [args.mod_alphabet.index(b) for b in new_alphabet], dtype=int
    )
    pysam.set_verbosity(0)
    in_bam = pysam.AlignmentFile(args.in_bam, "rb")
    out_bam = pysam.AlignmentFile(args.out_bam, "wb", template=in_bam)
    for read in tqdm(in_bam, smoothing=0):
        read = collapse_label(
            read, args.mod_alphabet, valid_indices, new_alphabet
        )
        out_bam.write(pysam.AlignedSegment.from_dict(read, out_bam.header))
    in_bam.close()
    out_bam.close()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_bam")
    parser.add_argument("out_bam")
    parser.add_argument("--mod-alphabet", default="Chm")
    parser.add_argument("--labels-to-remove", default="h")
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
