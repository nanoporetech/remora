import os
import json
import atexit
from collections import defaultdict, namedtuple

import torch
import pysam
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from remora.io import parse_mods_bed
from remora.util import softmax_axis1, revcomp
from remora import RemoraError, constants, encoded_kmers, log


LOGGER = log.get_logger()
VAL_METRICS = namedtuple(
    "VAL_METRICS",
    (
        "loss",
        "acc",
        "num_calls",
        "conf_mat",
        "filt_frac",
        "filt_acc",
        "filt_conf_mat",
        "filt_thresh",
    ),
)


def mat_to_str(mat):
    return json.dumps(mat.tolist(), separators=(",", ":"))


###################
# Core Validation #
###################


def compute_metrics(probs, labels, filt_frac):
    pred_labels = np.argmax(probs, axis=1)
    conf_mat = confusion_matrix(labels, pred_labels)
    correctly_labeled = pred_labels == labels
    acc = correctly_labeled.sum() / labels.size

    # extract probability of predicted class for each observation
    # numpy advanced indexing is weird but this is the right way to do this
    pred_probs = np.take_along_axis(
        probs, np.expand_dims(pred_labels, -1), -1
    ).squeeze(-1)
    filt_thr = np.quantile(pred_probs, filt_frac)
    # if all values would be filtered make filter threshold slightly smaller
    if filt_thr == pred_probs.max():
        filt_thr *= 0.999999
    conf_chunks = pred_probs > filt_thr
    filt_labels = labels[conf_chunks]
    filt_acc = correctly_labeled[conf_chunks].sum() / filt_labels.size
    filt_conf_mat = confusion_matrix(filt_labels, pred_labels[conf_chunks])
    filt_frac = 1 - (filt_labels.size / labels.size)

    return acc, conf_mat, filt_frac, filt_acc, filt_conf_mat, filt_thr


def add_unmodeled_labels(output, unmodeled_labels):
    """Add unmodeled labels into the neural network output for validation.

    Args:
        output (np.array): Output from a Remora neural network.
            Shape: (batch, modeled_labels)
        unmodeled_labels (np.array): Indices of unmodled labels in desired
            output

    For example, with a dataset containing C, 5hmC and 5mC labels (Chm), but
    validated using a model only predicting C vs 5mC (Cm), unmodeled_labels
    would be `[1]` and the return array would contain large negative values in
    the `1` index of the second axis of the return array. In this example the
    input array would have shape (batch, 2) and the returned array would have
    shape (batch, 3))
    """
    if unmodeled_labels.size == 0:
        return output
    nobs, nlab = output.shape
    n_new_lab = nlab + unmodeled_labels.size
    # fill with large negative number since this is the model output which
    # will be softmax-ed to get probabilities
    new_output = np.full((nobs, n_new_lab), -1000, dtype=output.dtype)
    new_output[:, 0] = output[:, 0]
    unused_idx = 0
    for idx in range(1, n_new_lab):
        if idx in unmodeled_labels:
            unused_idx += 1
            continue
        new_output[:, idx] = output[:, idx - unused_idx]
    return new_output


def _validate_model(
    model,
    model_mod_bases,
    criterion,
    dataset,
    filt_frac=constants.DEFAULT_FILT_FRAC,
    full_results_fh=None,
    disable_pbar=False,
):
    device = next(model.parameters()).device
    unmodeled_labels = np.array(
        [
            idx + 1
            for idx, mb in enumerate(dataset.mod_bases)
            if mb not in model_mod_bases
        ]
    )
    is_torch_model = isinstance(model, nn.Module)
    if is_torch_model:
        model.eval()
        torch.set_grad_enabled(False)

    bb, ab = dataset.kmer_context_bases
    all_labels = []
    all_outputs = []
    all_loss = []

    if os.environ.get("LOG_SAFE", False):
        disable_pbar = True
    for (
        (sigs, seqs, seq_maps, seq_lens),
        labels,
        (read_ids, read_focus_bases),
    ) in tqdm(
        dataset,
        smoothing=0,
        desc="Batches",
        disable=disable_pbar,
    ):
        all_labels.append(labels)
        enc_kmers = encoded_kmers.compute_encoded_kmer_batch(
            bb, ab, seqs, seq_maps, seq_lens
        )
        if is_torch_model:
            sigs = torch.from_numpy(sigs).to(device)
            enc_kmers = torch.from_numpy(enc_kmers).to(device)
            output = model(sigs, enc_kmers).detach().cpu().numpy()
        else:
            output = model.run([], {"sig": sigs, "seq": enc_kmers})[0]
        output = add_unmodeled_labels(output, unmodeled_labels)
        all_outputs.append(output)
        all_loss.append(
            criterion(torch.from_numpy(output), torch.from_numpy(labels))
            .detach()
            .cpu()
            .numpy()
        )
        if full_results_fh is not None:
            full_results_fh.write_results(
                output, labels, read_ids, read_focus_bases
            )
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels)
    if is_torch_model:
        torch.set_grad_enabled(True)
    all_probs = softmax_axis1(all_outputs)
    (
        acc,
        conf_mat,
        filt_frac,
        filt_acc,
        filt_conf_mat,
        filt_thr,
    ) = compute_metrics(all_probs, all_labels, filt_frac)
    return VAL_METRICS(
        loss=np.mean(all_loss),
        acc=acc,
        num_calls=all_labels.size,
        conf_mat=conf_mat,
        filt_frac=filt_frac,
        filt_acc=filt_acc,
        filt_conf_mat=filt_conf_mat,
        filt_thresh=filt_thr,
    )


def process_mods_probs(probs, labels, allow_unbalanced, pct_filt, name):
    if not allow_unbalanced:
        nlabs = labels.max() + 1
        # split probs
        label_probs = [probs[labels == mod_idx] for mod_idx in range(nlabs)]
        lab_sizes = [lp.shape[0] for lp in label_probs]
        if len(lab_sizes) == 1:
            raise RemoraError(
                "Cannot balance dataset with 1 label. "
                "Consider running with `--allow-unbalanced`"
            )
        LOGGER.debug(f"Balancing labels. Starting from: {lab_sizes}")
        min_size = min(lab_sizes)
        probs = np.empty((min_size * nlabs, nlabs), dtype=probs.dtype)
        labels = np.empty((min_size * nlabs), dtype=labels.dtype)
        for lab_idx, label_probs in enumerate(label_probs):
            if label_probs.shape[0] > min_size:
                np.random.shuffle(label_probs)
            probs[lab_idx * min_size : (lab_idx + 1) * min_size] = label_probs[
                :min_size
            ]
            labels[lab_idx * min_size : (lab_idx + 1) * min_size] = lab_idx

    (
        acc,
        conf_mat,
        filt_frac,
        filt_acc,
        filt_conf_mat,
        filt_thr,
    ) = compute_metrics(probs, labels, pct_filt / 100)
    ms = VAL_METRICS(
        loss=np.NAN,
        acc=acc,
        num_calls=labels.size,
        conf_mat=conf_mat,
        filt_frac=filt_frac,
        filt_acc=filt_acc,
        filt_conf_mat=filt_conf_mat,
        filt_thresh=filt_thr,
    )
    val_output = (
        f"\n{ValidationLogger.HEADER}\n"
        f"{name}\t0\t0\t"
        f"{ms.acc:.6f}\t{mat_to_str(ms.conf_mat)}\t"
        f"NAN\t{ms.num_calls}\t{ms.filt_frac:.4f}\t"
        f"{ms.filt_acc:.6f}\t{mat_to_str(ms.filt_conf_mat)}\t{ms.filt_thresh}\n"
    )
    LOGGER.info(val_output)


class ResultsWriter:
    def __init__(self, out_fh):
        self.sep = "\t"
        self.out_fh = out_fh
        df = pd.DataFrame(
            columns=[
                "read_id",
                "read_focus_base",
                "label",
                "class_pred",
                "class_probs",
            ]
        )
        df.to_csv(self.out_fh, sep=self.sep, index=False)

    def write_results(self, output, labels, read_ids, read_focus_bases):
        class_preds = output.argmax(axis=1)
        str_probs = [",".join(map(str, r)) for r in softmax_axis1(output)]
        pd.DataFrame(
            {
                "read_id": read_ids,
                "read_focus_base": read_focus_bases,
                "label": labels,
                "class_pred": class_preds,
                "class_probs": str_probs,
            }
        ).to_csv(self.out_fh, header=False, index=False, sep=self.sep)


class ValidationLogger:
    HEADER = "\t".join(
        (
            "Val_Type",
            "Epoch",
            "Iteration",
            "Accuracy",
            "Confusion_Matrix",
            "Loss",
            "Num_Calls",
            "Filtered_Fraction",
            "Filtered_Accuracy",
            "Filtered_Confusion_Matrix",
            "Filtered_Threshold",
        )
    )

    def __init__(self, fp, full_results_fh=None):
        self.fp = fp
        self.fp.write(self.HEADER + "\n")
        if full_results_fh is None:
            self.full_results_fh = None
        else:
            self.full_results_fh = ResultsWriter(full_results_fh)

    def validate_model(
        self,
        model,
        model_mod_bases,
        criterion,
        dataset,
        filt_frac=constants.DEFAULT_FILT_FRAC,
        val_type="val",
        nepoch=0,
        niter=0,
        disable_pbar=False,
    ):
        ms = _validate_model(
            model,
            model_mod_bases,
            criterion,
            dataset,
            filt_frac,
            full_results_fh=self.full_results_fh,
            disable_pbar=disable_pbar,
        )
        self.fp.write(
            f"{val_type}\t{nepoch}\t{niter}\t"
            f"{ms.acc:.6f}\t{mat_to_str(ms.conf_mat)}\t"
            f"{ms.loss:.6f}\t{ms.num_calls}\t{ms.filt_frac:.4f}\t"
            f"{ms.filt_acc:.6f}\t{mat_to_str(ms.filt_conf_mat)}\t"
            f"{ms.filt_thresh}\n"
        )
        return ms


##################
# ModBAM Parsing #
##################


def parse_mod_read(
    read,
    gt_sites,
    gt_ranges,
    alphabet,
    full_fh,
    nctx=5,
    max_sites=None,
):
    strand = "-" if read.is_reverse else "+"
    ctg_gt = gt_sites.get((read.reference_name, strand))
    ctg_gt_range = gt_ranges.get((read.reference_name, strand))

    try:
        aligned_pairs = read.get_aligned_pairs(with_seq=True)
    except ValueError:
        LOGGER.debug(f"Read missing MD tag {read.query_name}")
        return [], []
    r_align = "".join([b.upper() if b else "-" for _, _, b in aligned_pairs])
    q_align = "".join(
        [
            read.query_sequence[q_pos] if q_pos else "-"
            for q_pos, _, _ in aligned_pairs
        ]
    )

    # note read.modified_bases stores positions in forward reference strand
    # query sequence coordinates
    # TODO handle duplex mods on opposite strand
    q_mod_probs = defaultdict(dict)
    for (_, mod_strand, mod_name), mod_values in read.modified_bases.items():
        if (
            (mod_strand == 0 and read.is_reverse)
            or (mod_strand == 1 and not read.is_reverse)
            or mod_name not in alphabet
        ):
            continue
        for pos, prob in mod_values:
            q_mod_probs[pos][mod_name] = (prob + 0.5) / 256
    q_mod_probs_full = {}
    for q_pos, pos_probs in q_mod_probs.items():
        # TODO handle case with invalid probs and set can_prob to 0
        # and mod probs to softmax values
        # create array of probs in fixed order and fill 0 probs
        q_mod_probs_full[q_pos] = np.array(
            [1 - sum(pos_probs.values())]
            + [pos_probs.get(mod_name, 0) for mod_name in alphabet[1:]]
        )

    # loop over aligned pairs extracting ground truth and/or read modified bases
    # only store sites with ground truth value and modified base stats, but
    # record sites with either in full output.
    probs, labels = [], []
    prev_q_pos = prev_r_pos = None
    for a_idx, (q_pos, r_pos, _) in enumerate(aligned_pairs):
        # record last valid positions when determining "within" values over
        # indels for extended output
        if q_pos is not None:
            prev_q_pos = q_pos
        if r_pos is not None:
            prev_r_pos = r_pos
        r_pos_mod = None if ctg_gt is None else ctg_gt.get(r_pos)
        q_pos_mod_probs = q_mod_probs_full.get(q_pos)
        # if neither the basecalls contain modified base probabilities nor the
        # ground truth contains modified base information, skip this position.
        if r_pos_mod is None and q_pos_mod_probs is None:
            continue
        r_pos_mod_idx = None if r_pos_mod is None else alphabet.index(r_pos_mod)
        if full_fh is not None:
            q_pos_mod_probs_str = (
                None
                if q_pos_mod_probs is None
                else ",".join(map(str, q_pos_mod_probs))
            )

            if a_idx < nctx:
                r_pos_align = "-" * (a_idx - nctx) + r_align[: a_idx + nctx + 1]
                q_pos_align = "-" * (a_idx - nctx) + q_align[: a_idx + nctx + 1]
            else:
                r_pos_align = r_align[a_idx - nctx : a_idx + nctx + 1]
                q_pos_align = q_align[a_idx - nctx : a_idx + nctx + 1]
            r_pos_align = r_pos_align.rjust(nctx * 2 + 1, "-")
            q_pos_align = q_pos_align.rjust(nctx * 2 + 1, "-")
            if read.is_reverse:
                r_pos_align = revcomp(r_pos_align)
                q_pos_align = revcomp(q_pos_align)
            within_align = within_gt = False
            if prev_q_pos is not None:
                within_align = (
                    read.query_alignment_start
                    <= prev_q_pos
                    < read.query_alignment_end
                )
            if ctg_gt_range is not None and prev_r_pos is not None:
                within_gt = within_align and (
                    ctg_gt_range[0] <= prev_r_pos <= ctg_gt_range[1]
                )
            full_fh.write(
                f"{read.query_name}\t{q_pos}\t{read.reference_name}\t{r_pos}\t"
                f"{strand}\t{r_pos_mod_idx}\t{q_pos_mod_probs_str}\t"
                f"{r_pos_align}\t{q_pos_align}\t{within_align}\t{within_gt}\n"
            )

        if r_pos_mod is not None and q_pos_mod_probs is not None:
            labels.append(r_pos_mod_idx)
            probs.append(q_pos_mod_probs)
    if max_sites is not None and len(labels) > max_sites:
        indices = np.random.choice(
            len(labels),
            size=max_sites,
            replace=False,
        )
        labels = [labels[idx] for idx in indices]
        probs = [probs[idx] for idx in indices]
    return probs, labels


def check_mod_strand(read, bam_path, alphabet, do_warn_mod, do_warn_strand):
    if read.modified_bases is None:
        LOGGER.debug(f"No modified bases found in {read.query_name}")
        return do_warn_mod, do_warn_strand, False

    valid_mods = False
    for _, mod_strand, mod_name in read.modified_bases.keys():
        if (mod_strand == 0 and read.is_reverse) or (
            mod_strand == 1 and not read.is_reverse
        ):
            LOGGER.debug(
                f"Invalid mod strand {mod_strand} {read.query_name} {bam_path}"
            )
            if do_warn_strand:
                LOGGER.warning("Reverse strand (duplex) mods not supported ")
                do_warn_strand = False
            continue
        if mod_name not in alphabet:
            LOGGER.debug(
                f"Modified base found in BAM ({mod_name}) not found in "
                f"ground truth {read.query_name}"
            )
            if do_warn_mod:
                LOGGER.warning(
                    f"Modified base found in BAM ({mod_name}) not found in "
                    "ground truth. If this should be included in validation, "
                    "add with --extra-bases."
                )
                do_warn_mod = False
            continue
        valid_mods = True
    return do_warn_mod, do_warn_strand, valid_mods


def parse_mod_bam(
    bam_path,
    gt_sites,
    gt_ranges,
    alphabet,
    full_fh,
    context_bases=5,
    max_sites=None,
):
    """Parse modified base tags from BAM file recording probability of canonical
    and each mod at each site in ground truth sites.

    Arsg:
        bam_path (str): Path to mapped BAM file with modified base tags
        gt_sites (dict): First level keys are chromosome and strand 2-tuples,
            second level keys are reference positions pointing to a ground truth
            modified base single letter code.
        gt_ranges (dict): Min and max values from gt_sites dict values
        alphabet (str): Canonical base followed by modified bases found in
            ground truth data. Other modified bases in BAM file will be ignored.
        full_fh (File): File handle to write full results.
        context_bases (int): Number of context bases to include in full output
        max_sites (int): Max sites to extract per read

    Returns:
        2-tuple containing
            - Numpy array with shape (num_calls, num_mods + 1) containing
                probabilities at each valid site
            - Numpy array with shape (num_calls) containing ground truth labels
    """
    probs, labels = [], []
    # hid warnings for no index when using unmapped or unsorted files
    pysam_save = pysam.set_verbosity(0)
    do_warn_mod = do_warn_strand = True
    with pysam.AlignmentFile(bam_path, check_sq=False) as bam_fh:
        for read in tqdm(bam_fh, smoothing=0):
            do_warn_mod, do_warn_strand, valid_mods = check_mod_strand(
                read, bam_path, alphabet, do_warn_mod, do_warn_strand
            )
            if not valid_mods:
                continue
            read_probs, read_labels = parse_mod_read(
                read,
                gt_sites,
                gt_ranges,
                alphabet,
                full_fh,
                nctx=context_bases,
                max_sites=max_sites,
            )
            probs.extend(read_probs)
            labels.extend(read_labels)
    pysam.set_verbosity(pysam_save)
    if len(probs) < 1:
        raise RemoraError(
            f"No valid modification calls from {bam_path}. Confirm that "
            "contig names from reference FASTA and ground truth BED match."
        )
    LOGGER.debug(
        f"Parsed {len(probs)} modified base calls from file: {bam_path}"
    )
    return np.array(probs), np.array(labels)


def validate_modbams(
    bams_and_beds,
    full_results_path,
    name,
    pct_filt,
    allow_unbalanced=False,
    seed=None,
    extra_bases=None,
    max_sites_per_read=None,
):
    # seed for random balancing
    seed = (
        np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32)
        if seed is None
        else seed
    )
    LOGGER.debug(f"Seed selected is {seed}")

    full_fh = None
    if full_results_path is not None:
        full_fh = open(full_results_path, "w", buffering=512)
        atexit.register(full_fh.close)
        full_fh.write(
            "query_name\tquery_pos\tref_name\tref_pos\tstrand\t"
            "gt_mod_idx\tmod_probs\tref_align\tquery_align\t"
            "within_align\twithin_gt\n"
        )

    LOGGER.info("Parsing ground truth BED files")
    bams, beds = zip(*bams_and_beds)
    parsed_gt_sites = {}
    all_gt_sites = []
    all_gt_ranges = []
    all_mods = set()
    for bed_path in beds:
        try:
            gt_sites, samp_mods = parsed_gt_sites[bed_path]
        except KeyError:
            gt_sites, samp_mods = parse_mods_bed(bed_path)
            parsed_gt_sites[bed_path] = (gt_sites, samp_mods)
            tot_sites = sum(len(cs_sites) for cs_sites in gt_sites.values())
            LOGGER.info(
                f"Parsed {tot_sites} total sites with labels {samp_mods} "
                f"from {bed_path}"
            )
        all_gt_sites.append(gt_sites)
        all_gt_ranges.append(
            dict((cs, (min(poss), max(poss))) for cs, poss in gt_sites.items())
        )
        all_mods.update(samp_mods)
    if extra_bases is not None:
        all_mods.update(extra_bases)
    can_base = all_mods.intersection("ACGTU")
    if len(can_base) > 1:
        raise RemoraError(f"More than one canonical base found: {can_base}")
    if len(can_base) == 0:
        raise RemoraError("No canonical bases found in ground truth.")
    mod_bases = all_mods.difference("ACGTU")
    alphabet = "".join(can_base) + "".join(sorted(mod_bases))

    LOGGER.info("Parsing modBAM files")
    all_probs, all_labels = [], []
    for bam_path, gt_sites, gt_ranges in zip(bams, all_gt_sites, all_gt_ranges):
        probs, labels = parse_mod_bam(
            bam_path,
            gt_sites,
            gt_ranges,
            alphabet,
            full_fh,
            max_sites=max_sites_per_read,
        )
        all_probs.append(probs)
        all_labels.append(labels)

    LOGGER.info(f"Alphabet used (and order of reported metrics): {alphabet}")
    process_mods_probs(
        np.vstack(all_probs),
        np.concatenate(all_labels),
        allow_unbalanced,
        pct_filt,
        name,
    )
