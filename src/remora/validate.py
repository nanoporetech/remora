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
from remora.util import softmax_axis1
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
    conf_thr = np.quantile(pred_probs, filt_frac)
    conf_chunks = pred_probs > conf_thr
    filt_labels = labels[conf_chunks]
    filt_acc = correctly_labeled[conf_chunks].sum() / filt_labels.size
    filt_conf_mat = confusion_matrix(filt_labels, pred_labels[conf_chunks])
    filt_frac = 1 - (filt_labels.size / labels.size)

    return acc, conf_mat, filt_frac, filt_acc, filt_conf_mat


def get_label_coverter(from_labels, to_labels, base_pred):
    if base_pred:
        return np.arange(4)
    if not set(from_labels).issubset(to_labels):
        raise RemoraError(
            f"Cannot convert from superset of labels ({from_labels}) "
            f"to a subset ({to_labels})."
        )
    return np.array([0] + [to_labels.find(mb) + 1 for mb in from_labels])


def _validate_model(
    model,
    model_mod_bases,
    criterion,
    dataset,
    filt_frac=constants.DEFAULT_FILT_FRAC,
    display_progress_bar=True,
    full_results_fh=None,
):
    device = next(model.parameters()).device
    label_conv = get_label_coverter(
        dataset.mod_bases, model_mod_bases, dataset.base_pred
    )
    is_torch_model = isinstance(model, nn.Module)
    if is_torch_model:
        model.eval()
        torch.set_grad_enabled(False)

    bb, ab = dataset.kmer_context_bases
    all_labels = []
    all_outputs = []
    all_loss = []
    ds_iter = (
        tqdm(dataset, smoothing=0, desc="Batches", leave=False)
        if display_progress_bar
        else dataset
    )
    for (
        (sigs, seqs, seq_maps, seq_lens),
        labels,
        (read_ids, read_focus_bases),
    ) in ds_iter:
        model_labels = label_conv[labels]
        all_labels.append(model_labels)
        enc_kmers = encoded_kmers.compute_encoded_kmer_batch(
            bb, ab, seqs, seq_maps, seq_lens
        )
        if is_torch_model:
            sigs = torch.from_numpy(sigs).to(device)
            enc_kmers = torch.from_numpy(enc_kmers).to(device)
            output = model(sigs, enc_kmers).detach().cpu().numpy()
        else:
            output = model.run([], {"sig": sigs, "seq": enc_kmers})[0]
        all_outputs.append(output)
        all_loss.append(
            criterion(torch.from_numpy(output), torch.from_numpy(model_labels))
            .detach()
            .cpu()
            .numpy()
        )
        if full_results_fh is not None:
            full_results_fh.write_results(
                output, model_labels, read_ids, read_focus_bases
            )
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels)
    if is_torch_model:
        torch.set_grad_enabled(True)
    all_probs = softmax_axis1(all_outputs)
    acc, conf_mat, filt_frac, filt_acc, filt_conf_mat = compute_metrics(
        all_probs, all_labels, filt_frac
    )
    return VAL_METRICS(
        loss=np.mean(all_loss),
        acc=acc,
        num_calls=all_labels.size,
        conf_mat=conf_mat,
        filt_frac=filt_frac,
        filt_acc=filt_acc,
        filt_conf_mat=filt_conf_mat,
    )


def process_mods_probs(probs, labels, allow_unbalanced, pct_filt, name):
    if not allow_unbalanced:
        nlabs = labels.max() + 1
        # split probs
        label_probs = [probs[labels == mod_idx] for mod_idx in range(nlabs)]
        lab_sizes = [lp.shape[0] for lp in label_probs]
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

    acc, conf_mat, filt_frac, filt_acc, filt_conf_mat = compute_metrics(
        probs, labels, pct_filt / 100
    )
    ms = VAL_METRICS(
        loss=np.NAN,
        acc=acc,
        num_calls=labels.size,
        conf_mat=conf_mat,
        filt_frac=filt_frac,
        filt_acc=filt_acc,
        filt_conf_mat=filt_conf_mat,
    )
    val_output = (
        f"\n{ValidationLogger.HEADER}\n"
        f"{name}\t0\t0\t"
        f"{ms.acc:.6f}\t{mat_to_str(ms.conf_mat)}\t"
        f"NAN\t{ms.num_calls}\t{ms.filt_frac:.4f}\t"
        f"{ms.filt_acc:.6f}\t{mat_to_str(ms.filt_conf_mat)}\n"
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
        display_progress_bar=False,
    ):
        ms = _validate_model(
            model,
            model_mod_bases,
            criterion,
            dataset,
            filt_frac,
            display_progress_bar=display_progress_bar,
            full_results_fh=self.full_results_fh,
        )
        self.fp.write(
            f"{val_type}\t{nepoch}\t{niter}\t"
            f"{ms.acc:.6f}\t{mat_to_str(ms.conf_mat)}\t"
            f"{ms.loss:.6f}\t{ms.num_calls}\t{ms.filt_frac:.4f}\t"
            f"{ms.filt_acc:.6f}\t{mat_to_str(ms.filt_conf_mat)}\n"
        )
        return ms


##################
# ModBAM Parsing #
##################


def parse_mod_bam(bam_path, gt_sites, alphabet, full_fh):
    """Parse modified base tags from BAM file recording probability of canonical
    and each mod at each site in ground truth sites.

    Arsg:
        bam_path (str): Path to mapped BAM file with modified base tags
        gt_sites (dict): First level keys are chromosome and strand 2-tuples,
            second level keys are reference positions pointing to a ground truth
            modified base single letter code.
        alphabet (str): Canonical base followed by modified bases found in
            ground truth data. Other modified bases in BAM file will be ignored.
        full_fh (File): File handle to write full results.

    Returns:
        2-tuple containing
            - Numpy array with shape (num_calls, num_mods + 1) containing
                probabilities at each valid site
            - Numpy array with shape (num_calls) containing ground truth labels
    """
    been_warned_strand = been_warned_mod = False
    nnocalls = nnomods = ninvalid = nnoref = 0
    probs, labels = [], []
    # hid warnings for no index when using unmapped or unsorted files
    pysam_save = pysam.set_verbosity(0)
    with pysam.AlignmentFile(bam_path, check_sq=False) as bam_fh:
        for read in tqdm(bam_fh, smoothing=0):
            if read.modified_bases is None:
                nnocalls += 1
                continue
            strand = "-" if read.is_reverse else "+"
            try:
                ctg_gt = gt_sites[(read.reference_name, strand)]
            except KeyError:
                nnomods += 1
                continue
            # note read.modified_bases stores positions in forward strand
            # query sequence coordinates
            # TODO handle duplex mods on opposite strand
            mod_pos_probs = defaultdict(dict)
            for (
                can_base,
                mod_strand,
                mod_name,
            ), mod_values in read.modified_bases.items():
                if (mod_strand == 0 and read.is_reverse) or (
                    mod_strand == 1 and not read.is_reverse
                ):
                    LOGGER.debug(
                        f"Invalid mod strand {mod_strand} {read.query_name} "
                        f"{bam_path}"
                    )
                    if not been_warned_strand:
                        LOGGER.warning(
                            "Reverse strand (duplex) mods not supported"
                        )
                        been_warned_strand = True
                    continue
                if mod_name not in alphabet:
                    ninvalid += 1
                    if not been_warned_mod:
                        LOGGER.warning(
                            f"BAM mod ({mod_name}) not found in ground truth"
                        )
                        been_warned_mod = True
                    continue
                for pos, prob in mod_values:
                    mod_pos_probs[pos][mod_name] = (prob + 0.5) / 256
            if len(mod_pos_probs) == 0:
                nnomods += 1
                continue

            q_to_r = dict(read.get_aligned_pairs(matches_only=True))
            for q_pos, pos_probs in mod_pos_probs.items():
                try:
                    r_pos = q_to_r[q_pos]
                    gt_mod = ctg_gt[r_pos]
                except KeyError:
                    # skip modified base calls not mapped to ref pos
                    nnoref += 1
                    continue
                gt_mod_idx = alphabet.index(gt_mod)
                labels.append(gt_mod_idx)
                # TODO handle case with invalid probs and set can_prob to 0
                # and mod probs to softmax values
                # create array of probs in fixed order and fill 0 probs
                pos_probs_full = np.array(
                    [1 - sum(pos_probs.values())]
                    + [pos_probs.get(mod_name, 0) for mod_name in alphabet[1:]]
                )
                if full_fh is not None:
                    full_fh.write(
                        f"{read.query_name}\t{q_pos}\t{read.reference_name}\t"
                        f"{r_pos}\t{strand}\t{gt_mod_idx}\t"
                        f"{','.join(map(str, pos_probs_full))}\n"
                    )
                probs.append(pos_probs_full)
    pysam.set_verbosity(pysam_save)
    LOGGER.debug(f"Skipped {nnocalls} reads without mod tags from {bam_path}")
    LOGGER.debug(
        f"Skipped {nnomods} reads without valid mod calls from {bam_path}"
    )
    LOGGER.debug(f"Skipped {ninvalid} invalid base calls from {bam_path}")
    LOGGER.debug(f"Skipped {nnoref} calls without valid ref from {bam_path}")
    LOGGER.debug(f"Parsed {len(labels)} valid sites from {bam_path}")
    if len(probs) < 1:
        raise RemoraError(
            f"No valid modification calls from {bam_path}. May need to revert "
            "to original MM-tag style. Try `sed s/C+m?,/C+m,/g`and see "
            "https://github.com/pysam-developers/pysam/issues/1123"
        )
    return np.array(probs), np.array(labels)


def validate_modbams(
    bams_and_beds,
    full_results_path,
    name,
    pct_filt,
    allow_unbalanced=False,
    seed=None,
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
            "gt_mod_idx\tmod_probs\n"
        )
    bams, beds = zip(*bams_and_beds)
    all_gt_sites = []
    all_mods = set()
    for bed_path in beds:
        gt_sites, samp_mods = parse_mods_bed(bed_path)
        all_gt_sites.append(gt_sites)
        all_mods.update(samp_mods)
        tot_sites = sum(len(cs_sites) for cs_sites in gt_sites.values())
        LOGGER.debug(
            f"Parsed {tot_sites} total sites with labels {samp_mods} "
            f"from {bed_path}"
        )
    can_base = all_mods.intersection("ACGTU")
    if len(can_base) > 1:
        raise RemoraError("More than one canonical base found: {can_base}")
    if len(can_base) == 0:
        raise RemoraError("No canonical bases found in ground truth.")
    mod_bases = all_mods.difference("ACGTU")
    alphabet = "".join(can_base) + "".join(sorted(mod_bases))

    all_probs, all_labels = [], []
    for bam_path, gt_sites in zip(bams, all_gt_sites):
        probs, labels = parse_mod_bam(bam_path, gt_sites, alphabet, full_fh)
        all_probs.append(probs)
        all_labels.append(labels)

    LOGGER.info(f"Alphabet used (and order of reported metrics: {alphabet}")
    process_mods_probs(
        np.vstack(all_probs),
        np.concatenate(all_labels),
        allow_unbalanced,
        pct_filt,
        name,
    )
