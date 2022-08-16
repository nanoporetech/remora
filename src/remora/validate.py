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

from remora.io import parse_bed
from remora.util import softmax_axis1
from remora import RemoraError, constants, encoded_kmers


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
    full_results_fp=None,
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
        if full_results_fp is not None:
            full_results_fp.write_results(
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


class ResultsWriter:
    def __init__(self, out_fp):
        self.sep = "\t"
        self.out_fp = out_fp
        df = pd.DataFrame(
            columns=[
                "read_id",
                "read_focus_base",
                "label",
                "class_pred",
                "class_probs",
            ]
        )
        df.to_csv(self.out_fp, sep=self.sep, index=False)

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
        ).to_csv(self.out_fp, header=False, index=False, sep=self.sep)


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

    def __init__(self, fp, full_results_fp=None):
        self.fp = fp
        self.fp.write(self.HEADER + "\n")
        if full_results_fp is None:
            self.full_results_fp = None
        else:
            self.full_results_fp = ResultsWriter(full_results_fp)

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
            full_results_fp=self.full_results_fp,
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


def parse_mods(bam_fns, regs, mod_b, is_mod, full_fp):
    probs = []
    # hid warnings for no index when using unmapped or unsorted files
    pysam_save = pysam.set_verbosity(0)
    for bam_fn in bam_fns:
        with pysam.AlignmentFile(bam_fn, check_sq=False) as bam:
            for read in tqdm(bam, smoothing=0):
                if read.modified_bases is None:
                    continue
                strand = "-" if read.is_reverse else "+"
                ctg_coords = None
                try:
                    if regs is not None:
                        ctg_coords = regs[(read.reference_name, strand)]
                except KeyError:
                    continue
                # note read.modified_bases stores positions in forward strand
                # query sequence coordinates
                mod_pos_probs = [
                    pos_prob
                    for (_, _, mod_i), mod_values in read.modified_bases.items()
                    for pos_prob in mod_values
                    if mod_i == mod_b
                ]
                if len(mod_pos_probs) == 0:
                    continue
                q_to_r = dict(read.get_aligned_pairs(matches_only=True))
                for q_pos, m_prob in mod_pos_probs:
                    try:
                        r_pos = q_to_r[q_pos]
                    except KeyError:
                        continue
                    if ctg_coords is not None and r_pos not in ctg_coords:
                        continue
                    if full_fp is not None:
                        full_fp.write(
                            f"{read.query_name}\t{q_pos}\t{m_prob}\t"
                            f"{read.reference_name}\t{r_pos}\t{strand}\t"
                            f"{is_mod}\n"
                        )
                    probs.append(m_prob)
    pysam.set_verbosity(pysam_save)
    return np.array(probs)


def parse_gt_mods(bam_fns, mod_b, can_pos, mod_pos, full_fp):
    can_probs, mod_probs = [], []
    # hid warnings for no index when using unmapped or unsorted files
    pysam_save = pysam.set_verbosity(0)
    for bam_fn in bam_fns:
        with pysam.AlignmentFile(bam_fn, check_sq=False) as bam:
            for read in tqdm(bam, smoothing=0):
                if read.modified_bases is None:
                    continue
                ctg_can_pos = can_pos.get(read.reference_name)
                ctg_mod_pos = mod_pos.get(read.reference_name)
                if ctg_can_pos is None and ctg_mod_pos is None:
                    continue
                # note read.modified_bases stores positions in forward strand
                # query sequence coordinates
                mod_pos_probs = [
                    pos_prob
                    for (_, _, mod_i), mod_values in read.modified_bases.items()
                    for pos_prob in mod_values
                    if mod_i == mod_b
                ]
                if len(mod_pos_probs) == 0:
                    continue
                q_to_r = dict(read.get_aligned_pairs(matches_only=True))
                for q_pos, m_prob in mod_pos_probs:
                    try:
                        r_pos = q_to_r[q_pos]
                    except KeyError:
                        continue
                    if ctg_can_pos is not None and r_pos in ctg_can_pos:
                        is_mod = False
                    elif ctg_mod_pos is not None and r_pos in ctg_mod_pos:
                        is_mod = True
                    else:
                        continue
                    if full_fp is not None:
                        strand = "-" if read.is_reverse else "+"
                        full_fp.write(
                            f"{read.query_name}\t{q_pos}\t{m_prob}\t"
                            f"{read.reference_name}\t{r_pos}\t{strand}\t"
                            f"{is_mod}\n"
                        )
                    if is_mod:
                        mod_probs.append(m_prob)
                    else:
                        can_probs.append(m_prob)
    pysam.set_verbosity(pysam_save)
    return np.array(can_probs), np.array(mod_probs)


def parse_ground_truth_file(gt_data_fn):
    can_pos = defaultdict(set)
    mod_pos = defaultdict(set)
    with open(gt_data_fn) as fp:
        for line in fp:
            ctg, _, pos, is_mod = line.strip().split(",")
            if is_mod == "False":
                can_pos[ctg].add(int(pos))
            else:
                mod_pos[ctg].add(int(pos))
    return can_pos, mod_pos


def validate_from_modbams(
    bams,
    mod_bams,
    gt_pos_fn,
    regs_bed,
    full_results_fn,
    mod_base,
    name,
    pct_filt,
    allow_unbalanced=False,
):
    regs = None if regs_bed is None else parse_bed(regs_bed)
    full_fp = None
    if full_results_fn is not None:
        full_fp = open(full_results_fn, "w", buffering=512)
        atexit.register(full_fp.close)
        full_fp.write(
            "query_name\tquery_pos\tmod_prob\tref_name\tref_pos\tstrand\t"
            "is_mod\n"
        )
    if mod_bams is None:
        if gt_pos_fn is None:
            raise RemoraError(
                "Must provide either mod_bams or ground_truth_positions"
            )
        can_pos, mod_pos = parse_ground_truth_file(gt_pos_fn)
        can_probs, mod_probs = parse_gt_mods(
            bams, mod_base, can_pos, mod_pos, full_fp
        )
    else:
        can_probs = parse_mods(bams, regs, mod_base, False, full_fp)
        mod_probs = parse_mods(mod_bams, regs, mod_base, True, full_fp)

    if can_probs.size == 0:
        raise RemoraError("No valid modification calls from canonical set.")
    if mod_probs.size == 0:
        raise RemoraError("No valid modification calls from modified set.")
    if not allow_unbalanced:
        if can_probs.size > mod_probs.size:
            np.random.shuffle(can_probs)
            can_probs = can_probs[: mod_probs.size]
        else:
            np.random.shuffle(mod_probs)
            mod_probs = mod_probs[: can_probs.size]

    probs = np.empty((can_probs.size + mod_probs.size, 2))
    probs[: can_probs.size, 1] = (can_probs + 0.5) / 256
    probs[can_probs.size :, 1] = (mod_probs + 0.5) / 256
    probs[:, 0] = 1 - probs[:, 1]
    labels = np.zeros(can_probs.size + mod_probs.size)
    labels[can_probs.size :] = 1
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
    print(ValidationLogger.HEADER + "\n")
    print(
        f"{name}\t0\t0\t"
        f"{ms.acc:.6f}\t{mat_to_str(ms.conf_mat)}\t"
        f"NAN\t{ms.num_calls}\t{ms.filt_frac:.4f}\t"
        f"{ms.filt_acc:.6f}\t{mat_to_str(ms.filt_conf_mat)}\n"
    )
