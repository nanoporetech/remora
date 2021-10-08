import datetime
import imp
from os.path import isfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
from sklearn.metrics import precision_recall_curve

from remora import log, RemoraError, encoded_kmers

LOGGER = log.get_logger()


class ValidationLogger:
    def __init__(self, out_path, base_pred):
        self.fp = open(out_path / "validation.log", "w", buffering=1)
        self.base_pred = base_pred
        if base_pred:
            self.fp.write(
                "\t".join(
                    ("Val_Type", "Iteration", "Accuracy", "Loss", "Num_Calls")
                )
                + "\n"
            )
        else:
            self.fp.write(
                "\t".join(
                    (
                        "Val_Type",
                        "Iteration",
                        "Accuracy",
                        "Loss",
                        "F1",
                        "Precision",
                        "Recall",
                        "Num_Calls",
                    )
                )
                + "\n"
            )

    def close(self):
        self.fp.close()

    def validate_model(self, model, criterion, dataset, niter, val_type="val"):
        bb, ab = dataset.kmer_context_bases
        with torch.no_grad():
            model.eval()
            all_labels = []
            all_outputs = []
            all_loss = []
            for (sigs, seqs, seq_maps, seq_lens), labels, _ in dataset:
                all_labels.append(labels)
                enc_kmers = torch.from_numpy(
                    encoded_kmers.compute_encoded_kmer_batch(
                        bb, ab, seqs, seq_maps, seq_lens
                    )
                )
                if torch.cuda.is_available():
                    sigs = sigs.cuda()
                    enc_kmers = enc_kmers.cuda()
                output = model(sigs, enc_kmers).detach().cpu()
                all_outputs.append(output)
                loss = criterion(output, labels)
                all_loss.append(loss.detach().cpu().numpy())
            all_outputs = torch.cat(all_outputs)
            all_labels = torch.cat(all_labels)
            acc = (
                torch.argmax(all_outputs, dim=1) == all_labels
            ).float().sum() / all_outputs.shape[0]
            acc = acc.cpu().numpy()
            mean_loss = np.mean(all_loss)
            if self.base_pred:
                self.fp.write(
                    f"{val_type}\t{niter}\t{acc:.6f}\t{mean_loss:.6f}\t"
                    f"{len(all_labels)}\n"
                )
            else:
                with np.errstate(invalid="ignore"):
                    precision, recall, thresholds = precision_recall_curve(
                        all_labels.numpy(), all_outputs[:, 1].numpy()
                    )
                    f1_scores = 2 * recall * precision / (recall + precision)
                f1_idx = np.argmax(f1_scores)
                self.fp.write(
                    f"{val_type}\t{niter}\t{acc:.6f}\t{mean_loss:.6f}\t"
                    f"{f1_scores[f1_idx]}\t{precision[f1_idx]}\t"
                    f"{recall[f1_idx]}\t{len(all_labels)}\n"
                )
        return acc, mean_loss


def _load_python_model(model_file, **model_kwargs):

    netmodule = imp.load_source("netmodule", model_file)
    network = netmodule.network(**model_kwargs)
    return network


def continue_from_checkpoint(ckp_path, model_path=None):
    """Load a checkpoint in order to continue training."""
    if not isfile(ckp_path):
        raise RemoraError(f"Checkpoint path is not a file ({ckp_path})")
    LOGGER.info(f"Loading trained model from {ckp_path}")
    ckpt = torch.load(ckp_path)
    if ckpt["state_dict"] is None:
        raise RemoraError("Model state not saved in checkpoint.")

    model_path = ckpt["model_path"] if model_path is None else model_path
    model = _load_python_model(model_path, **ckpt["model_params"])
    model.load_state_dict(ckpt["state_dict"])
    return ckpt, model


def export_model(ckpt, model, save_filename):
    kmer_len = sum(ckpt["kmer_context_bases"]) + 1
    sig_len = sum(ckpt["chunk_context"])
    sig = torch.from_numpy(np.zeros((1, 1, sig_len), dtype=np.float32))
    seq = torch.from_numpy(
        np.zeros((1, kmer_len * 4, sig_len), dtype=np.float32)
    )
    model.eval()
    torch.onnx.export(
        model,
        (sig, seq),
        save_filename,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["sig", "seq"],
        output_names=["output"],
        dynamic_axes={
            "sig": {0: "batch_size"},
            "seq": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    onnx_model = onnx.load(save_filename)
    meta = onnx_model.metadata_props.add()
    meta.key = "creation_date"
    meta.value = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    meta = onnx_model.metadata_props.add()
    meta.key = "base_pred"
    meta.value = str(ckpt["base_pred"])
    meta = onnx_model.metadata_props.add()
    meta.key = "mod_bases"
    meta.value = str(ckpt["mod_bases"])
    for key in ("chunk_context", "kmer_context_bases"):
        for idx in range(2):
            meta = onnx_model.metadata_props.add()
            meta.key = f"{key}_{idx}"
            meta.value = str(ckpt[key][idx])
    meta = onnx_model.metadata_props.add()
    meta.key = "motif"
    meta.value = ckpt["motif"][0]
    meta = onnx_model.metadata_props.add()
    meta.key = "motif_offset"
    meta.value = str(ckpt["motif"][1])
    onnx_model.doc_string = "Nanopore Remora model"
    try:
        onnx_model.model_version = ckpt["model_version"]
    except KeyError:
        LOGGER.warning("Model version not found in checkpoint. Setting to 0.")
        onnx_model.model_version = 0
    onnx.save(onnx_model, save_filename)


def load_onnx_model(model_filename, device=None):
    """Load onnx model. If device is specified load onto specified device."""
    providers = None
    if device is not None:
        if ort.get_device() != "GPU":
            raise RemoraError(
                "GPU execution not available. Install compatible package via "
                "`pip install onnxruntime-gpu`"
            )
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device,
                },
            ),
            "CPUExecutionProvider",
        ]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model_sess = ort.InferenceSession(model_filename, so, providers=providers)
    model_metadata = dict(model_sess.get_modelmeta().custom_metadata_map)
    model_metadata["base_pred"] = model_metadata["base_pred"] == "True"
    if model_metadata["mod_bases"] == "None":
        model_metadata["mod_bases"] = None
    model_metadata["motif_offset"] = int(model_metadata["motif_offset"])
    model_metadata["kmer_context_bases"] = (
        int(model_metadata["kmer_context_bases_0"]),
        int(model_metadata["kmer_context_bases_1"]),
    )
    model_metadata["kmer_len"] = sum(model_metadata["kmer_context_bases"]) + 1
    model_metadata["chunk_context"] = (
        int(model_metadata["chunk_context_0"]),
        int(model_metadata["chunk_context_1"]),
    )
    model_metadata["chunk_len"] = sum(model_metadata["chunk_context"])
    model_metadata["motif"] = (
        model_metadata["motif"],
        int(model_metadata["motif_offset"]),
    )
    ckpt_attrs = "\n".join(
        f"  {k: >20} : {v}" for k, v in model_metadata.items()
    )
    LOGGER.debug(f"Loaded model attrs\n{ckpt_attrs}\n")
    return model_sess, model_metadata
