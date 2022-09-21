""" Test main module.
"""
from pathlib import Path
from subprocess import check_call

import pysam
import pytest

from remora.data_chunks import RemoraDataset
from remora import io

pytestmark = pytest.mark.main

# These may look like they are copied from remora.constants, but they're
# slightly different
FINAL_MODEL_FILENAME = "model_final.pt"
# SAVE_DATASET_FILENAME = "remora_train_data.npz"

MODELS_DIR = Path(__file__).absolute().parent.parent / "models"
MODEL_PATHS = [
    model_path
    for model_path in MODELS_DIR.iterdir()
    if str(model_path).endswith(".py")
    # TODO fix var width model data loading
    and str(model_path).find("var_width") == -1
]


@pytest.mark.smoke
def test_help():
    check_call(["remora", "-h"])


@pytest.mark.unit
@pytest.mark.etl
def test_prep_can(can_chunks):
    dataset = RemoraDataset.load_from_file(
        str(can_chunks),
        batch_size=10,
        balanced_batch=False,
    )
    assert dataset.nchunks == 75
    assert dict(dataset.get_label_counts()) == {0: 75}


@pytest.mark.unit
@pytest.mark.etl
def test_prep_mod(mod_chunks):
    dataset = RemoraDataset.load_from_file(
        str(mod_chunks),
        batch_size=10,
        balanced_batch=False,
    )
    assert dataset.nchunks == 75
    assert dict(dataset.get_label_counts()) == {1: 75}


@pytest.mark.unit
@pytest.mark.etl
def test_remora_dataset(chunks):
    dataset = RemoraDataset.load_from_file(
        str(chunks),
        batch_size=10,
        balanced_batch=False,
    )
    assert len(dataset.get_label_counts()) > 1, "label counts should be > 1"
    assert dataset.nchunks == 150
    assert dict(dataset.get_label_counts()) == {1: 75, 0: 75}


##################
# Mod Prediction #
##################


@pytest.mark.unit
@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_train(model_path, tmpdir_factory, chunks, train_cli_args):
    """Run `model train` on the command line."""
    print(f"Running command line `remora model train` with model {model_path}")
    out_dir = tmpdir_factory.mktemp("remora_tests") / "train_mod_model"
    print(f"Output file: {out_dir}")
    check_call(
        [
            "remora",
            "model",
            "train",
            str(chunks),
            "--output-path",
            str(out_dir),
            "--model",
            model_path,
            *train_cli_args,
        ],
    )
    return out_dir


@pytest.mark.unit
def test_mod_infer(tmpdir_factory, can_pod5, can_mappings, fw_mod_model_dir):
    out_file = tmpdir_factory.mktemp("remora_tests") / "mod_infer.txt"
    check_call(
        [
            "remora",
            "infer",
            "from_pod5_and_bam",
            can_pod5,
            can_mappings,
            "--model",
            str(fw_mod_model_dir / FINAL_MODEL_FILENAME),
            "--out-file",
            out_file,
        ],
    )


@pytest.mark.unit
@pytest.mark.duplex
def test_mod_infer_duplex(
    tmpdir_factory,
    simplex_alignments,
    duplex_mapped_alignments,
    duplex_reads_and_pairs_pod5,
    fw_mod_model_dir,
):
    reads_pod5_fp, pairs_fp = duplex_reads_and_pairs_pod5
    FINAL_MODEL_FILENAME = "model_final.pt"
    out_file_fp = tmpdir_factory.mktemp("remora_tests") / "mod_infer.txt"
    check_call(
        [
            "remora",
            "infer",
            "duplex_from_pod5_and_bam",
            reads_pod5_fp,
            simplex_alignments,
            duplex_mapped_alignments,
            pairs_fp,
            "--model",
            str(fw_mod_model_dir / FINAL_MODEL_FILENAME),
            "--out-file",
            out_file_fp,
        ],
    )

    assert out_file_fp.exists()

    n_expected_alignments = len(io.DuplexPairsIter.parse_pairs(pairs_fp))
    n_observed_alignments = 0
    with pysam.AlignmentFile(out_file_fp, "rb", check_sq=False) as out_bam:
        for alignment in out_bam:
            # KeyError when not present
            alignment.get_tag("MM")
            alignment.get_tag("ML")
            n_observed_alignments += 1

    assert n_expected_alignments == n_observed_alignments


@pytest.mark.unit
def test_mod_infer_pretrain(can_modbam):
    print(can_modbam)


@pytest.mark.unit
def test_mod_validate_from_dataset(tmpdir_factory, chunks, fw_mod_model_dir):
    out_dir = tmpdir_factory.mktemp("remora_tests")
    print(f"Trained validate results output: {out_dir}")
    out_file = out_dir / "mod_validate.txt"
    full_file = out_dir / "mod_validate_full.txt"
    check_call(
        [
            "remora",
            "validate",
            "from_remora_dataset",
            chunks,
            "--model",
            str(fw_mod_model_dir / FINAL_MODEL_FILENAME),
            "--batch-size",
            "20",
            "--out-file",
            out_file,
            "--full-results-filename",
            full_file,
        ],
    )


@pytest.mark.skip(
    reason="pysam MM bug https://github.com/pysam-developers/pysam/issues/1123"
)
@pytest.mark.unit
def test_mod_validate_from_modbams(tmpdir_factory, can_modbam, mod_modbam):
    out_dir = tmpdir_factory.mktemp("remora_tests")
    print(f"Pretrained validate results output: {out_dir}")
    full_file = out_dir / "mod_validate_full.txt"
    check_call(
        [
            "remora",
            "validate",
            "from_modbams",
            "--bams",
            can_modbam,
            "--mod-bams",
            mod_modbam,
            "--full-results-filename",
            full_file,
        ],
    )


###################
# Base Prediction #
###################


@pytest.mark.skip(reason="Base prediction")
@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_train_base_pred(
    model_path, tmpdir_factory, can_chunks, train_cli_args
):
    """Run `model train` on the command line with base prediction option."""
    print(f"Running command line `remora train_model` with model {model_path}")
    out_dir = tmpdir_factory.mktemp("remora_tests") / "train_base_pred_model"
    print(f"Output file: {out_dir}")
    check_call(
        [
            "remora",
            "model",
            "train",
            str(can_chunks),
            "--output-path",
            str(out_dir),
            "--model",
            model_path,
            *train_cli_args,
        ],
    )
    return out_dir


@pytest.mark.skip(reason="Base prediction")
@pytest.mark.unit
def test_base_pred_validate(tmpdir_factory, can_chunks, fw_base_pred_model_dir):
    out_dir = tmpdir_factory.mktemp("remora_tests")
    print(f"Trained validate base pred results output: {out_dir}")
    out_file = out_dir / "base_pred_validate.txt"
    full_file = out_dir / "base_pred_validate_full.txt"
    check_call(
        [
            "remora",
            "validate",
            "from_remora_dataset",
            can_chunks,
            "--model",
            str(fw_base_pred_model_dir / FINAL_MODEL_FILENAME),
            "--batch-size",
            "20",
            "--out-file",
            out_file,
            "--full-results-filename",
            full_file,
        ],
    )
