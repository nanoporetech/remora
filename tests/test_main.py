""" Test main module.
"""
from pathlib import Path
import pytest
from subprocess import check_call

pytestmark = pytest.mark.main

# These are copied from remora.constants
#  - can't figure out how to load this from pytest
FINAL_MODEL_FILENAME = "model_final.pt"
SAVE_DATASET_FILENAME = "remora_train_data.npz"

MODELS_DIR = Path(__file__).absolute().parent.parent / "models"
MODEL_PATHS = [
    model_path
    for model_path in MODELS_DIR.iterdir()
    if str(model_path).endswith(".py")
    # TODO fix var width model data loading
    and str(model_path).find("var_width") == -1
]


@pytest.mark.unit
def test_help():
    check_call(["remora", "-h"])


@pytest.mark.unit
def test_prep_can(can_chunks):
    print(can_chunks)


@pytest.mark.unit
def test_prep_mod(mod_chunks):
    print(mod_chunks)


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
