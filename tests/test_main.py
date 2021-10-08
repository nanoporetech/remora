""" Test main module.
"""
from pathlib import Path
import pytest
from subprocess import check_call

pytestmark = pytest.mark.main

# These are copied from remora.constants
#  - can't figure out how to load this from pytest
FINAL_MODEL_FILENAME = "model_final.onnx"
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
def test_train_mod(model_path, tmpdir_factory, mod_chunks, train_cli_args):
    """Run `train_model` on the command line."""
    print(f"Running command line `remora train_model` with model {model_path}")
    out_dir = tmpdir_factory.mktemp("remora_tests") / "train_mod_model"
    print(f"Output file: {out_dir}")
    check_call(
        [
            "remora",
            "train_model",
            str(mod_chunks),
            "--output-path",
            str(out_dir),
            "--model",
            model_path,
            *train_cli_args,
        ],
    )
    return out_dir


@pytest.mark.unit
def test_mod_infer(tmpdir_factory, mod_tai_map_sig, fw_mod_model_dir):
    out_dir = tmpdir_factory.mktemp("remora_tests") / "mod_infer"
    check_call(
        [
            "remora",
            "infer",
            mod_tai_map_sig,
            str(fw_mod_model_dir / FINAL_MODEL_FILENAME),
            "--batch-size",
            "20",
            "--output-path",
            out_dir,
        ],
    )


@pytest.mark.unit
def test_can_infer(tmpdir_factory, can_tai_map_sig, fw_mod_model_dir):
    out_dir = tmpdir_factory.mktemp("remora_tests") / "can_infer"
    check_call(
        [
            "remora",
            "infer",
            can_tai_map_sig,
            str(fw_mod_model_dir / FINAL_MODEL_FILENAME),
            "--batch-size",
            "20",
            "--output-path",
            out_dir,
        ],
    )


###################
# Base Prediction #
###################


@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_train_base_pred(
    model_path, tmpdir_factory, can_chunks, train_cli_args
):
    """Run `train_model` on the command line with base prediction option."""
    print(f"Running command line `remora train_model` with model {model_path}")
    out_dir = tmpdir_factory.mktemp("remora_tests") / "train_base_pred_model"
    print(f"Output file: {out_dir}")
    check_call(
        [
            "remora",
            "train_model",
            str(can_chunks),
            "--output-path",
            str(out_dir),
            "--model",
            model_path,
            *train_cli_args,
        ],
    )
    return out_dir


@pytest.mark.unit
def test_base_pred_infer(
    tmpdir_factory, can_tai_map_sig, fw_base_pred_model_dir
):
    out_dir = tmpdir_factory.mktemp("remora_tests") / "can_infer_base_pred"
    print(out_dir)
    check_call(
        [
            "remora",
            "infer",
            can_tai_map_sig,
            str(fw_base_pred_model_dir / FINAL_MODEL_FILENAME),
            "--batch-size",
            "20",
            "--output-path",
            out_dir,
        ],
    )
