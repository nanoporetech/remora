from pathlib import Path
import pytest
from subprocess import check_call


def pytest_collection_modifyitems(session, config, items):
    # For any test that is not marked by a registered mark, add the unit mark
    # so that the test is run by gitlab (unmarked tests are never run in
    # gitlab testing)
    for item in items:
        if (
            len(
                set(mark.name for mark in item.iter_markers()).intersection(
                    ("format", "unit")
                )
            )
            == 0
        ):
            item.add_marker("unit")


##################
# Input Fixtures #
##################


@pytest.fixture(scope="session")
def can_tai_map_sig():
    """Canonical Taiyaki mapped signal file"""
    return Path(__file__).absolute().parent / "data" / "can_tai_map_sig.hdf5"


@pytest.fixture(scope="session")
def mod_tai_map_sig():
    """Modified base Taiyaki mapped signal file"""
    return Path(__file__).absolute().parent / "data" / "mod_tai_map_sig.hdf5"


#############################
# Extract Training Fixtures #
#############################


@pytest.fixture(scope="session")
def can_chunks(tmpdir_factory, can_tai_map_sig):
    """Run `prepare_taiyaki_train_data canonical` on the command line."""
    print("Running command line `remora prepare_taiyaki_train_data canonical`")
    output = tmpdir_factory.mktemp("remora_tests") / "can_remora_chunks.hdf5"
    print(f"Output file: {output}")
    check_call(
        [
            "remora",
            "prepare_taiyaki_train_data",
            "canonical",
            str(can_tai_map_sig),
            "--output-mapped-signal-file",
            str(output),
        ],
    )
    return output


@pytest.fixture(scope="session")
def mod_chunks(tmpdir_factory, mod_tai_map_sig):
    """Run `prepare_taiyaki_train_data modbase` on the command line."""
    print("Running command line `remora prepare_taiyaki_train_data modbase`")
    output = tmpdir_factory.mktemp("remora_tests") / "mod_remora_chunks.hdf5"
    print(f"Output file: {output}")
    check_call(
        [
            "remora",
            "prepare_taiyaki_train_data",
            "modbase",
            str(mod_tai_map_sig),
            "--output-mapped-signal-file",
            str(output),
            "--mod-motif",
            "m",
            "CG",
            "0",
        ],
    )
    return output


########################
# Train Model Fixtures #
########################


@pytest.fixture(scope="session")
def train_cli_args():
    return [
        "--val-prop",
        "0.1",
        "--mod-motif",
        "m",
        "CG",
        "0",
        "--batch-size",
        "10",
        "--epochs",
        "3",
        "--size",
        "16",
        "--save-freq",
        "2",
    ]


@pytest.fixture(scope="session")
def fw_model_path():
    return (
        Path(__file__).absolute().parent.parent / "models" / "ConvLSTM_w_ref.py"
    )


@pytest.fixture(scope="session")
def fw_model_checkpoint(
    fw_model_path, tmpdir_factory, mod_chunks, train_cli_args
):
    """Run `train_model` on the command line."""
    print(
        f"Running command line `remora train_model` with model {fw_model_path}"
    )
    out_dir = tmpdir_factory.mktemp("remora_tests") / "train_mod_model"
    out_ckpt = out_dir / "model_final.checkpoint"
    print(f"Output file: {out_dir}")
    check_call(
        [
            "remora",
            "train_model",
            "--dataset-path",
            str(mod_chunks),
            "--output-path",
            str(out_dir),
            "--model",
            str(fw_model_path),
            *train_cli_args,
        ],
    )
    return out_ckpt
