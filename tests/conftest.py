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
def can_pod5():
    """Canonical POD5 signal file"""
    return Path(__file__).absolute().parent / "data" / "can_reads.pod5"


@pytest.fixture(scope="session")
def can_mappings():
    """Canonical mappings bam file"""
    return Path(__file__).absolute().parent / "data" / "can_mappings.bam"


@pytest.fixture(scope="session")
def mod_pod5():
    """Modified POD5 signal file"""
    return Path(__file__).absolute().parent / "data" / "mod_reads.pod5"


@pytest.fixture(scope="session")
def mod_mappings():
    """Modified mappings bam file"""
    return Path(__file__).absolute().parent / "data" / "mod_mappings.bam"


@pytest.fixture(scope="session")
def pretrain_model_args():
    """Arguments to select model matched to above data"""
    return (
        "--pore",
        "dna_r10.4.1_e8.2",
        "--basecall-model-type",
        "hac",
        "--basecall-model-version",
        "v3.5.1",
        "--remora-model-type",
        "CG",
        "--remora-model-version",
        "0",
        "--modified-bases",
        "5mC",
    )


#############################
# Extract Training Fixtures #
#############################


@pytest.fixture(scope="session")
def can_chunks(tmpdir_factory, can_pod5, can_mappings):
    """Run `remora dataset prepare` on canonical data."""
    print("\nRunning `remora dataset prepare` canonical")
    out_dir = tmpdir_factory.mktemp("remora_tests")
    chunks_fn = out_dir / "can_chunks.npz"
    log_fn = out_dir / "log.txt"
    print(f"Output file: {chunks_fn}")
    print(f"Log file: {log_fn}")
    check_call(
        [
            "remora",
            "dataset",
            "prepare",
            str(can_pod5),
            str(can_mappings),
            "--output-remora-training-file",
            str(chunks_fn),
            "--log-filename",
            str(log_fn),
            "--mod-base-control",
            "--motif",
            "CG",
            "0",
            "--num-extract-alignment-workers",
            "1",
            "--num-extract-chunks-workers",
            "1",
        ],
    )
    return chunks_fn


@pytest.fixture(scope="session")
def mod_chunks(tmpdir_factory, mod_pod5, mod_mappings):
    """Run `remora dataset prepare` on modified data."""
    print("\nRunning `remora dataset prepare` on modified data")
    out_dir = tmpdir_factory.mktemp("remora_tests")
    chunks_fn = out_dir / "mod_chunks.npz"
    log_fn = out_dir / "log.txt"
    print(f"Output file: {chunks_fn}")
    print(f"Log file: {log_fn}")
    check_call(
        [
            "remora",
            "dataset",
            "prepare",
            str(mod_pod5),
            str(mod_mappings),
            "--output-remora-training-file",
            str(chunks_fn),
            "--log-filename",
            str(log_fn),
            "--mod-base",
            "m",
            "5mC",
            "--motif",
            "CG",
            "0",
            "--num-extract-alignment-workers",
            "1",
            "--num-extract-chunks-workers",
            "1",
        ],
    )
    return chunks_fn


@pytest.fixture(scope="session")
def chunks(tmpdir_factory, can_chunks, mod_chunks):
    """Run `remora dataset merge`."""
    print("\nRunning `remora dataset merge`")
    out_dir = tmpdir_factory.mktemp("remora_tests")
    chunks_fn = out_dir / "chunks.npz"
    print(f"Output file: {chunks_fn}")
    check_call(
        [
            "remora",
            "dataset",
            "merge",
            "--input-dataset",
            str(can_chunks),
            "1000",
            "--input-dataset",
            str(mod_chunks),
            "1000",
            "--output-dataset",
            str(chunks_fn),
        ],
    )
    return chunks_fn


########################
# Train Model Fixtures #
########################


@pytest.fixture(scope="session")
def train_cli_args():
    return [
        "--val-prop",
        "0.1",
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
def fw_base_pred_model_dir(
    fw_model_path, tmpdir_factory, can_chunks, train_cli_args
):
    """Run `train_model` on the command line with --base-pred."""
    print(
        f"\nRunning command line `remora train_model` with model "
        f"{fw_model_path}"
    )
    out_dir = tmpdir_factory.mktemp("remora_tests") / "train_can_model"
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
            str(fw_model_path),
            *train_cli_args,
        ],
    )
    return out_dir


@pytest.fixture(scope="session")
def fw_mod_model_dir(fw_model_path, tmpdir_factory, chunks, train_cli_args):
    """Run `train_model` on the command line."""
    print(
        f"\nRunning command line `remora train_model` with model "
        f"{fw_model_path}"
    )
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
            str(fw_model_path),
            *train_cli_args,
        ],
    )
    return out_dir


###################
# ModBAM Fixtures #
###################


@pytest.fixture(scope="session")
def can_modbam(tmpdir_factory, can_pod5, can_mappings, pretrain_model_args):
    out_dir = tmpdir_factory.mktemp("remora_tests")
    print(f"\nPretrained infer results output: {out_dir}")
    out_file = out_dir / "can_infer_pretrain.bam"
    full_file = out_dir / "can_infer_pretrain_full.txt"
    check_call(
        [
            "remora",
            "infer",
            "from_pod5_and_bam",
            can_pod5,
            can_mappings,
            "--out-file",
            out_file,
            *pretrain_model_args,
        ],
    )
    return out_file


@pytest.fixture(scope="session")
def mod_modbam(tmpdir_factory, mod_pod5, mod_mappings, pretrain_model_args):
    out_dir = tmpdir_factory.mktemp("remora_tests")
    print(f"\nPretrained infer results output: {out_dir}")
    out_file = out_dir / "mod_infer_pretrain.bam"
    full_file = out_dir / "mod_infer_pretrain_full.txt"
    check_call(
        [
            "remora",
            "infer",
            "from_pod5_and_bam",
            mod_pod5,
            mod_mappings,
            "--out-file",
            out_file,
            *pretrain_model_args,
        ],
    )
    return out_file
