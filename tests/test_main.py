""" Test main module.
"""
from pathlib import Path
from subprocess import check_call

import pysam
import pytest

from remora.data_chunks import RemoraDataset, CoreRemoraDataset

pytestmark = pytest.mark.main

# These may look like they are copied from remora.constants, but they're
# slightly different
FINAL_MODEL_FILENAME = "model_final.pt"
# SAVE_DATASET_FILENAME = "remora_train_data.npz"

MODELS_DIR = Path(__file__).absolute().parent.parent / "models"
CONV_MODEL = MODELS_DIR / "Conv_w_ref.py"
CONV_LSTM_MODEL = MODELS_DIR / "ConvLSTM_w_ref.py"
MODEL_PATHS = [CONV_MODEL, CONV_LSTM_MODEL]

EXPECTED_CAN_SIZE = 205
EXPECTED_MOD_SIZE = 210


@pytest.mark.smoke
def test_help():
    check_call(["remora", "-h"])


@pytest.mark.unit
@pytest.mark.etl
def test_prep_can(can_chunks):
    dataset = CoreRemoraDataset(
        str(can_chunks),
        batch_size=10,
    )
    assert dataset.size == EXPECTED_CAN_SIZE
    assert dataset.get_modbase_label_counts()[0] == EXPECTED_CAN_SIZE


@pytest.mark.unit
@pytest.mark.etl
def test_prep_mod(mod_chunks):
    dataset = CoreRemoraDataset(
        str(mod_chunks),
        batch_size=10,
    )
    assert dataset.size == EXPECTED_MOD_SIZE
    assert dataset.get_modbase_label_counts()[1] == EXPECTED_MOD_SIZE


@pytest.mark.unit
@pytest.mark.etl
def test_prep_mod_chebi(mod_chebi_chunks):
    dataset = CoreRemoraDataset(
        str(mod_chebi_chunks),
        batch_size=10,
    )
    assert dataset.size == EXPECTED_MOD_SIZE
    assert dataset.get_modbase_label_counts()[1] == EXPECTED_MOD_SIZE


@pytest.mark.unit
@pytest.mark.etl
def test_remora_dataset(chunks):
    dataset = RemoraDataset.from_config(
        str(chunks),
        batch_size=10,
    )
    label_counts = dataset.get_modbase_label_counts()
    assert label_counts.size == 2, "label counts sie should be 2"
    assert dataset.size == EXPECTED_CAN_SIZE + EXPECTED_MOD_SIZE
    assert label_counts[0] == EXPECTED_CAN_SIZE
    assert label_counts[1] == EXPECTED_MOD_SIZE


@pytest.mark.unit
@pytest.mark.etl
def test_remora_dataset_chebi(chebi_chunks):
    dataset = RemoraDataset.from_config(
        str(chebi_chunks),
        batch_size=10,
    )
    label_counts = dataset.get_modbase_label_counts()
    assert label_counts.size == 4, "label counts sie should be 4"
    assert dataset.size == EXPECTED_CAN_SIZE + (3 * EXPECTED_MOD_SIZE)
    assert label_counts[0] == EXPECTED_CAN_SIZE
    assert label_counts[1] == EXPECTED_MOD_SIZE
    assert label_counts[2] == EXPECTED_MOD_SIZE
    assert label_counts[3] == EXPECTED_MOD_SIZE


@pytest.mark.unit
@pytest.mark.etl
def test_dataset_inspect(chunks, tmpdir_factory):
    """Run `dataset inspect` on the command line."""
    print(f"Running command line `remora dataset inspect` on {chunks}")
    out_path = tmpdir_factory.mktemp("remora_tests") / "dataset_inspect.cfg"
    print(f"Output config path: {out_path}")
    check_call(
        [
            "remora",
            "dataset",
            "inspect",
            str(chunks),
            "--out-path",
            str(out_path),
        ],
    )


@pytest.mark.unit
@pytest.mark.etl
def test_chebi_dataset_inspect(chebi_chunks, tmpdir_factory):
    """Run `dataset inspect` on the command line."""
    print(f"Running command line `remora dataset inspect` on {chebi_chunks}")
    out_path = tmpdir_factory.mktemp("remora_tests") / "dataset_inspect.cfg"
    print(f"Output config path: {out_path}")
    check_call(
        [
            "remora",
            "dataset",
            "inspect",
            str(chebi_chunks),
            "--out-path",
            str(out_path),
        ],
    )


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
            "--chunk-context",
            "50",
            "50",
            *train_cli_args,
        ],
    )


@pytest.mark.unit
@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_chebi_train(model_path, tmpdir_factory, chebi_chunks, train_cli_args):
    """Run `model train` on the command line."""
    print(f"Running command line `remora model train` with model {model_path}")
    out_dir = tmpdir_factory.mktemp("remora_tests") / "train_mod_model"
    print(f"Output file: {out_dir}")
    check_call(
        [
            "remora",
            "model",
            "train",
            str(chebi_chunks),
            "--output-path",
            str(out_dir),
            "--model",
            model_path,
            "--chunk-context",
            "50",
            "50",
            *train_cli_args,
        ],
    )


@pytest.mark.unit
@pytest.mark.parametrize("model_path", [CONV_LSTM_MODEL])
def test_train_dynamic_chunk_context(
    model_path, tmpdir_factory, chunks, train_cli_args
):
    """Run `model train` on the command line reducing chunk context from
    (50, 50) to (30, 25)."""
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
            "--chunk-context",
            "30",
            "25",
            *train_cli_args,
        ],
    )


@pytest.mark.unit
@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_train_dynamic_kmer_context(
    model_path, tmpdir_factory, chunks, train_cli_args
):
    """Run `model train` on the command line reducing kmer context bases from
    (4, 4) to (2, 3)."""
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
            "--kmer-context-bases",
            "2",
            "3",
            "--chunk-context",
            "50",
            "50",
            *train_cli_args,
        ],
    )


@pytest.mark.unit
@pytest.mark.parametrize("model_path", [CONV_LSTM_MODEL])
def test_train_dynamic_both(model_path, tmpdir_factory, chunks, train_cli_args):
    """Run `model train` on the command line reducing both chunk context and
    kmer context bases."""
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
            "--chunk-context",
            "30",
            "25",
            "--kmer-context-bases",
            "2",
            "3",
            *train_cli_args,
        ],
    )


@pytest.mark.unit
def test_mod_infer(tmpdir_factory, can_pod5, can_mappings, fw_mod_model_dir):
    out_dir = tmpdir_factory.mktemp("remora_tests")
    print(f"Output dir: {out_dir}")
    out_file = out_dir / "mod_infer.bam"
    log_file = out_dir / "mod_infer.log"
    check_call(
        [
            "remora",
            "infer",
            "from_pod5_and_bam",
            can_pod5,
            can_mappings,
            "--model",
            str(fw_mod_model_dir / FINAL_MODEL_FILENAME),
            "--out-bam",
            out_file,
            "--log-filename",
            log_file,
        ],
    )


@pytest.mark.unit
def test_chebi_mod_infer(
    tmpdir_factory, can_pod5, can_mappings, fw_mod_chebi_model_dir
):
    # TODO use this output bam to test validate from modbams
    out_dir = tmpdir_factory.mktemp("remora_tests")
    print(f"Output dir: {out_dir}")
    out_file = out_dir / "mod_infer.bam"
    log_file = out_dir / "mod_infer.log"
    check_call(
        [
            "remora",
            "infer",
            "from_pod5_and_bam",
            can_pod5,
            can_mappings,
            "--model",
            str(fw_mod_chebi_model_dir / FINAL_MODEL_FILENAME),
            "--out-bam",
            out_file,
            "--log-filename",
            log_file,
        ],
    )


@pytest.mark.unit
def test_model_download(pretrain_model_args):
    check_call(
        [
            "remora",
            "model",
            "download",
            *pretrain_model_args,
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
    reads_pod5_path, pairs_path = duplex_reads_and_pairs_pod5
    FINAL_MODEL_FILENAME = "model_final.pt"
    out_dir = tmpdir_factory.mktemp("remora_tests")
    print(f"Pretrained validate results output: {out_dir}")
    out_path = out_dir / "mod_infer.bam"
    log_path = out_dir / "mod_infer.log"
    check_call(
        [
            "remora",
            "infer",
            "duplex_from_pod5_and_bam",
            reads_pod5_path,
            simplex_alignments,
            duplex_mapped_alignments,
            pairs_path,
            "--model",
            str(fw_mod_model_dir / FINAL_MODEL_FILENAME),
            "--out-bam",
            out_path,
            "--log-filename",
            log_path,
        ],
    )

    assert out_path.exists()

    n_expected_alignments = 0
    with open(pairs_path) as fh:
        for line in fh:
            n_expected_alignments += 1
    n_observed_alignments = 0
    with pysam.AlignmentFile(out_path, "rb", check_sq=False) as out_bam_fh:
        for alignment in out_bam_fh:
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


@pytest.mark.unit
def test_chebi_mod_validate_from_dataset(
    tmpdir_factory, chebi_chunks, fw_mod_model_dir
):
    out_dir = tmpdir_factory.mktemp("remora_tests")
    print(f"Trained validate results output: {out_dir}")
    out_file = out_dir / "mod_validate.txt"
    full_file = out_dir / "mod_validate_full.txt"
    log_file = out_dir / "mod_validate.log"
    check_call(
        [
            "remora",
            "validate",
            "from_remora_dataset",
            chebi_chunks,
            "--model",
            str(fw_mod_model_dir / FINAL_MODEL_FILENAME),
            "--batch-size",
            "20",
            "--out-file",
            out_file,
            "--full-results-filename",
            full_file,
            "--log-filename",
            log_file,
        ],
    )


@pytest.mark.unit
def test_mod_validate_from_modbams(
    tmpdir_factory,
    can_modbam,
    can_gt_bed,
    mod_modbam,
    mod_gt_bed,
):
    out_dir = tmpdir_factory.mktemp("remora_tests")
    print(f"Pretrained validate results output: {out_dir}")
    full_file = out_dir / "mod_validate_full.txt"
    log_file = out_dir / "mod_validate.log"
    check_call(
        [
            "remora",
            "validate",
            "from_modbams",
            "--bam-and-bed",
            can_modbam,
            can_gt_bed,
            "--bam-and-bed",
            mod_modbam,
            mod_gt_bed,
            "--full-results-filename",
            full_file,
            "--log-filename",
            log_file,
            "--explicit-mod-tag-used",
            "--extra-bases",
            "h",
        ],
    )

    assert full_file.exists()
    assert log_file.exists()


####################
# Analyze Commands #
####################


@pytest.mark.unit
def test_plot_ref_region(
    tmpdir_factory,
    can_pod5,
    can_mappings,
    mod_pod5,
    mod_mappings,
    ref_regions,
    can_gt_bed,
    levels,
):
    """Run `analyze plot ref_region` on the command line."""
    print("Running command line `remora analyze plot ref_region`")
    out_dir = tmpdir_factory.mktemp("plot_ref_region")
    log_path = out_dir / "log.txt"
    plot_path = out_dir / "remora_raw_signal_plot.pdf"
    print(f"Output dir: {out_dir}")
    check_call(
        [
            "remora",
            "analyze",
            "plot",
            "ref_region",
            "--pod5-and-bam",
            can_pod5,
            can_mappings,
            "--pod5-and-bam",
            mod_pod5,
            mod_mappings,
            "--ref-regions",
            ref_regions,
            "--highlight-ranges",
            can_gt_bed,
            "--refine-kmer-level-table",
            levels,
            "--refine-rough-rescale",
            "--refine-scale-iters",
            "0",
            "--plots-filename",
            plot_path,
            "--log-filename",
            log_path,
        ],
    )
