""" Test that notebooks run
"""
from pathlib import Path
from subprocess import check_call

import pytest


NOTEBOOKS_DIR = Path(__file__).absolute().parent.parent / "notebooks"


@pytest.mark.notebook
@pytest.mark.parametrize("notebook_path", NOTEBOOKS_DIR.glob("*.ipynb"))
def test_notebook(notebook_path, tmpdir_factory):
    """Run notebook."""
    print(f"Running {notebook_path}")
    out_path = tmpdir_factory.mktemp("notebook_tests")
    print(f"Output config path: {out_path}")
    check_call(
        [
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "notebook",
            NOTEBOOKS_DIR / notebook_path,
            "--output-dir",
            out_path,
        ],
    )
