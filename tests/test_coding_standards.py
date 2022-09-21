"""
Test coding standards via black
"""

import pytest
from subprocess import Popen, STDOUT


@pytest.mark.format
def test_black():
    """Test that code meets black coding standards"""
    p = Popen(["black", "--check", "--diff", "."], stderr=STDOUT)
    p.communicate()
    if p.returncode:
        raise RuntimeError("black coding standards failed.")


@pytest.mark.format
def test_flake8():
    """Test that code meets flak8 coding standards"""
    p = Popen(["flake8", "."], stderr=STDOUT)
    p.communicate()
    if p.returncode:
        raise RuntimeError("flake8 coding standards failed.")


if __name__ == "__main__":
    pytest.main(["-m", "format", "--no-cov", "-s"])
