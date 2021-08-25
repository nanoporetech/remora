""" Test main module.
"""
import pytest
from subprocess import check_call

pytestmark = pytest.mark.main


@pytest.mark.unit
def test_help():
    check_call(["remora", "-h"])
