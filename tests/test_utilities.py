"""Utilities test."""
# import pytest

from uetai.utilities import module_available


def test_utilities_import():
    """Test utilities import."""
    assert module_available('os') is True
    assert module_available('os.bla') is False
    assert module_available('bla.bla') is False
