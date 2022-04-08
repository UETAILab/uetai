"""Utilities test."""
# import pytest
import unittest
from unittest import TestCase

from uetai.utilities import module_available

class TestUtilities(TestCase):
    """Test utilities."""
    def test_utilities_import(self,):
        """Test utilities import."""
        self.assertTrue(module_available('os'))
        self.assertFalse(module_available('os.path.bla'))
        self.assertFalse(module_available('bla.bla.asdf'))


if __name__ == '__main__':
    unittest.main()
