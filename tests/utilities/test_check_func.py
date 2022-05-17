"""Check function test."""
import unittest
from unittest import TestCase

from uetai.utilities import check_uetai_version


class TestVersion(TestCase):
    """Test version."""
    def test_check_version(self,):
        """Test version check."""
        self.assertFalse(check_uetai_version())  # current version is post1.dev


if __name__ == '__main__':
    unittest.main()
