"""Check function test."""
import unittest
from unittest import TestCase

from uetai.utilities import check_uetai_version


class TestVersion(TestCase):
    """Test version."""
    def test_check_version(self,):
        """Test version check."""
        self.assertTrue(check_uetai_version())


if __name__ == '__main__':
    unittest.main()
