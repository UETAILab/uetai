"""Utilities tests"""
import os
from unittest import TestCase

from uetai.logger.general import (
    check_online, check_requirements, check_python, colorstr, emojis
)


class TestGeneralFunctions(TestCase):
    """general.py function tests"""
    def test_check_requirements(self):
        print(os.getcwd())
        check_requirements("../requirements.txt")

    def test_check_python(self):
        check_python(minimum="3.6")

    def test_check_online(self):
        self.assertEqual(check_online(), True)
        # go offline and assert False

    def test_print_function(self):
        dummy_str = emojis(colorstr("This is a test 🧪"))
        print(dummy_str)
