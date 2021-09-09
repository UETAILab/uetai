import os
from unittest import TestCase

from uetai.logger.general import check_requirements, check_python


class TestGeneralFunctions(TestCase):
    def test_check_requirements(self):
        print(os.getcwd())
        check_requirements("../requirements.txt")

    def test_check_python(self):
        check_python(minimum="3.6")
