from unittest import TestCase

from uetai.logger.general import check_requirements, check_python


class TestGeneralFunctions(TestCase):
    def test_check_requirements(self):
        check_requirements("./requirement.txt")

    def test_check_python(self):
        check_python
