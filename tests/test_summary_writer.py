from unittest import TestCase

from uetai.logger import SummaryWriter


class TestSummaryWriter(TestCase):
    def __init__(self):
        super().__init__()
        self.logger = SummaryWriter("example")

    def test_name(self):
        assert isinstance(self.logger.name, str)

    def test_version(self):
        assert isinstance(self.logger.version, str)
