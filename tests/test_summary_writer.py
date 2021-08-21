from unittest import TestCase

from uetai.logger import SummaryWriter


class TestSummaryWriter(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSummaryWriter, self).__init__(*args, **kwargs)
        self.logger = SummaryWriter("example")

    def test_name(self):
        a = self.logger.name
        if not isinstance(a, str):
            raise TypeError("type must be string")