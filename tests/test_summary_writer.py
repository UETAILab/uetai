import os
from unittest import TestCase


class TestSummaryWriterLocal(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSummaryWriterLocal, self).__init__(*args, **kwargs)
        os.environ.pop("WANDB_API_KEY", None)
        from uetai.logger import SummaryWriter
        self.logger = SummaryWriter("example")

    def test_name(self):
        a = self.logger.name
        if not isinstance(a, str):
            raise TypeError("type must be string")


class TestSummaryWriterWandb(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSummaryWriterWandb, self).__init__(*args, **kwargs)
        from uetai.logger import SummaryWriter
        self.logger = SummaryWriter("example")

    def test_name(self):
        a = self.logger.name
        if not isinstance(a, str):
            raise TypeError("type must be string")
