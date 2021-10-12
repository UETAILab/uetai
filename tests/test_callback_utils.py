"""Callback utilities tests"""
import unittest
from parameterized import parameterized

from uetai.logger import SummaryWriter
from uetai.callbacks.utils import check_logger


class TestCallbackUtils(unittest.TestCase):
    @parameterized.expand([
        (None, False),
        (SummaryWriter('uetai'), True),
        (SummaryWriter('uetai', log_tool='tensorboard'), False, UserWarning),
    ])
    def test_check_logger(self, logger, available, expect=None):
        log = check_logger(logger)
        if expect is not None:
            self.assertWarns(expect)
        self.assertEqual(log, available)
