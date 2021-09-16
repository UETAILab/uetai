import os
from unittest import TestCase

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from uetai.logger import SummaryWriter


class TestSummaryWriter(TestCase):
    def test_instance(self):
        logger = SummaryWriter("uetai")
        assert isinstance(logger.logger, WandbLogger)

        api_key = os.environ.pop("WANDB_API_KEY", None)
        logger = SummaryWriter("uetai")
        os.environ["WANDB_API_KEY"] = api_key
        assert isinstance(logger.logger, TensorBoardLogger)

    def test_un_support_logger_type(self):
        with self.assertRaises(Exception):
            logger = SummaryWriter("uetai", log_tool="lcoal")