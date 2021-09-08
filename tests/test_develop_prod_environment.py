import os
from unittest import TestCase

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from uetai.logger import SummaryWriter


class TestSummaryWriter(TestCase):
    def test_instance(self):
        logger = SummaryWriter("uetai")
        assert isinstance(logger.logger, WandbLogger)

        os.environ.pop("WANDB_API_KEY", None)
        logger = SummaryWriter("uetai")
        assert isinstance(logger.logger, TensorBoardLogger)
