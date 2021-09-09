"""Image callback ./callbacks/image_monitor testcase"""
import unittest

import pytest
from unittest import mock

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.base import LoggerCollection
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from uetai.callbacks import ImageMonitorBase
from uetai.logger import SummaryWriter


class TestImageCallbacks(unittest.TestCase):
    def test_base_log_interval_overide(self,):
        """Test logging interval set by log_every_n_steps argument."""
        for log_every_n_steps in [1, 3, 4]:
            monitor = ImageMonitorBase()
            trainer = Trainer(
                callbacks=[monitor],
                log_every_n_steps=log_every_n_steps
            )
            # self.assertEqual(monitor._log_every_n_steps, log_every_n_steps)
            monitor.on_train_start(trainer=trainer, pl_module=None)
            assert monitor._log_every_n_steps == log_every_n_steps

    def test_base_no_logger_warning(self, ):
        """Test passing no logger warning."""
        monitor = ImageMonitorBase()
        trainer = Trainer(
            logger=False, callbacks=[ImageMonitorBase()]
        )
        with pytest.warns(
                UserWarning,
                match=(
                        "Cannot log image because Trainer has no logger.")):
            monitor.on_train_start(trainer, pl_module=None)

    def test_base_unsupported_logger_warning(self, ):
        """Test passing unsupported logger"""
        logger_collection = LoggerCollection([
            TensorBoardLogger('tmp'),
            WandbLogger(name='tmp')
        ])
        monitor = ImageMonitorBase()
        trainer = Trainer(
            logger=logger_collection, callbacks=[ImageMonitorBase()]
        )
        with pytest.warns(UserWarning):
            monitor.on_train_start(trainer, pl_module=None)

    def test_training_image_monitor(self):
        outputs = [
            dict((key, torch.rand((10, 10))) for key in ('loss', 'pred')),
            torch.rand(10, 10),
            # torch.rand(10, 10).tolist(), TODO: add xfail here
        ]
        tmp_images = [
            torch.rand(10, 3, 100, 100),
            # torch.rand(1, 5, 10, 10) TODO: add xfail here
        ]
        for output in outputs:
            for tmp_image in tmp_images:
                monitor = ImageMonitorBase()
                logger = SummaryWriter("uetai", log_tool='wandb')
                model = nn.Linear(100, 10)  # assume dataset have 10 classes
                trainer = Trainer(
                    logger=logger,
                    log_every_n_steps=1,
                    callbacks=[monitor],
                )
                monitor.on_train_start(trainer, model)

                # log input tensor and prediction
                example_data = [
                    tmp_image,  # tensor
                    torch.rand(10),  # ground-truth
                    10  # batch-size
                ]
                monitor.on_train_batch_end(
                    trainer,
                    model,
                    batch=example_data,
                    outputs=output,
                    batch_idx=0,
                    dataloader_idx=0
                )
