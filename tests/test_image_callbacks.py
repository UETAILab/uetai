"""Image callback ./callbacks/image_monitor testcase"""
import unittest
# import pytest
from parameterized import parameterized, param

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from uetai.callbacks import ImageMonitorBase
from uetai.logger import SummaryWriter


class TestImageCallbacks(unittest.TestCase):
    """Image callbacks test"""
    def __init__(self, *args, **kwargs) -> None:
        super(TestImageCallbacks, self).__init__(*args, **kwargs)
        self.monitor = ImageMonitorBase()

    def test_base_log_interval_overide(self,):
        """Test logging interval set by log_every_n_steps argument."""
        for log_every_n_steps in [1, 3, 4]:
            monitor = ImageMonitorBase()
            trainer = Trainer(
                callbacks=[self.monitor],
                log_every_n_steps=log_every_n_steps
            )
            # self.assertEqual(monitor._log_every_n_steps, log_every_n_steps)
            monitor.on_train_start(trainer=trainer, pl_module=None)
            self.assertEqual(monitor._log_every_n_steps, log_every_n_steps)

    @parameterized.expand([
        param(False),
        param(WandbLogger('tmp')),
        param(TensorBoardLogger('tmp')),
        param(SummaryWriter('uetai', log_tool='tensorboard')),
    ])
    def test_base_unsupported_logger_warning(self, logger):
        """Test passing unsupported logger"""
        trainer = Trainer(
            logger=logger, callbacks=[ImageMonitorBase()]
        )
        self.monitor.on_train_start(trainer, pl_module=None)

    @parameterized.expand([
        param(
            dict((key, torch.rand((10, 10))) for key in ('loss', 'pred')),
            torch.rand(10, 3, 100, 100)
        ),
        param(
            torch.rand(10, 10),
            torch.rand(10, 3, 100, 100),
        ),
    ])
    def test_training_image_monitor(self, outputs, tmp_images):
        logger = SummaryWriter("uetai", log_tool='wandb')
        model = nn.Linear(100, 10)  # assume dataset have 10 classes
        trainer = Trainer(
            logger=logger,
            log_every_n_steps=1,
            callbacks=[self.monitor],
        )
        self.monitor.on_train_start(trainer, model)

        # log input tensor and prediction
        example_data = [
            tmp_images,  # tensor
            torch.rand(10),  # ground-truth
            10  # batch-size
        ]
        self.monitor.on_train_batch_end(
            trainer,
            model,
            batch=example_data,
            outputs=outputs,
            batch_idx=0,
            dataloader_idx=0
        )

    @parameterized.expand([
        param(
            torch.rand(10, 10),
            torch.rand(1, 5, 10, 10)
        ),
        param(
            torch.rand(10, 10).tolist(),
            torch.rand(10, 3, 100, 100)
        )
    ])
    def test_training_image_monitor_xfail(self, outputs, tmp_images):
        with self.assertRaises(TypeError):
            self.test_training_image_monitor(outputs, tmp_images)
