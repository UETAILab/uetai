"""Image callback ./callbacks/image_monitor testcase"""
import unittest
# import pytest
from parameterized import parameterized, param, parameterized_class

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from uetai.callbacks import ImageMonitorBase
from uetai.logger import SummaryWriter


@parameterized_class(('outputs', 'images'), [
    (dict((key, torch.rand(10, requires_grad=True)) for key in ('loss', 'pred')), torch.rand(10, 3, 100, 100)),
    (torch.rand(10, requires_grad=True), torch.rand(10, 3, 100, 100)),
])
class TestImageCallbacks(unittest.TestCase):
    """Image callbacks test"""

    def __init__(self, *args, **kwargs):
        super(TestImageCallbacks, self).__init__(*args, **kwargs)
        self.monitor = ImageMonitorBase()
        self.logger = SummaryWriter('uetai', log_tool='wandb')

    def test_base_log_interval_override(self, log_every_n_steps=1):
        """Test logging interval set by log_every_n_steps argument."""
        monitor = ImageMonitorBase()
        trainer = Trainer(
            logger=False,
            callbacks=[monitor],
            log_every_n_steps=log_every_n_steps
        )
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
        monitor = ImageMonitorBase()
        trainer = Trainer(
            logger=logger, callbacks=[ImageMonitorBase()]
        )
        monitor.on_train_start(trainer, pl_module=None)
        self.assertWarns(UserWarning)

    def test_training_image_monitor_by_step(self):
        monitor = ImageMonitorBase(on_step=True)
        trainer = Trainer(
            logger=self.logger,
            log_every_n_steps=1,
            callbacks=[monitor],
        )
        self._test_training_image_monitor(self.images, self.outputs, monitor, trainer)

    def test_training_monitor_by_epoch(self):
        monitor = ImageMonitorBase(on_epoch=True, log_n_element_per_epoch=2)
        trainer = Trainer(
            logger=self.logger,
            callbacks=[monitor]
        )
        self._test_training_image_monitor(self.images, self.outputs, monitor, trainer)
        monitor.on_train_epoch_end(trainer, None)

    # @parameterized.expand([
    #     (torch.rand(10), torch.rand(1, 5, 10, 10), TypeError),
    #     (str('fail'), torch.rand(10, 3, 100, 100), UserWarning)
    # ])
    # def test_training_image_monitor_xfail(self, outputs, images, expectation):
    #     with self.assertRaises(expectation):
    #         self._test_training_image_monitor(images, outputs)

    def _test_training_image_monitor(self, images, outputs, monitor=None, trainer=None):
        if monitor is None:
            monitor = ImageMonitorBase()
        if trainer is None:
            trainer = Trainer(logger=self.logger, callbacks=[monitor])
        monitor.on_train_start(trainer, None)
        example_data = [
            images,  # tensor
            torch.rand(10, requires_grad=True),  # ground-truth
            10  # batch-size
        ]
        monitor.on_train_batch_end(
            trainer, None,
            batch=example_data,
            outputs=outputs,
            batch_idx=0,
            dataloader_idx=0
        )
