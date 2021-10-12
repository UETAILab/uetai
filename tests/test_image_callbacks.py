"""Image callback ./callbacks/image_monitor testcase"""
import unittest
from parameterized import parameterized, param

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from uetai.callbacks import ImageMonitorBase, ClassificationMonitor
from uetai.logger import SummaryWriter


class TestImageCallbacks(unittest.TestCase):
    """Image callbacks test"""

    def __init__(self, *args, **kwargs):
        super(TestImageCallbacks, self).__init__(*args, **kwargs)
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
        monitor.on_train_epoch_end(trainer, None)

    @parameterized.expand([
        (dict((key, torch.rand((10, 10))) for key in ('loss', 'pred')), torch.rand(10, 3, 100, 100)),
        (torch.randint(10, (10,)), torch.rand(10, 3, 100, 100)),  # outputs is position of highest probability label
    ])
    def test_training_image_monitor(self, outputs, images):
        mapping = {i: str(i) for i in range(10)}
        monitor = ImageMonitorBase(label_mapping=mapping, log_every_n_steps=1)
        self.pipeline_image_monitor(outputs=outputs, images=images, monitor=monitor)

    @parameterized.expand([
        (ImageMonitorBase(on_step=True),),
        (ImageMonitorBase(on_epoch=True, log_n_element_per_epoch=2),),
        (ClassificationMonitor(on_epoch=True, label_mapping={i: str(i) for i in range(10)}),),
    ])
    def test_training_callbacks_by_epoch_n_step(self, monitor=None):
        sample_images = torch.rand(10, 3, 100, 100)
        sample_outputs = torch.rand((10, 10))
        trainer = Trainer(
            logger=self.logger,
            callbacks=[monitor]
        )
        self.pipeline_image_monitor(
            images=sample_images, outputs=sample_outputs, monitor=monitor, trainer=trainer
        )

    @parameterized.expand([
        (torch.rand(10, 10), torch.rand(1, 5, 10, 10), ValueError),
        (str('fail'), torch.rand(10, 3, 100, 100), TypeError)
    ])
    def test_training_image_monitor_xfail(self, outputs, images, expectation):
        with self.assertRaises(expectation):
            self.pipeline_image_monitor(outputs=outputs, images=images)

    def pipeline_image_monitor(self, outputs, images, monitor=None, trainer=None):
        if monitor is None:
            monitor = ImageMonitorBase(log_every_n_steps=1)
        if trainer is None:
            trainer = Trainer(logger=self.logger, callbacks=[monitor])
        monitor.on_train_start(trainer, None)
        example_data = [
            images,  # tensor
            torch.randint(10, (10,), dtype=torch.float),  # ground-truth
            10  # batch-size
        ]
        monitor.on_train_batch_end(
            trainer, None,
            batch=example_data,
            outputs=outputs,
            batch_idx=0,
            dataloader_idx=0
        )
        monitor.on_train_epoch_end(trainer, None)
        monitor.on_validation_batch_end(
            trainer, None,
            batch=example_data,
            outputs=outputs,
            batch_idx=0,
            dataloader_idx=0
        )
        monitor.on_validation_epoch_end(trainer, None)
        monitor.on_fit_end(trainer, None)
