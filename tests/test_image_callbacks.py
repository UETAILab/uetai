"""Image callback ./callbacks/image_monitor testcase"""
import pytest
from unittest import mock

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.base import LoggerCollection
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from uetai.callbacks import ImageMonitorBase


# class TestImageMonitorBase(TestCase):
@pytest.mark.parametrize('log_every_n_steps', [1, 3, 5])
def test_base_log_interval_overide(log_every_n_steps):
    """Test logging interval set by log_every_n_steps argument."""
    monitor = ImageMonitorBase(log_every_n_steps)
    trainer = Trainer(callbacks=[monitor])
    # self.assertEqual(monitor._log_every_n_steps, log_every_n_steps)
    assert monitor._log_every_n_steps == log_every_n_steps


def test_base_no_logger_warning():
    """Test passing no logger warning."""
    monitor = ImageMonitorBase()
    trainer = Trainer(logger=False, callbacks=[ImageMonitorBase()])
    with pytest.warns(
        UserWarning,
        match=(
            "Cannot log image because Trainer has no logger.")):
        monitor.on_train_start(trainer, pl_module=None)


def test_base_unsupported_logger_warning(tmpdir):
    """Test passing unsupported logger"""
    logger_collection = LoggerCollection([
        TensorBoardLogger(save_dir=tmpdir),
        WandbLogger(name='tmp', save_dir=tmpdir)
    ])
    monitor = ImageMonitorBase()
    trainer = Trainer(
        logger=logger_collection, callbacks=[ImageMonitorBase()]
    )
    with pytest.warns(UserWarning):
        monitor.on_train_start(trainer, pl_module=None)


@mock.patch("uetai.callbacks.ImageMonitorBase.add_image")
def test_training_image_monitor(image):
    monitor = ImageMonitorBase()
    pass
