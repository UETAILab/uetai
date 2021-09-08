"""Image callback ./callbacks/image_monitor testcase"""
import pytest
from unittest import mock

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.base import LoggerCollection
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from uetai.callbacks import ImageMonitorBase
from uetai.logger import SummaryWriter


@pytest.mark.parametrize('log_every_n_steps', [1, 3, 5])
def test_base_log_interval_overide(log_every_n_steps):
    """Test logging interval set by log_every_n_steps argument."""
    monitor = ImageMonitorBase()
    trainer = Trainer(
        callbacks=[monitor],
        log_every_n_steps=log_every_n_steps
    )
    # self.assertEqual(monitor._log_every_n_steps, log_every_n_steps)
    monitor.on_train_start(trainer=trainer, pl_module=None)
    assert monitor._log_every_n_steps == log_every_n_steps


def test_base_no_logger_warning():
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


def test_base_unsupported_logger_warning():
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


@pytest.mark.parametrize('outputs', [
    # Dict
    pytest.param(
        dict((key, torch.rand((10, 10))) for key in ('loss', 'pred')),
        id="Dict[str, Tensor]"
    ),
    # Tensor
    pytest.param(
        torch.rand(10, 10),
        id="Tensor"
    ),
    # unsupported type
    pytest.param(
        torch.rand(10, 10).tolist(),  # List
        marks=pytest.mark.xfail(exception=Exception),
        id="List"
    )
])
@pytest.mark.parametrize('tmp_image', [
    # normal image (1D - black-white; 2D - LA; 3D - RGB; 4D - RGBA)
    pytest.param(torch.rand(10, 3, 100, 100), id="Image"),
    # Not an image (>4D)
    pytest.param(
        torch.rand(1, 5, 10, 10),
        marks=pytest.mark.xfail(exception=ValueError),
        id="Non_image"
    )
])
@mock.patch("uetai.callbacks.ImageMonitorBase.add_image")
def test_training_image_monitor(tmpdir, outputs, tmp_image):
    monitor = ImageMonitorBase()
    logger = SummaryWriter("cov-test")
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
        outputs=outputs,
        batch_idx=0,
        dataloader_idx=0
    )
