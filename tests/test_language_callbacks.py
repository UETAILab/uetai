import unittest

import torch
from pytorch_lightning import Trainer

from uetai.callbacks import TextMonitorBase


class TestLanguageCallbacks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLanguageCallbacks, self).__init__(*args, **kwargs)
        self.monitor = TextMonitorBase()

    def test_text_monitor(self):
        """this is dummy test"""
        self.monitor.add_text()
        example_data = torch.randint(10, (10,))
        example_output = torch.rand(10)
        self.pipeline_image_monitor(outputs=example_output, batch=example_data)

    def pipeline_image_monitor(self, outputs, batch, monitor=None, trainer=None):
        if monitor is None:
            monitor = self.monitor
        if trainer is None:
            trainer = Trainer(callbacks=[monitor])
        monitor.on_train_start(trainer, None)

        monitor.on_train_batch_end(
            trainer, None,
            batch=batch,
            outputs=outputs,
            batch_idx=0,
            dataloader_idx=0
        )
        monitor.on_train_epoch_end(trainer, None)
        monitor.on_validation_batch_end(
            trainer, None,
            batch=batch,
            outputs=outputs,
            batch_idx=0,
            dataloader_idx=0
        )
        monitor.on_validation_epoch_end(trainer, None)
        monitor.on_fit_end(trainer, None)
