"""image classifier callbacks"""
from typing import Any, Dict, Optional

# import torch
from torch import Tensor
from pytorch_lightning import Trainer, Callback, LightningModule
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.warnings import rank_zero_warn

from summary_writer import SummaryWriter

try:
    import wandb
except (ImportError, AssertionError):
    wandb = None


class ImageMonitorBase(Callback):
    """Base class for monitoring image data in a LightningModule.
    """
    def __init__(self, log_every_n_steps: int = None):
        super().__init__()
        self._log_ever_n_steps: Optional[int] = log_every_n_steps
        self._trainer = Trainer
        self._train_batch_idx: int
        self._log = False

    def on_train_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._log = self._check_logger(trainer.logger)
        self._log_ever_n_steps = self._log_ever_n_steps or trainer.log_every_n_steps
        self._trainer = trainer

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule,
        batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self._train_batch_idx = batch_idx

    def log_image(self, batch: Any, tag: str = None,):
        """Log image(s) to Weight & Biases dashboard.

        :param batch: The image or group of images
        :type image: torch tensor, or or a collection of it
        (tuple, list, dict, ...)
        :param tag: The tag the images
        :type tag: str, optional

        .. example::
            .. code::python
            >>> buh buh lmao

        """
        if tag is None:
            tag = 'Media'
        if isinstance(batch, Tensor):
            self.__log_image(tag=tag, tensor=batch)

        if isinstance(batch, Dict[str, Tensor]):
            for name, tensor in batch.items():
                self.__log_image(tag=tag, tensor=tensor, name=name)

    def __log_image(self, tag: str, tensor: Tensor, name: str = None) -> None:
        """Override this method to customize the logging of Image.

        :param image: The tensor for which to log as image
        :type image: Tensor
        :param name: The name of the image or image's class
        :type name: str, optional
        """
        if wandb is None:
            raise ImportError(
                "To log image with `wandb`, please it install with `pip install wandb`"
            )
        logger = self._trainer.logger
        tensor = tensor.detach().cpu()

        logger.experiment.log({
            f"{tag}": wandb.Image(tensor, caption=name)
        })

    def _check_logger(self, logger: LightningLoggerBase) -> bool:
        available = True
        if not logger:
            rank_zero_warn("Cannot log histograms because Trainer has no logger.")
            available = False
        if not isinstance(logger, SummaryWriter):
            rank_zero_warn(
                f"{self.__class__.__name__} does not "
                "support logging with {logger.__class__.__name__}."
            )
            available = False
        return available


