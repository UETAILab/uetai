"""image classifier callbacks"""
from typing import Any, Dict, List, Optional

# import torch
from torch import Tensor
from pytorch_lightning import Trainer, Callback, LightningModule
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.warnings import rank_zero_warn

from uetai.logger import SummaryWriter

try:
    import wandb
except (ImportError, AssertionError):
    wandb = None


class ImageMonitorBase(Callback):
    """Base class for monitoring image data in a LightningModule.
    """
    def __init__(self, log_every_n_steps: int = 20):
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

    def on_train_batch_end(
        self, trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        # log every n steps
        if (batch_idx + 1) % self._log_ever_n_steps != 0:
            return

        named_tensor: Dict[str, Tensor] = {}
        tensor, gt, _ = batch  # tensor, label, batch_size
        if isinstance(outputs, Dict):
            pred = outputs['pred']
        elif isinstance(outputs, List):
            pred = outputs

        # TODO: this is classification task, it'll be moved to another callback soon
        for predict, idx in enumerate(pred):
            # convert each tensor from [batch, w, h] -> [w, h, batch]
            image = tensor[idx].permute(1,2,0)
            named_tensor[str(predict)] = image
        self.add_image(tag='train/media', batch=named_tensor)

    def add_image(
        self, batch: Dict[str, Tensor], tag: str = None,
    ) -> None:
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
            self.__add_image(tag=tag, tensor=batch)

        if isinstance(batch, Dict):
            for name, tensor in batch.items():
                self.__add_image(tag=tag, tensor=tensor, name=name, step=step)

    def __add_image(
        self, tag: str, tensor: Tensor, name: str = None,
    ) -> None:
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
        tensor = tensor.detach().cpu().numpy()

        logger.experiment.log(
            {tag: wandb.Image(tensor, caption=name)},
        )

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


