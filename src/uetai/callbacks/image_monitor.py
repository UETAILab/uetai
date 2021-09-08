"""image classifier callbacks"""
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torchvision import transforms
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
    def __init__(
        self, log_every_n_steps: int = None, label_mapping: Dict[int, str] = None
    ):
        super().__init__()
        self._log_every_n_steps: Optional[int] = log_every_n_steps
        self._log = False
        self._trainer = Trainer
        self._train_batch_idx: int
        self._mapping = label_mapping

    def on_train_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._log = self._check_logger(trainer.logger)
        self._log_every_n_steps = self._log_every_n_steps or trainer.log_every_n_steps
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
        if not self._log or (batch_idx + 1) % self._log_every_n_steps != 0:
            return

        named_tensor: List = []
        tensor, gt, _ = batch  # tensor, label, batch_size
        if isinstance(outputs, Dict):
            pred = outputs['pred']
        elif isinstance(outputs, Tensor):
            pred = outputs
        else:
            raise Exception(
                f"Except `outputs` to be List or Dict, get {type(outputs)}"
            )

        # get prediction
        pred = torch.max(pred.detach(), dim=1)[1]

        # this is classification task, it'll be moved to another callback soon
        for idx, predict in enumerate(pred):
            image = transforms.ToPILImage()(tensor[idx]).convert("RGB")
            if self._mapping is not None:
                assert predict <= len(self._mapping), (
                    f"Can't mapping because {predict} doesn't belong to {self._mapping}"
                )
                predict = self._mapping[predict.item()]  # mapping label
            named_tensor.append([str(predict), image])
        self.add_image(tag='Media/train', batch=named_tensor)

    def add_image(
        self, batch: List, tag: str = None,
    ) -> None:
        """Log image(s) to Weight & Biases dashboard.

        :param batch: Image or collection of images (List of PIL Image)
        :type image: List
        :param tag: The tag the images
        :type tag: str, optional

        .. example::
            .. code::python
            >>> buh buh lmao

        """
        images = []
        if tag is None:
            tag = 'Media'
        for item in batch:
            # map label's id to name
            label = item[0]
            image = item[1]
            images.append(wandb.Image(image, caption=label))
        self.__add_image(tag=tag, images=images)

    def __add_image(
        self, tag: str, images: List or wandb.Image,
    ) -> None:
        """Override this method to customize the logging of Image.

        :param tag: The name of the logging panel in dashboard
        :type tag: str
        :param images: The `wandb.Image` that going to be logging
        :type images: List
        """
        if wandb is None:
            raise ImportError(
                "To log image with `wandb`, please it install with `pip install wandb`"
            )
        logger = self._trainer.logger
        logger.wandb_run.log({tag: images})

    def _check_logger(self, logger: LightningLoggerBase) -> bool:
        available = True
        if not logger:
            rank_zero_warn("Cannot log image because Trainer has no logger.")
            available = False
        if not isinstance(logger, SummaryWriter):
            rank_zero_warn(
                f"{self.__class__.__name__} does not "
                "support logging with {logger.__class__.__name__}."
            )
            available = False
        else:
            if 'wandb' not in logger.log_tool:
                rank_zero_warn(
                    "Current `Summary_Writer` is not running with `wandb`."
                    "Please enable `wandb` to log image"
                )
        return available
