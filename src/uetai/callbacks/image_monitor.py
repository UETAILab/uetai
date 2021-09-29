"""image classifier callbacks"""
from typing import Any, Dict, List, Optional, Union

from torch import Tensor
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import Callback

from uetai.logger import SummaryWriter
from uetai.utils import warn_missing_pkg
from uetai.callbacks.utils import check_logger, trainer_finish_run

try:
    import wandb
except (ImportError, AssertionError):
    warn_missing_pkg("wandb")
    wandb = None


class ImageMonitorBase(Callback):
    """Base class for monitoring image data.
    """
    def __init__(self, log_every_n_steps: int = None, label_mapping: Dict[int, str] = None):
        super().__init__()
        self._log_every_n_steps: Optional[int] = log_every_n_steps
        self._log = False
        self._trainer = pl.Trainer
        self._mapping = label_mapping

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log = check_logger(trainer.logger)
        self._log_every_n_steps = self._log_every_n_steps or trainer.log_every_n_steps
        self._trainer = trainer

    def on_train_batch_end(
        self, trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        # log every n steps
        if not self._log or (batch_idx + 1) % self._log_every_n_steps != 0:
            return

        named_tensor: List = []
        tensor, _gt, _ = batch  # tensor, label, batch_size
        if isinstance(outputs, Dict):
            pred = outputs['pred']
        elif isinstance(outputs, Tensor):
            pred = outputs
        else:
            raise Exception(
                f"Except `outputs` to be List or Dict, get {type(outputs)}"
            )

        # this is classification task, it'll be moved to another callback soon
        for idx, predict in enumerate(pred):
            image = transforms.ToPILImage()(tensor[idx])
            if self._mapping is not None:
                assert predict <= len(self._mapping), (
                    f"Can't mapping because {predict} doesn't belong to {self._mapping}"
                )
                predict = self._mapping[predict.item()]  # mapping label
            named_tensor.append([str(predict), image])
        self.add_image(tag='Media/train', batch=named_tensor)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: Optional = None
    ) -> None:
        pass

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        pass

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_keyboard_interrupt(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        trainer_finish_run(trainer=trainer)

    def add_image(self, batch: List, tag: str = None,) -> None:
        """Override this method to customize the logging of Image.

        :param batch: Image or collection of images (List of PIL Image)
        :type batch: List
        :param tag: The tag the images
        :type tag: str, optional
        """
        images = []
        if tag is None:
            tag = 'Media'
        for item in batch:
            # map label's id to name
            label = item[0]
            image = item[1]
            images.append(wandb.Image(image, caption=label))
        self._add_image(tag=tag, images=images)

    def _add_image(self, tag: str, images: Union[wandb.Image, List[wandb.Image]],) -> None:
        """
        Log image(s) to Weight & Biases dashboard.

        :param tag: The name of the logging panel in dashboard
        :type tag: str
        :param images: The `wandb.Image` that going to be logging
        :type images: List
        """
        if wandb is None:
            raise ImportError("To log image with `wandb`, please it install with `pip install wandb`")
        logger = self._trainer.logger
        if isinstance(logger, SummaryWriter) and 'wandb' in logger.log_tool:
            logger.wandb_run.log({tag: images})


class ClassificationMonitor(ImageMonitorBase):
    def __init__(self, *args, **kwargs):
        # TODO: mapping label
        super().__init__(*args, **kwargs)

    def add_image(self, batch: List, tag: str = None,) -> None:
        """
        Override `add_image` method
        """
        pass
