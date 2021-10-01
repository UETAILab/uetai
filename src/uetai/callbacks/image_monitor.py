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
    def __init__(
        self,
        on_step: bool = True,
        on_epoch: bool = False,
        log_every_n_steps: int = None,
        log_n_element_per_epoch: int = None,
        label_mapping: Dict[int, Any] = None,  # idx, name
    ):
        super().__init__()
        self._log = False
        self._trainer = pl.Trainer
        self._on_step = on_step
        self._on_epoch = on_epoch
        self._label_mapping = label_mapping
        self._log_every_n_steps = log_every_n_steps  # default is 50 in trainer.log_every_n_steps
        self._log_n_element_per_epoch = log_n_element_per_epoch  # default log all
        if self._on_epoch:
            self._epoch: Dict[str, List] = {'images': [], 'ground_truths': [], 'predictions': []}
            self._on_step = False

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._trainer = trainer
        self._log = check_logger(trainer.logger)
        self._log_every_n_steps = self._log_every_n_steps or trainer.log_every_n_steps
        if not (self._on_step or self._on_epoch):
            self._log = False

    def on_train_batch_end(
        self, trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        """
        This will handle the batch, output and pass it to `add_image` method.
        In `add_image`, user should customize how they want image would be logged.
        In `ImageMonitorBase`, we only provide a simple logging demonstration.
        """
        # log every n steps
        if not self._log or (batch_idx + 1) % self._log_every_n_steps != 0:
            return
        compressed_batch: Dict[str, List] = {'images': [], 'ground_truths': [], 'predictions': []}
        tensor, gt, _ = batch  # tensor, label, batch_size
        # output must be Tensor or Dict ơf Tensor
        if isinstance(outputs, Dict):
            pred = outputs['pred']
        elif isinstance(outputs, Tensor):
            pred = outputs
        else:
            raise TypeError(f"Except `outputs` to be List or Dict, get {type(outputs)}")

        for idx, image in enumerate(tensor):
            transformed_image = transforms.ToPILImage()(image).convert("RGB")  # WxH dimension
            compressed_batch['images'].append(transformed_image)  # batch_size x W x H dimension
            compressed_batch['ground_truths'].append(gt[idx].item())
            compressed_batch['predictions'].append(pred[idx].item())
        if self._on_epoch:
            for key, value in compressed_batch.items():
                self._epoch[key] += value   # epoch:  number_of_data x W x H dimension
        if self._on_step:
            self.add_image(tag='Train/media_step', media=compressed_batch)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: Optional = None
    ) -> None:
        if not self._on_epoch:
            return

        assert len(self._epoch) > 0, f"Expect List of `tensor` or `np.array`, receive {len(self._epoch)} elements"
        if self._log_n_element_per_epoch is None:
            top_n_image = self._epoch  # default log all images (not recommend)
        else:
            top_n_image = {key: val[:self._log_n_element_per_epoch] for key, val in self._epoch.items()}
        self.add_image(tag='Train/media_epoch', media=top_n_image)
        self._epoch = {}  # reset List for next epoch

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

    def add_image(self, media: Dict[str, List], tag: str = None,) -> None:
        """Override this method to customize the logging of Image.

        :param media: Dictionary contains images, ground_truth and predict
        :type media: Dict[str, List]
        :param tag: The tag the images
        :type tag: str, optional
        """
        images = []
        if tag is None:
            tag = 'Media'

        for idx, item in enumerate(media['images']):
            predict = media['predictions'][idx]
            gt = media['ground_truths'][idx]
            if self._label_mapping is not None:
                # TODO: check mapping valid
                predict = self._label_mapping[predict]  # mapping label
            images.append(wandb.Image(item, caption=f'Ground truth: {gt} Prediction: {predict}'))

        self._add_image(tag=tag, images=images)

    def _add_image(self, tag: str, images: Union[wandb.Image, List[wandb.Image]],) -> None:
        """
        Log image(s) to Weight & Biases dashboard.

        :param tag: The name of the logging panel in dashboard
        :type tag: str
        :param images: The `wandb.Image` or List of images that going to be logging
        :type images: List[wandb.Image] or wandb.Image
        """
        if wandb is None:
            raise ImportError("To log image with `wandb`, please it install with `pip install wandb`")
        logger = self._trainer.logger
        if isinstance(logger, SummaryWriter) and 'wandb' in logger.log_tool:
            logger.wandb_run.log({tag: images})


# class ClassificationMonitor(ImageMonitorBase):
#     def __init__(self, label_mapping, *args, **kwargs):
#         # TODO: mapping label
#         super().__init__(*args, **kwargs)
#         self._mapping = label_mapping
#
#     def add_image(self, batch: List, tag: str = None,) -> None:
#         """
#         Override `add_image` method
#         """
#         # TODO: extract predict, mapping label
#
#         #     if self._mapping is not None:
#         #         assert predict <= len(self._mapping), (
#         #             f"Can't mapping because {predict} doesn't belong to {self._mapping}"
#         #         )
#         #         predict = self._mapping[predict.item()]  # mapping label
#         pass
#         # map label's id to name
#         # label = item[0]
#         # image = item[1]
