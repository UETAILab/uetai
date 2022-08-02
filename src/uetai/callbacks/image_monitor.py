"""image classifier callbacks"""
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torchvision import transforms

from ..utilities import module_available
_PT_LIGHTNING_AVAILABEL = module_available('pytorch_lightning')

if _PT_LIGHTNING_AVAILABEL:
    import pytorch_lightning as pl
    from pytorch_lightning import Callback

try:
    import wandb
except (ImportError, AssertionError):
    wandb = None
    warnings.warn('Missing package `wandb`. Run `pip install wandb` to install it')


class ImageMonitorBase(pl.Callback):
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
        self._on_step = on_step
        self._on_epoch = on_epoch
        self._label_mapping = label_mapping
        self._log_every_n_steps = log_every_n_steps  # default is 50 in trainer.log_every_n_steps
        self._log_n_element_per_epoch = log_n_element_per_epoch  # default log all
        if self._on_epoch:
            self._init_epoch()
            self._on_step = False

    def _init_epoch(self):
        self._epoch: Dict[str, List] = {'inputs': [], 'truths': [], 'predictions': []}

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: "pl.LightningModule") -> None:
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
        if self._log:
            compressed_batch: Dict[str, Any] = self._extract_output_and_batch(outputs, batch)
            if self._on_step and (batch_idx + 1) % self._log_every_n_steps == 0:
                self.add_image(tag='Train/media_step', media=compressed_batch, logger=trainer.logger)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: Optional = None
    ) -> None:
        if self._on_epoch:
            compressed_epoch = self._extract_epoch()
            self.add_image(tag='Train/media_epoch', media=compressed_epoch, logger=trainer.logger)
            self._init_epoch()

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self._log:
            compressed_batch: Dict[str, Any] = self._extract_output_and_batch(outputs, batch)
            if self._on_step and (batch_idx + 1) % self._log_every_n_steps == 0:
                self.add_image(tag='Valid/media_step', media=compressed_batch, logger=trainer.logger)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._on_epoch:
            compressed_epoch = self._extract_epoch()
            self.add_image(tag='Valid/media_epoch', media=compressed_epoch, logger=trainer.logger)
            self._init_epoch()

    def on_keyboard_interrupt(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        trainer_finish_run(trainer=trainer)

    def add_image(self, media: Dict[str, List], logger: Any, tag: str = 'Media',) -> None:
        """Override this method to customize the logging of Image.
        :param media: Dictionary contains `images`, `truths` and `predictions` as key
        :type media: Dict[str, List]
        :param logger: Logger
        :type logger: SummaryWriter
        :param tag: The tag the images
        :type tag: str, optional
        """
        images = []
        for idx, item in enumerate(media['inputs']):
            predict = media['predictions'][idx]
            if predict.dim() > 0:
                predict = torch.max(predict, dim=0)[1]
            gt = media['truths'][idx]
            if self._label_mapping is not None:
                predict = mapping_idx_label(predict.item(), self._label_mapping)
                gt = mapping_idx_label(gt, self._label_mapping)
            images.append(wandb.Image(item, caption=f'Truth: {gt} - Predict: {predict}'))

        _log_media(tag=tag, media=images, logger=logger)

    def _extract_output_and_batch(self, outputs: Any, batch: Any):
        compressed_batch: Dict[str, List] = {'inputs': [], 'truths': [], 'predictions': []}
        tensor, gt, _ = batch  # tensor, label, batch_size
        # output must be Tensor or Dict ơf Tensor
        if isinstance(outputs, Dict):
            pred = outputs['pred']
        elif isinstance(outputs, Tensor):
            pred = outputs
        else:
            raise TypeError(f"Except `outputs` to be List or Dict, get {type(outputs)}")

        for idx, image in enumerate(tensor):
            transformed_image = transforms.ToPILImage()(image)  # WxH dimension
            compressed_batch['inputs'].append(transformed_image)  # batch_size x W x H dimension
            compressed_batch['truths'].append(gt[idx].item())
            compressed_batch['predictions'].append(pred[idx])
        if self._on_epoch:
            for key, value in compressed_batch.items():
                self._epoch[key] += value  # epoch:  number_of_data x W x H dimension
        return compressed_batch

    def _extract_epoch(self,):
        assert len(self._epoch) > 0, f"Expect List of `tensor` or `np.array`, receive empty list"
        if self._log_n_element_per_epoch is None:
            return self._epoch  # default log all images (not recommend)

        top_n_image = {key: val[:self._log_n_element_per_epoch] for key, val in self._epoch.items()}
        return top_n_image


class ClassificationMonitor(ImageMonitorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._on_epoch and self._log_n_element_per_epoch is None:
            warnings.warn("Monitoring all elements in epoch is not recommended")
        if self._label_mapping is None:
            warnings.warn("Classification callback should init with mapping rules. Missing `label_mapping`")
        self.state = {"epoch": 0}
        columns = ['id', 'image', 'ground_truth', 'prediction']
        if self._label_mapping is not None:
            for id in range(len(self._label_mapping)):
                mapped_id = mapping_idx_label(id, self._label_mapping)
                columns.append("score_" + str(mapped_id))
        self.train_summary_table = wandb.Table(columns=columns)
        self.valid_summary_table = wandb.Table(columns=columns)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: Optional = None
    ) -> None:
        self.state['epoch'] += 1
        super().on_train_epoch_end(trainer, pl_module)

    def add_image(self, media: Dict[str, List], logger: Any, tag: str = 'Media',) -> None:
        """
        Override `add_image` method
        """
        # TODO: cache table or log table by epoch and merge for final report
        for idx, item in enumerate(media['inputs']):
            id = f"{self.state['epoch']}_{idx}"
            cache_data = [id, wandb.Image(item)]  # id + media/input
            predict = media['predictions'][idx]
            gt = media['truths'][idx]

            if self._label_mapping is not None:
                gt = mapping_idx_label(gt, self._label_mapping)
                cache_data.append(gt)  # truth
                if len(predict) > 1:
                    max_predict = torch.max(predict.detach(), dim=0)[1]
                    cache_data.append(mapping_idx_label(max_predict.item(), self._label_mapping))  # predictions
                    for pred in predict:
                        cache_data.append(pred.item())  # individual class score
            else:
                cache_data.extend([gt, torch.max(predict.detach(), dim=0)[1]])

            if 'Train' in tag:
                self.train_summary_table.add_data(*cache_data)
            else:
                self.valid_summary_table.add_data(*cache_data)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_fit_end(trainer, pl_module)
        _log_media(tag='Train_media_epoch', media=self.train_summary_table, logger=trainer.logger)
        _log_media(tag='Valid_media_epoch', media=self.valid_summary_table, logger=trainer.logger)

    def on_keyboard_interrupt(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_fit_end(trainer, pl_module)
        trainer_finish_run(trainer)


# Utilities -------------------------------------------------------------------
def mapping_idx_label(idx: int, label_mapping: Dict[int, str]):
    assert idx in label_mapping, f'Mapping dictionary is not including index {idx}'
    return label_mapping[idx]


def _log_media(
    tag: str, logger: Any,
    media: Union[wandb.Image, wandb.Table, List[wandb.Image]],
) -> None:
    """
    Log image(s) to Weight & Biases dashboard.
    :param tag: The name of the logging panel in dashboard
    :type tag: str
    :param media: The `wandb.Image` or List of images that going to be logging
    :type media: List[wandb.Image] or wandb.Image
    """
    if isinstance(media, wandb.Table):
        tag = tag.replace('/', '_')
    logger.experiment.log({tag: media})


def trainer_finish_run(trainer: "pl.Trainer") -> None:
    logger = trainer.logger
    logger.experiment.finish()
