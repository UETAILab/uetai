"""Custom Wandb logger by UETAI."""
import os
import logging
from argparse import Namespace
from typing import Any, Optional, Dict, Union, List

import numpy as np
from PIL.Image import Image

import torch
from torch import nn

from .base import UetaiLoggerBase
from ..utilities import module_available

_WANDB_AVAILABLE = module_available("wandb")
_SAVING_PATH = ".uetai"

if not os.path.exists(_SAVING_PATH):
    os.makedirs(_SAVING_PATH, exist_ok=True)

if _WANDB_AVAILABLE:
    os.environ["WANDB_DIR"] = os.path.abspath(_SAVING_PATH)
    import wandb
    from wandb.sdk.wandb_run import Run
else:
    raise Warning("Wandb is not available. Try run `pip install wandb`.")


class WandbLogger(UetaiLoggerBase):
    def __init__(
            self,
            project_name: Optional[str] = 'uetai',
            workspace: Optional[str] = None,
            experiment=None,
            name=None,
            **kwargs,
    ):
        super().__init__()
        self.project_name = project_name
        self.workspace = workspace
        self._experiment = experiment
        self._name = name
        self._wandb_init = dict(
            name=name,
            project_name=project_name,
            entity=workspace,
        )
        self._wandb_init.update(**kwargs)

    # Initialize experiment -----------------------------------------------------------
    def _init_experiment(self, **kwargs) -> [Run, None]:
        if not _WANDB_AVAILABLE:
            raise ModuleNotFoundError(
                "Wandb is not available. Try install with `pip install wandb`.")
        experiment = wandb.init(project=self.project_name, entity=self.workspace, **kwargs, )
        return experiment

    @property
    def experiment(self) -> Run:
        """
        Getter wandb object for using wandb feature.
        """
        if self._experiment is None:
            self._experiment = self._init_experiment(**self._wandb_init)

        return self._experiment

    @property
    def name(self) -> str:
        """
        Gets name of the current experiment.

        :return: name of the current existed experiment.
        """
        return self._experiment.project_name() if self._experiment else self._name

    @property
    def version(self):
        """
        Gets id of the current experiment.

        :return: id of the current existed experiment.
        """
        return self._experiment.id if self._experiment else None

    # Logging -------------------------------------------------------------------------
    def watch(
        self, model: nn.Module,
        criterion: Optional[nn.Module] = None,
        log: str = "gradients",
        log_freq: int = 100,
        log_graph: bool = True
    ) -> None:
        """
        Integrate wandb with pytorch, and log gradients and parameters of the model.

        .. seealso::

            - `wandb.watch <https://docs.wandb.ai/ref/python/watch>`_

        .. code-block:: python

            logger = WandbLogger(project_name="uetai")
            model = Net()
            criterion = nn.CrossEntropyLoss()

            logger.watch(model, criterion, log="parameters")


        :param model: model to log.
        :type model: nn.Module
        :param criterion: criterion to log.
        :type criterion: nn.Module
        :param log: log type.
        :type log: str ["gradients", "parameters", "all", or None]
        :param log_freq: log frequency.
        :type log_freq: int
        :param log_graph: log graph.
        :type log_graph: bool
        :return:`wandb.Graph` object.
        """
        return self.experiment.watch(model,criterion=criterion, log=log, log_freq=log_freq, log_graph=log_graph)

    def log(self, data: Dict[str, Any], step: Optional[int] = None, **kwargs):
        """Wandb design there `log` function to log every type of data that they support.
        Eg:
        *  metrics (floats, ints)
        *  media (images, audios, text, ...)
        *  table
        *  graph (histogram, ...)
        *  3d object

        .. note::

            If you know the type of data, we recommend you use our `log_metric` or `log_image` function directly,
            where we can automatically detect the type of data and assign the right wandb object to it.

        .. code-block:: python
            logger = WandbLogger(project_name="uetai", workspace="uetai")

            # log metrics
            logger.log({"loss": 0.1, "acc": 0.9})

            # log media
            images = wandb.Image("myimage.jpg")
            logger.log({"image": images})  # PIL Image

            # log table
            my_table = wandb.Table(columns=["a", "b"], data=[["1a", "1b"], ["2a", "2b"]])
            logger.log({"table_key": my_table})


        :param data: data to log.
        :type data: dict
        :param step: step of the experiment.
        :type step: int
        :param kwargs: additional keyword arguments.
        :type kwargs: dict

        """
        return self.experiment.log(data=data, step=step, **kwargs)

    def log_parameters(self, params: Union[Dict[str, Any], Namespace], *args, **kwargs):
        return self.experiment.config.update(params, allow_val_change=True)

    def log_metric(self, metric_name: str, metric_value: float, step: Optional[int] = None):
        """
        Log a metric to wandb.

        .. code-block:: python

            logger = WandbLogger(project_name="uetai")
            logger.log_metric(metric_name="accuracy", metric_value=0.9}

        :param metric_name: name of the metric.
        :type metric_name: str
        :param metric_value: value of the metric.
        :type metric_value: float
        :param step: step of the metric.
        :type step: int
        """
        return self.experiment.log({metric_name: metric_value}, step=step)

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        Log multiple metrics to wandb.

        .. code-block:: python

            logger = WandbLogger(project_name="uetai")
            logger.log_metrics({"accuracy": 0.9, "loss": 0.1}, step=1)

        :param metrics: metrics to log.
        :type metrics: dict
        :param step: step of the experiment.
        :type step: int
        """
        self.experiment.log({**metrics}, step=step)

    def log_text(
        self,
        text: List[List[str]],
        name: Optional[str] = None,
        step: Optional[int] = None,
        metadata: List[str] = None
    ):
        """
        Log a text to wandb.

        :param text: text to log.
        :type text: list
        :param name: name of the text.
        :type name: str
        :param step: step of the text.
        :type step: int
        :param metadata: metadata of the text.
        :type metadata: list
        """
        text_table = wandb.Table(columns=metadata, data=text)
        if name is None:
            name = "text"
        self.experiment.log({name: text_table}, step=step)
        pass

    def log_table(
        self, key: str,
        data: Any,
        columns: Optional[List] = None,
        dataframe: Any = None,
        step: Optional[int] = None,
    ):
        """
        Log a table to wandb.

        :param key: key of the table.
        :type key: str
        :param data: data of the table.
        :type data: list
        :param columns: columns of the table.
        :type columns: list
        :param dataframe: dataframe of the table.
        :type dataframe: pandas.DataFrame
        :param step: step of the table.
        :type step: int
        """
        metrics = {key: wandb.Table(columns=columns, data=data, dataframe=dataframe)}
        self.log_metrics(metrics, step)

    def log_image(self, image_data: Any, name:str, step: Optional[int] = None, **kwargs):
        """
        Log an image to wandb.
        """
        # check image_data is tensor, pil, numpy arry or str
        if isinstance(image_data, (torch.Tensor, np.ndarray, Image)):
            image_data = wandb.Image(image_data, **kwargs)
        return self.experiment.log({name: image_data}, step)

    def log_images(self, image_data: List[Any], name: str, step: Optional[int] = None, **kwargs):
        if not isinstance(image_data, list):
            raise TypeError(f'Expected a list as "images", found {type(image_data)}')
        n = len(image_data)
        for k, v in kwargs.items():
            if len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        kwarg_list = [{k: kwargs[k][i] for k in kwargs.keys()} for i in range(n)]
        image_list = {name: [wandb.Image(img, **kwarg) for img, kwarg in zip(image_data, kwarg_list)]}
        self.experiment.log(image_list, step)

    # Artifact functions -------------------------------------------------------------------
