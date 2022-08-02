"""Custom Wandb logger by UETAI."""
import os
import logging
from argparse import Namespace
from typing import Any, Optional, Dict, Union, List

import pandas as pd
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
            **kwargs
        )
        self._wandb_init.update(**kwargs)
        self.experiment = self._init_experiment(**self._wandb_init)
        self._experiment_path = self._set_experiment_path()
        self._train_table_summary, self._valid_table_summary = self._set_visualize_table()

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

    @experiment.setter
    def experiment(self, value):
        self._experiment = value

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

    def _set_experiment_path(self) -> str:
        """Set experiment folder.
        :return: experiment folder.
        """
        experiment_path = os.path.join(_SAVING_PATH, self.experiment.id)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        return experiment_path

    def _set_visualize_table(self, ):
        self._train_table_summary = []
        self._valid_table_summary = []

        return self._train_table_summary, self._valid_table_summary

    def visualize_table(self, is_train=True):
        if is_train:
            return self._train_table_summary
        return self._valid_table_summary

    @property
    def save_dir(self) -> Optional[str]:
        """Get experiment save directory.
        :return: Experiment save directory.
        """
        return self._experiment_path

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
        return self.experiment.watch(model, criterion=criterion, log=log, log_freq=log_freq, log_graph=log_graph)

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

    def log_image(self, image_data: Any, name: str, step: Optional[int] = None, **kwargs):
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
    def log_artifact(
            self,
            artifact_name: str,
            artifact_path: str,
            artifact_type: str = 'Artifact',
            vis_table: Union[bool, wandb.Table] = False,
            auto_profiling: Union[bool, pd.DataFrame] = False,
    ):
        """Log artifact to dashboard. The argument `vis_table` is True
        or an `wandb.Table` object has passed into the function, it will automatically
        create visualize table for summary dataset. On the other hands, if `auto_profiling` is True,
        the pandas_profiling will automatically generate data's profile.

        .. warning::

            - Only image and text are currently supported to use `vis_table`
            - User should only pass `pandas.Dataframe` into `auto_profiling`

        There are 2 ways to generate visualization table for image and text: directly by passing
        preprocessed table into `vis_table` argument or pass the `True` flag for generate table for artifact.
        .. code-block:: python

            # visualize image/text
            # by `True` flag
            logger.log_artifact(artifact_name=’cifar10’,
                                artifact_path=’data/cifar10’,
                                vis_table=True) # default: False

            # by passing preprocessed table
            table = uetai.logger.visualize_table(dataset)
            data_artifact = logger.log_artifact(artifact_name=’cifar10’,
                                                artifact_path=’data/cifar10’
                                                vis_table=table)
        .. note::

            See more: uetai.data.visualize_table()

        .. code-block:: python

            # profiling tabular
            housing = pd.read_csv("housing.csv")
            logger.log_artifact(artifact_name=’housing’,
            artifact_path=’housing.csv’,
            auto_profiling=True) # default: False

        :param artifact_name: Name of artifact
        :type artifact_name: str
        :param artifact_path: Path to artifact
        :type artifact_path: str
        :param artifact_type: Type of artifact
        :type artifact_type: Optional[str], default is 'Artifact'
        :param vis_table: Generate visualization table
        :type vis_table: bool or wandb.Table
        :param auto_profiling: Generate profile for tabular data
        :type auto_profiling: bool or pandas.DataFrame
        """
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"{artifact_path} does not exist.")

        # create artifact
        artifact = wandb.Artifact(artifact_name, type=artifact_type)

        if os.path.isfile(artifact_path):
            artifact.add_file(local_path=artifact_path)
        else:
            artifact.add_dir(local_path=artifact_path)

        if vis_table:  # vis_table is True or not empty
            if isinstance(vis_table, (bool, wandb.Table)):
                table = self._gen_visualization_table(vis_table)
                #  log table with artifact
                artifact.add(table, "data_visualization")
            else:
                raise TypeError(f'Visualization table does not support {type(vis_table)} type')

        if auto_profiling:
            if isinstance(auto_profiling, (bool, pd.DataFrame)):
                try:
                    from pandas_profiling import ProfileReport
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(f'Pandas profiling was not installed, '
                                              'run `pip install pandas_profiling` '
                                              'to install it')

                if isinstance(auto_profiling, pd.DataFrame):
                    profile = ProfileReport(auto_profiling, title=self.project_name)
                else:
                    if artifact_path[-3:] == 'csv':
                        dataframe = pd.read_csv(artifact_path)
                    else:
                        raise TypeError(f'Expect file csv, got {artifact_path[-3:]}')
                    profile = ProfileReport(dataframe, title=self.project_name)
                save_path = os.path.join(self._experiment_path, 'profile_report.html')
                profile.to_file(save_path)

                profile_table = wandb.Table(columns=['report'])
                profile_table.add_data(wandb.Html(open(save_path)))
                artifact.add(profile_table, "profiling")

            else:
                raise TypeError(f'Auto profiling does not supported {type(vis_table)} type')

        self.experiment.log_artifact(artifact)
        return artifact

    @staticmethod
    def _gen_visualization_table(table_or_path: Union[str, wandb.Table], map2int: List = None):
        """
        Supported data format:
        |--dir
            |-- 0.jpg
            |-- 4.jpg
            |-- 5.jpg

        """
        if isinstance(table_or_path, wandb.Table):
            return table_or_path

        table = wandb.Table(columns=['id', 'label', 'media'])
        if os.path.exists(table_or_path):
            listfile = os.listdir(table_or_path)
            for idx, file in enumerate(listfile):
                label = file.removesuffix(".jpg").removesuffix(".png")
                if map2int is not None:
                    label = map2int[int(label)]
                table.add_data(idx, label, wandb.Image(os.path.join(table_or_path, file)))

        else:
            raise FileNotFoundError(f'{table_or_path} not found')

        return table

    def monitor(self,
                task: str,
                inputs: torch.Tensor,
                ground_truths: torch.Tensor,
                output: torch.Tensor,
                map2int: List,
                epoch: int = 0,
                log_n_items: int = 10,
                log_every_n_epoch: int = 2,
                is_train: bool = True,
                ):
        """
        Running prediction monitor for specific task (currently only support 'image_classify')

        .. note::
            See also: uetai.callbacks

        .. code-block:: python

            for idx, (input, target) in enumerate(dataloader):
                ...
                preds = model(input)
                ...
                logger.monitor(task=’image_classify’, inputs=input,
                ground_truths=target,
                predictions=preds,
                log_n_items=100)

        :param task: Logging task (only support "image_classify")
        :type task: str
        :param inputs: input of model
        :type inputs: torch.Tensor
        :param ground_truths: label of inputs
        :type ground_truths: torch.Tensor
        :param output: output of model
        :type output: torch.Tensor
        :param map2int: map function from int to label
        :type map2int: List
        :param epoch: epoch
        :type epoch: int
        :param log_n_items: Log n items
        :type log_n_items: int
        :param log_every_n_epoch: Log every n epoch
        :type log_every_n_epoch: int
        :param is_train: log train summary or valid/test summary
        :type is_train: bool
        """
        table = self.visualize_table(is_train)
        if epoch % log_every_n_epoch != 0:
            return

        if task == 'image_classify':
            for i in range(inputs.shape[0]):  # batch-size
                idx = f"{epoch}_{i}"

                pred = torch.max(output.detach(), dim=1)[1]

                cache_data = [idx, wandb.Image(inputs[i])]
                if map2int:
                    _gt = map2int[ground_truths[i]]
                    _pred = map2int[pred[i]]
                    cache_data.extend([_gt, _pred])

                for x in output[i]:
                    cache_data.append(x.item())

                # cache_data.append(cache_data)
                table.append([*cache_data])

                if i % log_n_items == 0:
                    break

        else:
            raise ValueError(f'Task {task} have not supported yet.')

        columns = [f'score_{x}' for x in map2int]
        columns = ['id', 'image', 'ground_truth', 'prediction'] + columns

        wb_table = wandb.Table(columns=columns, data=table)
        self.experiment.log({'table': wb_table})

    def shap_summary_plot(self, explainer, X_test, attributes: List = None):
        """

        """
        try:
            import shap
            import matplotlib.pyplot as plt
            # from shap.explainers._explainer import Explainer

        except ModuleNotFoundError:
            raise ModuleNotFoundError('package shap not found, run `pip install shap` to install it')

        # if isinstance(explainer, Explainer):
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, feature_names=attributes, show=False)
        save_file = os.path.join(self._experiment_path, 'shap_result.png')
        plt.savefig(save_file, dpi=200, bbox_inches='tight')

        self.experiment.log({'shap explain': wandb.Image(save_file)})


# Utilities -------------------------------------------------------------------
def visualization_table(dataset, log_n_items: int = 100):
    """
    Create visualization table directly from variables

    .. code-block:: python

        table = uetai.data.visualize_table(dataset)
        data_artifact = logger.log_artifact(artifact_name=’cifar10’,
        artifact_path=’data/cifar10’
        vis_table=table)

    :param dataset: Dataset for visualize
    :type dataset: torch.DataLoader
    :param log_n_items: Number of item for logging
    :type log_n_items: int

    """
    table = wandb.Table(columns=['id', 'label', 'media'])

    for idx, inputs in enumerate(dataset):
        img, label = inputs[0], inputs[1]
        table.add_data(idx, label, wandb.Image(img))

        if log_n_items & idx >= log_n_items:
            break

    return table
