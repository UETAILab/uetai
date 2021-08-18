"""
Init SummaryWriter object
"""
import os
import zipfile
import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter as TFWriter

from uetai.logger.general import colorstr
from uetai.logger.wandb.wandb_logger import WandbLogger, download_model_artifact

try:
    import wandb

    WANDB_ARTIFACT_PREFIX = "wandb-artifact://"
    assert hasattr(wandb, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

LOGGER = ("wandb", "tb")


class SummaryWriter:
    """Init a custome Logger to write out events, capture run's metadata, version
    weight or dataset artifact and summary to Tensorboard event file
    (or Weight & Biases's dashboard).

    .. note::
        - The class updates the contents asynchronously. This allows a training
        program to call methods to add data to the file directly from the training
        loop, without slowing down training.

    .. example::
        .. code::python
        >>> logger = uetai.SummaryWriter(log_dir='demo')
        Tensorboard: run 'pip install wandb' to automatically track \
and visualize runs.
        Tensorboard: Start with 'tensorboard --logdir {self.log_dir}', \
view at http://localhost:6006/
        >>> for idx in range(100):
        >>>     logger.add_scalar('loss', idx)
    """
    def __init__(
        self,
        log_dir: str = None,
        logger: str or List = None,
        opt: argparse.Namespace = None,
    ):
        """
        Initalize a SummaryWriter object to start write out events, metadata or
        log artifacts.

        :param log_dir: Tensorboard save directotry location or Weight & Biases
            project's name, defaults to None
        :type log_dir: str, optional
        :param logger: Select logger (Weight & Bias or Tensorboard),
            must be one of this string 'wandb'; 'tensorboard' or both in a List
        :type logger: str or List, optional
        :param opt: option, defaults to None
        :type opt: argparse.Namespace, optional

        .. note::
            - ``SummaryWriter`` will automatically decide to logging by 'wandb'
            or not if the user did not pass any command to `logger` option while
            initializing. In the case that user does have ``wandb``,
            the ``SummaryWriter`` will activate ``wandb`` function and otherwise.

        :examples:
            .. code::python
            >>> logger = uetai.SummaryWriter(log_dir='demo', logger='tensorboard')
            Tensorboard: run 'pip install wandb' to automatically track \
and visualize runs.
            Tensorboard: Start with 'tensorboard --logdir {self.log_dir}', \
view at http://localhost:6006/

            .. code::python
            >>> logger_list = ['tensorboard', 'wandb']
            >>> logger = uetai.SummaryWriter(log_dir='demo', logger=logger_list)
            wandb: Currently logged in as: user-name \
    (use `wandb login --relogin` to force relogin)
            wandb: Tracking run with wandb version 0.11.2
            wandb: Syncing run run_name
            wandb: View project at https://wandb.ai/user-name/demo/runs/run_id
        """
        self.log_dir = log_dir
        # self.config     = config # move config to opt (namespace)
        self.opt = opt if opt is not None else None
        # check selected logger is valid
        if logger is not None:
            if isinstance(logger, str):
                assert logger in ('wandb', 'tensorboard'), f"Logger must \
be 'wandb' or 'tensorboard', found {logger}"
                self.logger = 'wandb'
            elif isinstance(logger, List):
                raise Exception("We've not supported this feature yet, please\
try 'wandb' or 'tensorboard'")
#                 self.logger = []
#                 for item in logger:
#                     if item in ('wandb', 'tensorboard'):
#                         self.logger.append(item)
#                     else:
#                         raise Exception(f"Logger must be one of 'wandb' or \
# 'tensorboard', found {item}")
        elif logger is None:
            self.logger = ('wandb' if wandb is not None else 'tensorboard')

        # Init logger
        if self.logger == 'tensorboard':
            self._log_message(
                "run 'pip install wandb' to automatically track and visualize runs."
            )
            self.__init_tensorboard()
        elif self.logger == 'wandb':
            self.__init_wandb()
        # else:
        #     self.__init_tensorboard()
        #     self.__init_wandb()

    def _log_message(
        self,
        message: str,
        prefix: str = None,
    ):
        if prefix is None:
            if isinstance(self.logger, str):
                prefix = (
                    "Weights & Biases: "
                    if self.logger == 'wandb' else "Tensorboard: ")
            else:
                prefix = "Tensorboard and W&B: "

        prefix = colorstr(prefix)
        mess_str = f"{prefix}{message}"
        print(str(mess_str))

    def __init_tensorboard(
        self,
    ):
        self._log_message(
            f"Start with 'tensorboard --logdir {self.log_dir}',"
            "view at http://localhost:6006/"
        )
        self.tensorboard = TFWriter(str(self.log_dir))

    def __init_wandb(
        self,
    ):
        # wandb_artifact_resume = isinstance(self.opt.weights, str)
        # and self.opt.weights.startswith(WANDB_ARTIFACT_PREFIX)
        # run_id = self.opt.weights if not wandb_artifact_resume else None
        self.wandb = WandbLogger(project=self.log_dir)

    def get_logdir(self):
        """Get run's name or log_dir.

        :return: run's name (Weight & Biases) or log_dir (Tensorboard)
        :rtype: str
        """
        return self.log_dir

    def watch_model(
        self,
        model: nn.Module,
        criterion: nn.Module = None,
        log: str = "gradients",
        log_freq: int = 1000,
        idx: int = None,
    ):
        """Calling Wandb API to track model's weights and biases into W&B dashboard.

        :param model: The model to hook, can be a tuple
        :type model: nn.Module
        :param criterion: An optional loss value being optimized, defaults to None
        :type criterion: An optional loss value being optimized, optional
        :param log: One of "gradients", "parameters", "all", or None, \
defaults to "gradients"
        :type log: str, optional
        :param log_freq: log gradients and parameters every N batches, defaults to 1000
        :type log_freq: int, optional
        :param idx: an index to be used when calling `wandb.watch` on multiple models, \
defaults to None
        :type idx: [type], optional

        :return: A model histogram of weights and biases
        :rtype: ``wandb.Graph`` or None

        :raises ValueError: If called before `wandb.init` \
or if any of models is not a torch.nn.Module.

        :example:
            .. code::python
            >>> from torchvision.models as models
            >>> model = models.resnet18()
            >>> # before training process
            >>> logger.watch(model=model, log_freq=10)

        """
        if self.logger == 'wandb':
            model_graph = self.wandb.watch(model, criterion, log, log_freq, idx)
            return model_graph
        self._log_message(
            "Does not support watch model with Tensorboard, please use W&B"
        )
        return None

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int = None,
    ):
        """Adding scalar data to summary with Tensorboard or logging into W&B

        :param tag: Data identifier
        :type tag: str
        :param scalar_value: Scalar from run
        :type scalar_value: float
        :param global_step: The global step in processing, defaults to None
        :type global_step: int, optional

        .. admonition:: See also
            :class: tip

            **add_scalars**

        :example:
            .. code:: python
            >>> # basic usage
            >>> logger.add_scalar(tag='train/loss', scalar_value=0.5)

            >>> for epoch in range(epochs):
            >>>     loss = 1/epoch
            >>>     logger.add_scalar('train/loss',
            >>>             scalar_value=loss,
            >>>             global_step=epoch)

        """
        if self.logger == 'wandb':
            self.wandb.log({tag: scalar_value}, step=global_step)
        else:
            self.tensorboard.add_scalar(tag, scalar_value, global_step)

    def add_scalars(
        self,
        main_tag,
        tag_scalar_dict: Dict[str, Any],
        global_step: int = None,
        walltime: float = None,
    ):
        """Adding scalar data to summary with Tensorboard or logging into W&B

        :param main_tag: Data identifier
        :type main_tag: str
        :param tag_scalar_value: Scalars from run
        :type tag_scalar_value: Dict[str, Any]
        :param global_step: The global step in processing, defaults to None
        :type global_step: int, optional
        :param walltime: Override default walltime (time.time()) seconds \
after epoch of event
        :type walltime: float, optional

        .. admonition:: See also
            :class: tip

            **add_scalar**

        :example:
            .. code:: python
            >>> main_tag = 'train'
            >>> for i in range(10):
            >>>    tag_scalar_dict = {'loss_cls' = i/10,
            >>>                     'loss_bbox' = i/10,}
            >>>    logger.add_scalars(main_tag, tag_scalar_dict, global_step=i)
        """
        if self.logger == 'wandb':
            wb_scalar_dict = {}
            for key, value in tag_scalar_dict.items():
                wb_scalar_dict[str(main_tag + "/" + key)] = value

            self.wandb.log(wb_scalar_dict, step=global_step)
        else:
            self.tensorboard.add_scalars(
                main_tag, tag_scalar_dict, global_step, walltime
            )

    def data_path(
        self, local_path: str, dataset_name: str = None, alias: str = "latest"
    ):
        """Check local dataset path if user are using Tensorboard, otherwise check W&B
        artifact and download (if need). User can pass url, which starts with "http",
        to local_path for download it (and unzip if url ends with ".zip")

        :param local_path: path to local dataset folder or download url
        :type local_path: str
        :param dataset_name: For download W&B dataset artifact
        :type dataset_name: str, optional
        :param alias: Dataset artifact version, defaults to "latest"
        :type alias: str, optional

        :raises Exception: If local path not found or dataset artifact does not exist.
        :return: Path to dataset (downloaded) folder
        :rtype: str

        .. admonition:: See also
            :class: tip

            **log_dataset_artifact**, **download_dataset_artifact**

        :example:
            .. code:: python
            >>> # basic usage (with Tensorboard)
            >>> data_dir = logger.data_path('./dataset/MNIST')

            .. code:: python
            >>> # using `data_path` to download dataset by url
            >>> url = 'https://data.deepai.org/mnist.zip'
            >>> data_dir = logger.data_path(url)

            .. code:: python
            >>> # download dataset artifact (with W&B)
            >>> data_dir = logger.data_path(
            >>>                 local_path='./datasets/',
            >>>                 dataset_name='mnist',
            >>>                 alias='latest')
        """
        if local_path.startswith("http"):
            root = Path("./datasets")
            root.mkdir(parents=True, exist_ok=True)  # create root
            filename = root / Path(local_path).name
            print(f"Downloading {local_path} to {filename}")
            torch.hub.download_url_to_file(local_path, filename)
            local_path = str(filename)
            if local_path.endswith(".zip"):  # unzip
                save_path = root / Path(filename.name[: -len(".zip")])
                print(f"Unziping {filename} to {save_path}")
                with zipfile.ZipFile(filename, "r") as zip_ref:
                    zip_ref.extractall(save_path)
                local_path = str(save_path)
            return local_path

        if Path(local_path).exists():
            # TODO: check whether local_path contains the right version
            return local_path

        if not Path(local_path).exists():
            if self.logger == 'wandb':
                if dataset_name is not None:
                    data_path, _ = self.download_dataset_artifact(
                        dataset_name, alias, save_path=local_path
                    )
                    return data_path

        raise Exception("Dataset not found.")

    def log_dataset_artifact(
        self,
        path: str,
        artifact_name: str,
        dataset_type: str = "dataset",
        dataset_metadata: Dict[str, Any] = None,
    ):
        """Logging dataset as W&B artifact

        :param path: Path to weight local file
        :type path: str
        :param artifact_name: Dataset artifact name
        :type artifact_name: str
        :param dataset_type: Dataset's type, defaults to "dataset"
        :type dataset_type: str, optional
        :param dataset_metadata: Dataset's metadata, defaults to None
        :type dataset_metadata: Dict[str, Any], optional

        :raise Exception: if ``path`` does not exist.
        :return: A W&B dataset artifact
        :rtype: ``wandb.Artifact``

        .. admonition:: See also
            :class: tip

            **download_dataset_artifact**

        :example:
            .. code::python
            >>> logger.log_dataset_artifact('path/to/dataset', 'mnist')
        """
        if self.logger == 'wandb':
            dataset_artifact = self.wandb.log_dataset_artifact(
                path, artifact_name, dataset_type, dataset_metadata
            )
            return dataset_artifact

        self._log_message("Does not support upload dataset to W&B.")
        return None

    def download_dataset_artifact(
        self, dataset_name: str, version: str = "latest", save_path: str = None
    ):
        """Download artifact dataset from W&B

        :param dataset_name: Dataset name
        :type dataset_name: str
        :param version: Dataset version, defaults to latest
        :type version: str, optional
        :param save_path: Path to save dir, defaults to None
        :type save_path: str, optional
        :return: Local dataset path and artifact object
        :rtype: (Path, ``wandb.Artifact``)

        .. admonition:: See also
            :class: tip

            **log_dataset_artifact**

        :example:
            .. code::python
            >>> # basic usage
            >>> data_dir, _ = logger.download_dataset_artifact('mnist', 'v1')

        """
        if self.logger == 'wandb':
            dataset_dir, version = self.wandb.download_dataset_artifact(
                dataset_name=WANDB_ARTIFACT_PREFIX + dataset_name,
                alias=version,
                save_path=save_path,
            )
            return dataset_dir, version
        self._log_message(
            "Please enable wandb not support download dataset artifact from W&B."
        )

        return None, None

    def log_model_artifact(
        self,
        path: str,
        epoch: int = None,
        scores: float or Dict[str, float] = None,
        opt: argparse.Namespace = None,
    ):
        """Logging model weight as W&B artifact

        :param path: Path to weight local file
        :type path: str
        :param epoch: Current epoch, defaults to None
        :type epoch: int, optional
        :param scores: Model score(s) in current epoch, defaults to None
        :type scores: float or Dict[str, float], optional
        :param opt: Comand line arguments to store on artifact, defaults to None
        :type opt: argparse.Namespace, optional

        :return: Models' weight as W&B aritfact
        :rtype: ``wandb.Artifact`` or None

        .. admonition:: See also
            :class: tip

            **save**

        :example:
            .. code::python
            >>> # basic usage
            >>> for epoch in range(epochs):
            >>>     torch.save(model.state_dict(), 'weight.pt')
            >>>     log_model_artifact('weight.pt', epoch)
        """
        if self.logger == 'wandb':
            model_artifact = self.wandb.log_model(path, epoch, scores, opt)
            return model_artifact

        self._log_message("Does not support upload dataset artifact to W&B.")
        return None

    def save(
        self,
        obj,
        path: str,
        epoch: int = None,
        scores: float or Dict[str, float] = None,
    ):
        """Saving model ``state_dict`` and logging into W&B

        :param obj: [description]
        :type obj: nn.Module or list
        :param path: Save path
        :type path: str
        :param epoch: Current epoch, defaults to None
        :type epoch: int, optional
        :param scores: Model score(s) in current epoch, defaults to None
        :type scores: float or Dict[str, float], optional

        .. admonition:: See also
            :class: tip

            **log_model_artifact**

        :example:
            .. code::python
            >>> # basic usage
            >>> for epoch in range(epochs):
            >>>     logger.save(model.state_dict, './weight.pt')
        """
        parent_path = os.path.normpath(os.path.join(path, os.path.pardir))
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)

        if epoch is not None:
            obj["epoch"] = epoch
        if scores is not None:
            if isinstance(scores, float):
                obj["score"] = scores
            elif isinstance(scores, dict):
                for key, value in scores.items():
                    obj[key] = value
        torch.save(obj, path)

        if self.logger == 'wandb':
            self.log_model_artifact(path=path, epoch=epoch, scores=scores)
        else:
            self._log_message(
                f"Saved model in {path}. Using `wandb` to upload model into W&B."
            )

    def download_model_artifact(self, artifact_name: str, alias: str = "latest"):
        """Download model artifact from W&b and extract model run's metadata

        :param artifact_name: W&B artifact name
        :type artifact_name: str
        :param alias: Artifact version, defaults to latest
        :type alias: str, optional
        :return: Artifact path to directory and `wandb.Artifact` object or None, None
        :rtype: str, wandb.Artifact

        .. admonition:: See also
            :class: tip

            **log_model_artifact, log_dataset_artifact, download_dataset_artifact**
        """
        # TODO: extract run's metadata
        if self.logger == 'wandb':
            artifact_dir, artifact = download_model_artifact(
                model_artifact_name=artifact_name, alias=alias
            )
            return artifact_dir, artifact

        self._log_message("Does not support download dataset artifact from W&B.")
        return None, None
