"""
Init SummaryWriter object
"""
import os
import zipfile
import argparse
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

import torch
from torch import nn

from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

from pytorch_lightning.loggers import (
    WandbLogger,
    TensorBoardLogger,
    LightningLoggerBase
)

from uetai.logger.general import colorstr

try:
    import wandb
    WANDB_ARTIFACT_PREFIX = "wandb-artifact://"
    assert hasattr(wandb, "__version__")  # verify package import not local dir
    if os.environ.get("WANDB_API_KEY", None) is None:
        wandb = None
except (ImportError, AssertionError):
    wandb = None


class SummaryWriter(LightningLoggerBase):
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
        log_dir: Optional[str] = None,
        log_tool: Optional[str or List] = None,
        organization: Optional[str] = None,
        opt: argparse.Namespace = None,
    ):
        """
        Initalize a SummaryWriter object to start write out events, metadata or
        log artifacts.

        :param log_dir: Tensorboard save directotry location or Weight & Biases
        project's name, defaults to None
        :type log_dir: str, optional
        :param log_tool: Select logger (Weight & Bias or Tensorboard),
        must be one of this string 'wandb'; 'tensorboard' or both in a List
        :param entity: Log and save artifact in Wandb team, defaults to None
        :type entity: str, optional
        :type log_tool: str or List, optional
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
        super().__init__()
        self.log_dir = log_dir
        self.organization = organization
        self.opt = opt if opt is not None else None

        # check selected logger is valid
        if log_tool is not None:
            if isinstance(log_tool, str):
                assert log_tool in ('wandb', 'tensorboard'), (
                    f"Logger must be 'wandb' or 'tensorboard', found {log_tool}")
                self.log_tool = 'wandb'
            elif isinstance(log_tool, List):
                raise Exception(
                    "We've not supported this feature yet, please"
                    "try 'wandb' or 'tensorboard'"
                )

        elif log_tool is None:
            self.log_tool = ('wandb' if wandb is not None else 'tensorboard')

        # Init logger
        if self.log_tool == 'tensorboard':
            self._log_message(
                "run 'pip install wandb' to automatically track and visualize runs."
            )
            self.__init_tensorboard()
        elif self.log_tool == 'wandb':
            self.__init_wandb()
            self.wandb_run = self.logger.experiment
        # else:
        #     self.__init_tensorboard()
        #     self.__init_wandb()

    def _log_message(self, message: str, prefix: str = None, ):
        if prefix is None:
            if isinstance(self.log_tool, str):
                prefix = (
                    "wandb: "
                    if self.log_tool == 'wandb' else "Tensorboard: ")
            else:
                prefix = "Tensorboard and W&B: "

        prefix = colorstr(prefix)
        mess_str = f"{prefix}{message}"
        print(str(mess_str))

    def __init_tensorboard(self):
        self._log_message(
            f"Start with 'tensorboard --logdir {self.log_dir}',"
            "view at http://localhost:6006/"
        )
        self.logger = TensorBoardLogger(str(self.log_dir))

    def __init_wandb(self):
        # TODO: extract run name, id -> resume run
        if wandb is None:
            self._log_message(
                "`wandb` is not installed yet, "
                "trying to install with pip"
            )

            # attempt to install with `pip install wandb`
            install_package('wandb')
        if self.organization is not None:
            os.environ['WANDB_ENTITY'] = self.organization
        self.logger = WandbLogger(project=str(self.log_dir), log_model=True)

    # Lightning Logger methods =======================================
    @property
    @rank_zero_experiment
    def experiment(self) -> Any:
        return self.logger.experiment

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        return self.logger.log_metrics(metrics, step)

    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs):
        return self.logger.log_hyperparams(params, *args, **kwargs)

    @property
    def name(self) -> str:
        return self.logger.name

    @property
    def version(self) -> Union[int, str]:
        return self.logger.version

    @property
    def save_dir(self) -> Optional[str]:
        return self.logger.save_dir

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.logger.finalize(status)

    # Custom function ================================================
    def get_logdir(self):
        """Get run's name or log_dir.

        :return: run's name (Weight & Biases) or log_dir (Tensorboard)
        :rtype: str
        """
        return self.log_dir

    def watch(
        self,
        model: nn.Module,
        criterion: nn.Module = None,
        log: str = "gradients",
        log_freq: int = 100
    ):
        """Calling Wandb API to track model's weights and biases into W&B dashboard.

        :param model: The model to hook, can be a tuple
        :type model: nn.Module
        :param criterion: An optional loss value being optimized, defaults to None
        :type criterion: nn.Module, optional
        :param log: One of "gradients", "parameters", "all", or None,
        defaults to "gradients"
        :type log: str, optional
        :param log_freq: log gradients and parameters every N batches, defaults to 1000
        :type log_freq: int, optional

        :return: A model histogram of weights and biases
        :rtype: ``wandb.Graph`` or None

        :raises ValueError: If called before `wandb.init`
        or if any of models is not a torch.nn.Module.

        :example:
            .. code::python
            >>> from torchvision.models as models
            >>> model = models.resnet18()
            >>> # before training process
            >>> logger.watch(model=model, log_freq=10)

        """
        if self.log_tool == 'wandb':
            self.wandb_run.watch(
                models=model,
                criterion=criterion,
                log=log,
                log_freq=log_freq
            )
        elif self.log_tool == 'tensorboard':
            self._log_message(
                "Does not support watch model with Tensorboard, please use W&B"
            )

    def data_path(
        self, path: str, dataset_name: str = None, alias: str = "latest"
    ):
        """Check local dataset path if user are using Tensorboard, otherwise check W&B
        artifact and download (if need). User can pass url, which starts with "http",
        to path for download it (and unzip if url ends with ".zip")

        :param path: path to local dataset folder or download url
        :type path: str
        :param dataset_name: For download W&B dataset artifact
        :type dataset_name: str, optional
        :param alias: Dataset artifact version, defaults to "latest"
        :type alias: str, optional

        :raises Exception: If local path not found or dataset artifact does not exist.
        :return: Path to dataset directory
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
            >>>                 path='./datasets/',
            >>>                 dataset_name='mnist',
            >>>                 alias='latest')
        """
        if path.startswith("http"):
            root = Path("./datasets")
            root.mkdir(parents=True, exist_ok=True)  # create root
            filename = root / Path(path).name
            print(f"Downloading {path} to {filename}")
            torch.hub.download_url_to_file(path, filename)
            path = str(filename)
            if path.endswith(".zip"):  # unzip
                save_path = root / Path(filename.name[: -len(".zip")])
                print(f"Unziping {filename} to {save_path}")
                with zipfile.ZipFile(filename, "r") as zip_ref:
                    zip_ref.extractall(save_path)
                path = str(save_path)
            return path

        if Path(path).exists():
            # TODO: check whether `path` contains the right version
            self._log_message(f"Local path to datasets found, return {path}")
            return path

        if not Path(path).exists():
            if self.log_tool == 'wandb':
                self._log_message(
                    "Local path to datasets not found, "
                    "attempt to download datasets from `wandb` project")
                if dataset_name is not None:
                    path, _ = self.download_dataset_artifact(
                        dataset_name, alias, save_path=path
                    )
                    return path

        raise Exception(
            "Dataset not found. Please try using `wandb` to download artifact")

    def log_dataset_artifact(
        self,
        path: str,
        dataset_name: str,
        dataset_type: str = "dataset",
        dataset_metadata: Dict[str, Any] = None,
    ):
        """Logging dataset as W&B artifact

        :param path: Path to weight local file
        :type path: str
        :param dataset_name: Dataset artifact name
        :type dataset_name: str
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
        if self.log_tool == 'wandb':
            dataset_artifact = self._check_and_log_dataset(
                path, dataset_name, dataset_type, dataset_metadata
            )
            return dataset_artifact

        self._log_message("Does not support upload dataset to W&B.")
        return None

    def _check_and_log_dataset(
        self,
        path: str,
        artifact_name: str,
        dataset_type: str = "dataset",
        dataset_metadata: Dict[str, Any] = None,
    ):
        """Log the dataset as W&B artifact

        :param path: Path to dataset artifact dir or file.
        :type path: str
        :param artifact_name: Name of logging dataset
        :type artifact_name: str
        :param dataset_type: Type of logging dataset, defaults to 'dataset'
        :type dataset_type: str, optional
        :param dataset_metadata: Metadata of logging dataset, defaults to None
        :type dataset_metadata: Dict[str, Any], optional
        :raises Exception: If dataset path does not exist
        :return: W&B dataset artifact
        :rtype: ``wandb.Artifact``

        :example:
            .. code::python
            >>> path = './path/to/dir/or/file'
            >>> wandb_logger = WandbLogger()
            >>> wandb_logger.log_dataset_artifact(path, 'raw-mnist', 'dataset')
        """
        if not Path(path).exists():
            raise Exception(f"{path} does not exist.")

        dataset_artifact = wandb.Artifact(
            name=artifact_name,
            type=dataset_type,
            metadata=dataset_metadata,
        )
        if os.path.isdir(path):
            dataset_artifact.add_dir(path)
        elif os.path.isfile(path):
            dataset_artifact.add_file(path)
        print("Upload dataset into Weight & Biases.")
        self.wandb_run.log_artifact(dataset_artifact)
        return dataset_artifact

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
        if self.log_tool == 'wandb':
            dataset_dir, version = self._check_and_download_dataset(
                dataset_name=WANDB_ARTIFACT_PREFIX + dataset_name,
                alias=version,
                save_path=save_path,
            )
            return dataset_dir, version
        self._log_message(
            "Please enable wandb not support download dataset artifact from W&B."
        )

        return None, None

    def _check_and_download_dataset(
        self,
        dataset_name: str,
        alias: str = "latest",
        save_path: str = None,
    ):
        if isinstance(dataset_name, str):  # and path.startswith(WANDB_ARTIFACT_PREFIX)
            # artifact_path = remove_prefix(dataset_name, WANDB_ARTIFACT_PREFIX)
            artifact_path = Path(dataset_name + f":{alias}")
            dataset_artifact = self.wandb_run.use_artifact(
                artifact_path.as_posix().replace("\\", "/")
            )
            assert dataset_artifact is not None, "W&B dataset artifact does not exist"
            data_dir = dataset_artifact.download(save_path)
            return data_dir, dataset_artifact
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
        if self.log_tool == 'wandb':
            model_artifact = self._log_model(path, epoch, scores, opt)
            return model_artifact

        self._log_message("Does not support upload dataset artifact to W&B.")
        return None

    def _log_model(
        self,
        path: str,
        epoch: int = None,
        scores: float or Dict[str, Any] = None,
        opt: argparse.Namespace = None,
    ):
        """Log the model checkpoint as W&B artifact

        :param path: Path to the checkpoint file
        :type path: str
        :param epoch: Curren epoch, defaults to None
        :type epoch: int, optional
        :param scores: Model epoch score(s), defaults to None
        :type scores: floatorDict[str, Any], optional
        :param opt: Comand lien arguments to store on artifact, defaults to None
        :type opt: argparse.Namespace, optional
        :return: Model artifact
        :rtype: ``wandb.Artifact``

        :example:
            .. code::python
            >>> for i in range(epochs):
            >>>     accuracy = i
            >>>     torch.save(model.state_dict(), 'weights.pt')
            >>>     wandb_logger.log_model('weights.pt', epoch, accuracy)
        """
        # TODO: log opt metadata to Wandb run summary
        metadata = (
            {"project": opt.project, "total_epochs": opt.epochs}
            if opt is not None
            else {}
        )
        metadata["epochs_trained"] = epoch + 1 if epoch is not None else None
        if isinstance(scores, float):
            metadata["scores"] = scores
        elif isinstance(scores, dict):
            for key, val in scores.items():
                metadata[key] = val
        else:
            metadata["scores"] = None

        model_artifact = wandb.Artifact(
            "run_" + self.wandb_run.id + "_model", type="model", metadata=metadata
        )
        model_artifact.add_file(str(path))
        # logging
        aliases = ["latest"]
        if epoch is not None:
            aliases.append("epoch " + str(epoch + 1))
        self.wandb_run.log_artifact(model_artifact, aliases=aliases)
        print(f"Saving model on epoch {epoch} done.")
        return model_artifact

    def save_model(
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
            >>>     logger.save_model(model.state_dict, './weight.pt')
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

        if self.log_tool == 'wandb':
            self.log_model_artifact(path=path, epoch=epoch, scores=scores)
        elif self.log_tool == 'tensorboard':
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
        if self.log_tool == 'wandb':
            artifact_dir, artifact = self._check_and_download_model(
                model_artifact_name=artifact_name, alias=alias
            )
            return artifact_dir, artifact

        self._log_message("Does not support download dataset artifact from W&B.")
        return None, None

    def _check_and_download_model(
        self,
        model_artifact_name: str = None,
        alias: str = "latest",
    ):
        """Download the model checkpoint as W&B artifact

        :param model_artifact_name: The name of model that will be downloaded,
        defaults to None
        :type model_artifact_name: str, optional
        :param alias: Version of the model that will be downloaded,
        defaults to 'latest'
        :type alias: str, optional
        :return: Path to model and ``wandb.Artifact`` corresponds to it
        :rtype: (str, ``wandb.Artifact``)
        """
        if isinstance(model_artifact_name, str):
            model_artifact = wandb.use_artifact(model_artifact_name + f":{alias}")
            assert model_artifact is not None, "W&B model artifact doesn not exist"
            model_dir = model_artifact.download()
            return model_dir, model_artifact

        return None, None
