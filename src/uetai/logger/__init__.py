# Init logging object
import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter as TFWriter

from uetai.logger.general import colorstr
from uetai.logger.wandb.wandb_logger import WandbLogger

try:
    import wandb

    WANDB_ARTIFACT_PREFIX = "wandb-artifact://"
    assert hasattr(wandb, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

LOGGER = ("wandb", "tb")


class SummaryWriter:
    """
    A custome Logger writes out events, capture run's metadata, version
    weight or dataset artifact and summary to Tensorboard event file
    (or Weight & Biases's dashboard)

    :param opt: option, defaults to None
    :type opt: argparse.Namespace, optional
    :param log_dir: Tensorboard save directotry location or Weight & Biases
        project's name, defaults to None
    :type log_dir: str, optional

    .. note::
        The class updates the contents asynchronously. This allows a training program to
        call methods to add data to the file directly from the training loop, without
        slowing down training.

    .. tip:: See also
        <something in here>
            Callback for doing something

    :examples: .. code:: python
        >>> # use Tensorboard
        >>> logger = uetai.SummaryWriter(log_dir='demo')
        Tensorboard: run 'pip install wandb' to automatically track and visualize runs.
        Tensorboard: Start with 'tensorboard --logdir {self.log_dir}', \
view at http://localhost:6006/

        >>> # use Weight & Biases
        >>> logger = uetai.SummaryWriter(log_dir='demo')
        wandb: Currently logged in as: user-name \
(use `wandb login --relogin` to force relogin)
        wandb: Tracking run with wandb version 0.11.2
        wandb: Syncing run desert-firefly-63
        wandb: View project at https://wandb.ai/user-name/demo
    """

    def __init__(self, opt: argparse.Namespace = None, log_dir: str = None,):
        self.log_dir = log_dir
        # self.config     = config # move config to opt (namespace)
        self.opt = opt if opt is not None else None
        self.use_wandb = (wandb is not None) and os.environ.get(
            "WANDB_API_KEY", None
        ) is not None
        self.log_prefix = "Weights & Biases: " if self.use_wandb else "Tensorboard: "
        # Message
        if not self.use_wandb:
            self.log_message(
                "run 'pip install wandb' to automatically track and visualize runs."
            )

        if self.use_wandb:
            self.__init_wandb()
        else:
            self.__init_tensorboard()

    def log_message(self, message: str, prefix: str = None):
        if prefix is None:
            prefix = self.log_prefix

        prefix = colorstr(prefix)
        s = f"{prefix}{message}"
        print(str(s))

    def __init_tensorboard(
        self,
    ):
        self.log_message(
            f"Start with 'tensorboard --logdir {self.log_dir}', view at http://localhost:6006/"
        )
        self.tensorboard = TFWriter(str(self.log_dir))

    def __init_wandb(
        self,
    ):
        # wandb_artifact_resume = isinstance(self.opt.weights, str)
        # and self.opt.weights.startswith(WANDB_ARTIFACT_PREFIX)
        # run_id = self.opt.weights if not wandb_artifact_resume else None
        self.wandb = WandbLogger(self.opt)

    def get_logdir(self):
        """Return directory"""
        return self.log_dir

    def watch_model(
        self, model: nn.Module, criterion=None, log="gradients", log_freq=1000, idx=None
    ):
        if self.use_wandb:
            self.wandb.watch(model, criterion, log, log_freq, idx)
        else:
            self.log_message(
                "Does not support watch model with Tensorboard, please use wandb"
            )

    def add_scalar(self, tag: str, scalar_value, global_step=None):
        """
        log(
            data: Dict[str, Any],
            step: int = None,
            commit: bool = None,
            sync: bool = None
        ) -> None
        """
        if self.use_wandb:
            self.wandb.log({tag: scalar_value}, step=global_step)
        else:
            self.tensorboard.add_scalar(tag, scalar_value, global_step)

    def add_scalars(
        self, main_tag, tag_scalar_dict: dict, global_step=None, walltime=None
    ):
        """
        Usage:
            main_tag = 'train'
            for i in range(10):
                tag_scalar_dict = {
                    'loss_cls' = i/10,
                    'loss_bbox' = i/10,
                }
                add_scalars(main_tag, tag_scalar_dict, global_step=i)

        """
        if self.use_wandb:
            wb_scalar_dict = {}
            for key, value in tag_scalar_dict.items():
                wb_scalar_dict[str(main_tag + "/" + key)] = value

            self.wandb.log(wb_scalar_dict, step=global_step)
        else:
            self.tensorboard.add_scalars(
                main_tag, tag_scalar_dict, global_step, walltime
            )

    def data_path(self, local_path: str, dataset_name: str, version: str = "latest"):
        """
        This function will return the local dataset path in local environment,
         return the wandb datapath in wandb environment.
        @param local_path:
        @param dataset_name:
        @param version:
        @return:
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
                import zipfile

                with zipfile.ZipFile(filename, "r") as zip_ref:
                    zip_ref.extractall(save_path)
                local_path = str(save_path)
            return local_path

        if Path(local_path).exists():
            # TODO: check whether local_path contains the right version
            return local_path

        elif not Path(local_path).exists():
            if self.use_wandb:
                data_path, _ = self.download_dataset_artifact(dataset_name, version)
                return data_path

        else:
            raise Exception("Dataset not found.")

    def log_dataset_artifact(
        self,
        path: str,
        artifact_name: str,
        dataset_type: str = "dataset",
        dataset_metadata: dict = None,
    ):
        """
        Log dataset as W&B artifact.

        Args:
            path (str): Path to weight local file
            artifact_name (str): Name represents the dataset artifact
            dataset_type (str): Datasets' type
            dataset_metadata (dict): Datasets' metadata
        """
        if self.use_wandb:
            self.wandb.log_dataset_artifact(
                path, artifact_name, dataset_type, dataset_metadata
            )
            pass
        else:
            self.log_message("Does not support upload dataset to Weight & Biases.")

    def download_dataset_artifact(
        self, dataset_name: str, version: str, save_path: str = None
    ):
        """
        Download dataset artifact from Weight & Biases

        Agrs:
            dataset_name (str): Artifact name
            version (str): artifact version

        Returns:
            (Path, wandb.Artifact) Local dataset path, Artifact object
        """
        if self.use_wandb:
            dataset_dir, version = self.wandb.download_dataset_artifact(
                path=WANDB_ARTIFACT_PREFIX + dataset_name,
                alias=version,
                save_path=save_path,
            )
            return dataset_dir, version
        else:
            self.log_message(
                "Please enable wandb not support download dataset artifact from Weight & Biases database."
            )

        return None, None

    def log_model_artifact(
        self,
        path: str,
        epoch: int = None,
        scores: float or dict = None,
        opt: argparse.Namespace = None,
    ):
        """
        Logging the model as W&B artifact.

        Args:
            path (str): Path to weight local file
            epoch (int): Current epoch number
            scores (float/dict): score(s) represents for current epoch
            opt (namespace): Comand line arguments to store on artifact
        """
        if self.use_wandb:
            self.wandb.log_model(path, epoch, scores, opt)
        else:
            self.log_message(
                "Does not support upload dataset artifact to Weight & Biases."
            )

    def save(self, obj, path: str, epoch: int = None, scores: float or dict = None):
        """Saving 

        :param obj: [description]
        :type obj: [type]
        :param path: [description]
        :type path: str
        :param epoch: [description], defaults to None
        :type epoch: int, optional
        :param scores: [description], defaults to None
        :type scores: floatordict, optional
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

        if self.use_wandb:
            self.log_model_artifact(path=path, epoch=epoch, scores=scores)
        else:
            self.log_message(
                f"Saved model in {path}. Using `wandb` to upload model into Weight & Biases."
            )

    def download_model_artifact(self, artifact_name: str = None, alias: str = None):
        """
        Download model artifact from Weight & Biases and extract model run's metadata

        Args:
            artifact_name (str): Artifact name
            alias (str): artifact version

        Returns:
            (Path, wandb.Artifact)
        """
        # TODO: extract run's metadata
        if self.use_wandb:
            artifact_dir, artifact = self.wandb.download_model_artifact(
                path=artifact_name, alias=alias
            )
            return artifact_dir, artifact
        else:
            self.log_message(
                "Does not support download dataset artifact from Weight & Biases database."
            )

        return None, None
