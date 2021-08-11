"""
Wandb logger and utilities
"""
import os
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
from torch import nn

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def remove_prefix(string: str, prefix: str = WANDB_ARTIFACT_PREFIX):
    """Remove ``prefix`` from string

    :param string: String need to remove prefix
    :type string: str
    :param prefix: Prefix, defaults to WANDB_ARTIFACT_PREFIX
    :type prefix: prefix, optional
    :return: String after remove prefix
    :rtype: str
    """
    return string[len(prefix):]


def get_run_info(run_path):
    """Get W&B run's information

    :param run_path: Path to W&B run
    :type run_path: str
    :return: W&B entity, project, run id, model artifact name
    :rtype: (str, str, str, ``wandb.Artifact``)
    """
    run_path = Path(remove_prefix(run_path, WANDB_ARTIFACT_PREFIX))
    run_id = run_path.stem
    project = run_path.parent.stem
    entity = run_path.parent.parent.stem
    model_artifact_name = 'run_' + run_id + '_model'
    return entity, project, run_id, model_artifact_name


def create_dataset_artifact(
        path: str,
        dataset_name: str = 'dataset',
        dataset_type: str = 'dataset',):
    """Create a dataset W&B artifact

    :param path: Path to dataset dir/file
    :type path: str
    :param dataset_name: Dataset name, defaults to 'dataset'
    :type dataset_name: str, optional
    :param dataset_type: Dataset type, defaults to 'dataset'
    :type dataset_type: str, optional
    :return: W&B Dataset artifact
    :rtype: ``wandb.Artifact``

    :example:
        .. code::python
        >>> wandb_logger.create_dataset_artifact(
        >>>                         path='path/to/data',
        >>>                         dataset_name='image-128',
        >>>                         dataset_type='raw dataset')
    """
    artifact = wandb.Artifact(name=dataset_name, type=dataset_type)
    if isinstance(path, str) and os.path.exists(path):
        if os.path.isdir(path):
            artifact.add_dir(path)
        elif os.path.isfile(path):
            artifact.add_file(path)

    return artifact


def download_model_artifact(
        model_artifact_name: str = None,
        alias: str = 'latest',):
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


class WandbLogger:
    """
    Log training runs, datasets, models, and predictions to Weights & Biases.
    This logger sends information to W&B at wandb.ai.

    By default, this information
    includes hyperparameters, system configuration and metrics, model metrics,
    and basic data metrics and analyses.

    """
    def __init__(self, opt: argparse.Namespace = None, job_type: str = "Training"):
        """
        Initialize a Wandb runs or resume a previous run to upload dataset
        (if `opt.upload_dataset` is True) or monitoring training processes.

        :param opt: Comandline of this run, default to None
        :type opt: argparse.Namespace, optional
        :param job_type: Name of this run, default to Training
        :type opt: str, optional

        .. note::
            This object is currently under development. Some function might be not
            sustainable

        :example:
            .. code::python
            >>> wandb_logger = uetai.logger.wandb.WandbLogger()
        """
        self.run_id = None
        self.job_type = job_type
        self.wandb, self.wandb_run = wandb, None if not wandb else wandb.run

        if (isinstance(self.run_id, str) and
                self.run_id.startswith(WANDB_ARTIFACT_PREFIX)):
            # TODO: resume run
            entity, project, run_id, model_artifact_name = get_run_info(self.run_id)
            self.model_artifact_name = WANDB_ARTIFACT_PREFIX + model_artifact_name

            assert wandb, 'install wandb to resume wandb runs'
            # Resume wandb-artifact:// runs
            self.wandb_run = wandb.init(job_type=job_type,
                                        id=self.run_id,
                                        project=opt.project or 'uetai-logger',
                                        entity=opt.entity or 'uet-ailab',
                                        resume='allow', )

        elif self.wandb:
            self.wandb_run = wandb.init(config=opt,
                                        resume="allow",
                                        project=opt.project or 'uetai-logger',
                                        entity=opt.entity or 'uet-ailab',
                                        job_type=job_type,
                                        id=self.run_id,
                                        allow_val_change=True,
                                        ) if not wandb.run else wandb.run

    def log(self, log_dict: Dict[str, Any], step: int = None):
        """Log a dict to the global run's history.

        Use `wandb.log` to log data from runs, such as scalars, images, video,
        histograms, and matplotlib plots.

        :param log_dict: A dict of serializable python objects i.e `str`,
        `ints`, `floats`, `Tensors`, `dicts`, or `wandb.data_types`.
        :type log_dict: Dict[str, Any]
        :param step: The global step in processing. This persists
        any non-committed earlier steps but defaults to not committing the
        specified step., defaults to None
        :type step: int, optional

        .. admonition:: See also
            :class: tip
            <https://docs.wandb.ai/ref/python/log>
        """
        if self.wandb_run:
            self.wandb_run.log(data=log_dict, step=step)

    def watch(
        self,
        model: nn.Module or Tuple,
        criterion: nn.Module = None,
        log: str = "gradients",
        log_freq: int = 1000,
        idx: int = None
    ):
        """Hooks into the torch model to collect gradients and the topology.

        :param model: The model to hook, can be a tuple
        :type model: nn.Module or Tuple
        :param criterion: An optional loss value being optimized, defaults to None
        :type criterion: nn.Module, optional
        :param log: One of "gradients", "parameters", "all", or None,
        defaults to "gradients"
        :type log: str, optional
        :param log_freq: log gradients and parameters every N batches, defaults to 1000
        :type log_freq: int, optional
        :param idx: an index to be used when calling wandb.watch on multiple models,
        defaults to None
        :type idx: int, optional
        :return: The graph object that will populate after the first backward pass
        :rtype: ``wandb.Graph``
        """
        if self.wandb_run:
            return self.wandb_run.watch(model, criterion, log, log_freq, idx)
        raise Exception('Wandb run not found. Please init wandb before call watch')

    def check_and_upload_dataset(self, opt: argparse.Namespace = None):
        """
        Check if the dataset format is compatible and upload it as W&B artifact

        Args:
            opt (namespace): Commandline arguments for current run

        Returns:
            Updated dataset info dictionary where local dataset paths are replaced
              by WAND_ARFACT_PREFIX links.
        """
        # TODO: upload dataset by sperate scipt
        assert wandb, 'Install wandb to upload dataset'
        config_path = self.log_dataset_artifact(opt.data, opt.project)
        print("Created dataset config file ", config_path)
        with open(config_path, encoding='ascii', errors='ignore') as f:
            wandb_data_dict = yaml.safe_load(f)
        return wandb_data_dict

    # def setup_training(self, opt: argparse.Namespace = None):
    #     """
    #     Setup the necessary processes for training models:
    #     - Attempt to download model checkpoint and dataset artifact (if opt.weights
    #       or opt.dataset starts with WANDB_ARTIFACT_PREFIX)
    #     - Update data_dict, to contain info of previous run if resumed and the paths
    #       of dataset artifact if downloaded
    #     """
    #     self.log_dict = {}
    #     if isinstance(opt.weights, str) and
    #           opt.weights.startswith(WANDB_ARTIFACT_PREFIX):
    #         model_dir, _ = self.download_model_artifact(self.model_artifact_name)
    #         if model_dir:
    #             self.weights = Path(model_dir)
    #             config = self.wandb_run.config

    #     else:
    #         data_dict = self.data_dict
    #     if self.val_artifact is None:
    #         self.train_artifact_path, \
    #           self.train_artifact = self.download_dataset_artifact(
    #                                                   data_dict.get('train'),
    #                                                   opt.artifact_alias)
    #         self.val_artifact_path, \
    #           self.val_artifact = self.download_dataset_artifact(
    #                                                   data_dict.get('val'),
    #                                                   opt.artifact_alias)

    #     if self.train_artifact_path is not None:
    #         train_path = Path(self.train_artifact_path)
    #         data_dict['train'] = str(train_path)

    #     if self.val_artifact_path is not None:
    #         val_path = Path(self.val_artifact_path)
    #         data_dict['val'] = str(val_path)

    #     train_from_artifact = self.train_artifact_path is not None and
    #                                          self.val_artifact_path is not None
    #     if train_from_artifact:
    #         self.data_dict = data_dict

    def download_dataset_artifact(
        self,
        dataset_name: str,
        alias: str = 'latest',
        save_path: str = None,
    ):
        """Download dataset artifact from W&B

        :param path: Dataset artifact name
        :type path: str
        :param alias: alias of the artifact to be download, defaults to 'latest'
        :type alias: str, optional
        :param save_path: Path to save the downloaded, defaults to None
        :type save_path: str, optional
        :return: Path of the downloaded dataset and it's corresponding
        artifact object if dataset is found
        :rtype: (str, ``wandb.Artifact``)
        """
        if isinstance(dataset_name, str):  # and path.startswith(WANDB_ARTIFACT_PREFIX)
            # artifact_path = remove_prefix(dataset_name, WANDB_ARTIFACT_PREFIX)
            artifact_path = Path(dataset_name + f':{alias}')
            dataset_artifact = self.wandb_run.use_artifact(
                                    artifact_path.as_posix().replace("\\", "/")
                                    )
            assert dataset_artifact is not None, "W&B dataset artifact does not exist"
            data_dir = dataset_artifact.download(save_path)
            return data_dir, dataset_artifact
        return None, None

    def log_dataset_artifact(self,
                             path: str,
                             artifact_name: str,
                             dataset_type: str = 'dataset',
                             dataset_metadata: dict = None):
        """
        Log the dataset as W&B artifact and return the new data file with W&B links

        Args:
            path (str): Path to dataset artifact dir/file.
            artifact_name (str): 
            dataset_type (str): 

        Example:
            path = './path/to/dir/or/file'
            wandb_logger = WandbLogger()
            wandb_logger.log_dataset_artifact(path, 'raw-mnist', 'dataset')
        """
        if not Path(path).exists():
            raise Exception(f'{path} does not exist.')

        dataset_artifact = wandb.Artifact(name=artifact_name,
                                          type=dataset_type,
                                          metadata=dataset_metadata, )
        if os.path.isdir(path):
            dataset_artifact.add_dir(path)
        elif os.path.isfile(path):
            dataset_artifact.add_file(path)
        print('Upload dataset into Weight & Biases.')
        self.wandb_run.log_artifact(dataset_artifact)
        return dataset_artifact

    def log_model(
        self, path: str,
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
        metadata = {'project': opt.project,
                    'total_epochs': opt.epochs} if opt is not None else {}
        metadata['epochs_trained'] = epoch + 1 if epoch is not None else None
        if isinstance(scores, float):
            metadata['scores'] = scores
        elif isinstance(scores, dict):
            for key, val in scores.items():
                metadata[key] = val
        else:
            metadata['scores'] = None

        model_artifact = wandb.Artifact(
                                'run_' + self.wandb_run.id + '_model',
                                type='model',
                                metadata=metadata)
        model_artifact.add_file(str(path))
        # logging
        aliases = ['latest']
        if epoch is not None:
            aliases.append('epoch ' + str(epoch + 1))
        self.wandb_run.log_artifact(model_artifact, aliases=aliases)
        print(f"Saving model on epoch {epoch} done.")
        return model_artifact
