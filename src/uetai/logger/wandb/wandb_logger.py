import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from torch import nn

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def remove_prefix(from_string, prefix=WANDB_ARTIFACT_PREFIX):
    return from_string[len(prefix):]


def get_run_info(run_path):
    run_path = Path(remove_prefix(run_path, WANDB_ARTIFACT_PREFIX))
    run_id = run_path.stem
    project = run_path.parent.stem
    entity = run_path.parent.parent.stem
    model_artifact_name = 'run_' + run_id + '_model'
    return entity, project, run_id, model_artifact_name


class WandbLogger:
    """
    Log training runs, datasets, models, and predictions to Weights & Biases.
    This logger sends information to W&B at wandb.ai. 
    
    By default, this information
    includes hyperparameters, system configuration and metrics, model metrics,
    and basic data metrics and analyses.

    """

    def __init__(self, opt: argparse.Namespace = None, job_type="Training"):
        """
        Initialize a Wandb runs or resume a previous run to upload dataset 
        (if `opt.upload_dataset` is True) or monitoring training processes.

        args:
            opt (namespace): Commandline arguments for this run
            run_id (str): Run ID of W&B run to be resumed
            job_type (str): Set job_type for this run
        """
        self.job_type = job_type
        # TODO: resume
        self.run_id = None
        self.wandb, self.wandb_run = wandb, None if not wandb else wandb.run

        if isinstance(self.run_id, str) and self.run_id.startswith(WANDB_ARTIFACT_PREFIX):
            entity, project, run_id, model_artifact_name = get_run_info(self.run_id)
            self.model_artifact_name = WANDB_ARTIFACT_PREFIX + model_artifact_name

            assert wandb, 'install wandb to resume wandb runs'
            # Resume wandb-artifact:// runs
            self.wandb_run = wandb.init(job_type=job_type,
                                        id=self.run_id,
                                        project=opt.project if opt is not None else 'mlops-wandb-demo',
                                        entity=opt.entity if opt is not None else None,
                                        resume='allow', )

        elif self.wandb:
            self.wandb_run = wandb.init(config=opt,
                                        resume="allow",
                                        project=opt.project if opt is not None else 'mlops-wandb-demo',
                                        entity=opt.entity if opt is not None else None,
                                        job_type=job_type,
                                        id=self.run_id,
                                        allow_val_change=True) if not wandb.run else wandb

    def log(self, log_dict: Dict[str, Any], step: int = None):
        """
        Log a dict to the global run's history.

        Use `wandb.log` to log data from runs, such as scalars, images, video,
        histograms, and matplotlib plots.

        Arguments:
            log_dict (dict): A dict of serializable python objects i.e `str`,
              `ints`, `floats`, `Tensors`, `dicts`, or `wandb.data_types`.
            step (int): (integer, optional) The global step in processing. This persists
              any non-committed earlier steps but defaults to not committing the
              specified step.
        """
        if self.wandb_run:
            self.wandb_run.log(data=log_dict, step=step)

    def watch(self, model: nn.Module, criterion=None, log="gradients", log_freq=1000, idx=None):
        """
        Hooks into the torch model to collect gradients and the topology.

        Should be extended to accept arbitrary ML models.
        
        Args:
            @models (torch.nn.Module): The model to hook, can be a tuple
            @criterion (torch.F): An optional loss value being optimized
            @log (str): One of "gradients", "parameters", "all", or None
            @log_freq: (int) log gradients and parameters every N batches
            @idx: (int) an index to be used when calling wandb.watch on multiple models
            @log_graph: (boolean) log graph topology
        
        Returns:
            `wandb.Graph` The graph object that will populate after the first backward pass
        """
        if self.wandb_run:
            self.wandb_run.watch(model, criterion, log, log_freq, idx)

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

    def setup_training(self, opt: argparse.Namespace = None):
        """
        Setup the necessary processes for training models:
        - Attempt to download model checkpoint and dataset artifact (if opt.weights 
          or opt.dataset starts with WANDB_ARTIFACT_PREFIX)
        - Update data_dict, to contain info of previous run if resumed and the paths 
          of dataset artifact if downloaded
        """
        self.log_dict = {}
        if isinstance(opt.weights, str) and opt.weights.startswith(WANDB_ARTIFACT_PREFIX):
            model_dir, _ = self.download_model_artifact(self.model_artifact_name)
            if model_dir:
                self.weights = Path(model_dir)
                config = self.wandb_run.config

                # TODO: define opt format

        else:
            data_dict = self.data_dict
        if self.val_artifact is None:  # If --upload_dataset is set, use the existing artifact
            self.train_artifact_path, self.train_artifact = self.download_dataset_artifact(data_dict.get('train'),
                                                                                           opt.artifact_alias)
            self.val_artifact_path, self.val_artifact = self.download_dataset_artifact(data_dict.get('val'),
                                                                                       opt.artifact_alias)

        if self.train_artifact_path is not None:
            train_path = Path(self.train_artifact_path)
            data_dict['train'] = str(train_path)

        if self.val_artifact_path is not None:
            val_path = Path(self.val_artifact_path)
            data_dict['val'] = str(val_path)

        # TODO: init object for summary at the end of training process

        train_from_artifact = self.train_artifact_path is not None and self.val_artifact_path is not None
        if train_from_artifact:
            self.data_dict = data_dict

    def download_dataset_artifact(self, path: str, alias: str = 'latest', save_path: str = None):
        """
        Download the dataset artifact if the path starts with WANDB_ARTIFACT_PREFIX

        Args:
            path (Path): path of the dataset to be used for training
            alias (str): alias of the artifact to be download/used for training

        Returns:
            (str, wandb.Artifact): path of the downladed dataset and it's corresponding 
              artifact object if dataset is found otherwise returns (None, None)

        """
        if isinstance(path, str):  # and path.startswith(WANDB_ARTIFACT_PREFIX)
            artifact_path = Path(remove_prefix(path, WANDB_ARTIFACT_PREFIX) + f":{alias}")
            dataset_artifact = self.wandb_run.use_artifact(artifact_path.as_posix().replace("\\", "/"))
            assert dataset_artifact is not None, "'Error: W&B dataset artifact doesn\'t exist'"
            data_dir = dataset_artifact.download(save_path if save_path is not None else None)
            return data_dir, dataset_artifact
        return None, None

    def download_model_artifact(self, model_artifact_name: str = None, alias: str = 'latest'):
        """
        Download the model checkpoint artifact if the weigth
        start with WANDB_ARTIFACT_PREFIX

        Args:
            :opt (namespace): Comandline arguments for this run
        """
        if isinstance(model_artifact_name, str):  # and model_artifact_name.startswith(WANDB_ARTIFACT_PREFIX):
            model_artifact = wandb.use_artifact(remove_prefix(model_artifact_name, WANDB_ARTIFACT_PREFIX) + f":{alias}")
            assert model_artifact is not None, 'Error: W&B model artifact doesn\'t exist'
            model_dir = model_artifact.download()
            return model_dir, model_artifact

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
            logger = WandbLogger()
            logger.log_dataset_artifact(path, 'raw-mnist', 'dataset')
        """
        if not os.path.exists(path):
            print('File or dir does not exist.')
            return

        dataset_artifact = wandb.Artifact(name=artifact_name,
                                          type=dataset_type,
                                          metadata=dataset_metadata, )
        if os.path.isdir():
            dataset_artifact.add_dir(path)
        elif os.path.isfile():
            dataset_artifact.add_file(path)
        print('Upload dataset into Weight & Biases.')
        self.wandb_run.log_artifact(dataset_artifact)

    def create_dataset_artifact(self, path, name='dataset', type='dataset'):
        """
        Create and return W&B artifact containing W&B Table of the dataset.
        
        Args:
            path (Path): Path to dataset dir/file
            name (str) -- name of the artifact
        
        Returns: 
            Dataset artifact to be logged or used
            :param type:
        """
        artifact = wandb.Artifact(name=name, type=type)
        if isinstance(path, str) and os.path.exists(path):
            if os.path.isdir(path):
                artifact.add_dir(path)
            elif os.path.isfile(path):
                artifact.add_file(path)

        return artifact

    def log_model(self, path: str,
                  epoch: int = None,
                  scores: float or Dict[str, Any] = None,
                  opt: argparse.Namespace = None, ):
        """
        Log the model checkpoint as W&B artifact

        Args:
            path (path): Path to the checkpoints file
            epoch (int): Current epoch number
            scores (float/dict): score(s) represents for current epoch
            opt (namespace): Comand line arguments to store on artifact
        
        Example:
            wandb_logger = WandbLogger()
            for i in range(epochs):
                accuracy = i
                wandb_logger.log_model()
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

        model_artifact = wandb.Artifact('run_' + self.wandb_run.id + '_model', type='model', metadata=metadata)
        model_artifact.add_file(str(path))
        # logging
        aliases = ['latest']
        if epoch is not None:
            aliases.append('epoch ' + str(epoch + 1))
        self.wandb_run.log_artifact(model_artifact, aliases=aliases)
        print(f"Saving model on epoch {epoch} done.")

    def log_training_process(self):
        pass
