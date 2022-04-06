"""Comet Logger customize by UETAI"""
import os
import getpass
import logging
import imghdr
from typing import Any, Dict, Optional, Union

import PIL
import yaml
import numpy as np
from PIL.Image import Image

from torch import Tensor

from src.uetai.logger.base import UetaiLoggerBase
from src.uetai.utilities import module_available

log = logging.getLogger(__name__)
_COMET_AVAILABLE = module_available("comet_ml")
_SAVING_PATH = ".uetai"

if not os.path.exists(_SAVING_PATH):
    os.makedirs(_SAVING_PATH, exist_ok=True)

if _COMET_AVAILABLE:
    # For more information about Comet auto logging, see:
    # https://www.comet.ml/docs/python-sdk/advanced/#comet-auto-logging
    os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'
    import comet_ml
    from comet_ml import Experiment

    # try:
    #     from comet_ml.api import API
    # except ModuleNotFoundError:  # pragma: no-cover
    #     from comet_ml.papi import API  # pragma: no-cover


class CometLogger(UetaiLoggerBase):
    """CometLogger for logging experiment into Comet ML dashboard. We create a customed logger
    for synchronized API between our supported dashboard. See supported dashboard/framework at 
    :ref:`Quickstart <quickstart>`.

    :params project_name: Name of the project.
    :type project_name: Optional[str]
    :params workspace: Name of the workspace.
    :type workspace: Optional[str]
    :params api_key: Comet API key of the user.
    :type api_key: Optional[str]

    .. note::
        
        * For more information about Comet auto logging, see: https://www.comet.ml/docs/
        
    .. code-block:: python

        # from comet_ml import Experiment
        from uetai.logger import CometLogger

        logger = CometLogger(project_name='UETAI_Project')
    """

    def __init__(
            self,
            project_name: Optional[str] = 'UETAI_Project',
            workspace: Optional[str] = None,
            api_key: Optional[str] = None,
    ):
        super().__init__()
        # TODO: Collect metadata with traceback
        self.project_name = project_name
        self.workspace = workspace
        self.api_key = self._check_api_key(api_key)
        self.experiment = self._init_experiment()  # Experiment object
        self._experiment_path = self._set_experiment_path()

    # Initialize experiment -----------------------------------------------------------
    def _init_experiment(self, ) -> Experiment:
        """Initialize Comet ML experiment."""
        if not _COMET_AVAILABLE:
            raise ModuleNotFoundError(
                "Comet_ml is not available. Try install with `pip install comet_ml`.")
        experiment = comet_ml.Experiment(
            project_name=self.project_name,
            api_key=self.api_key,
            workspace=self.workspace, )
        return experiment

    @staticmethod
    def _check_api_key(api_key: str) -> str:
        """Check API key from saved file, environment variable, or input.

        :params api_key: API key of the user.
        :type api_key: Optional[str].
        :return: str.
        """
        api_key_path = os.path.join(_SAVING_PATH, 'api_key.yaml')
        if api_key is None:
            if 'COMET_API_KEY' in os.environ:
                api_key = os.environ['COMET_API_KEY']
            else:
                try:
                    with open(api_key_path, 'r', encoding="utf8") as file:
                        api_key = yaml.safe_load(file)['COMET_API_KEY']
                except FileNotFoundError:
                    api_key = getpass.getpass("Please enter your Comet API key: ")
        else:
            os.environ["COMET_API_KEY"] = api_key
        # save api_key to file
        with open(api_key_path, "w", encoding="utf8", errors="surrogateescape") as file:
            yaml.dump({"COMET_API_KEY": api_key}, file)
        return api_key

    def _set_experiment_path(self) -> str:
        """Set experiment folder.
        :return: experiment folder.
        """
        experiment_path = os.path.join(_SAVING_PATH, self.experiment.id)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        return experiment_path

    @property
    def name(self) -> str:
        """Get experiment name.
        :return: Experiment name.
        """
        return self.experiment.get_name()

    @property
    def save_dir(self) -> Optional[str]:
        """Get experiment save directory.
        :return: Experiment save directory.
        """
        return self._experiment_path

    # Logging --------------------------------------------------------------------------
    def log(self, data: Any, step: Optional[int] = None):
        """Wizard to log data to Comet.ml dashboard. We supported following data types:

        - Metrics: Dictionaries of metrics with keys as metric names and values as metric values.
        - Image: Should be a numpy array or a PIL.Image or a torch.Tensor.
        - Text: Dictionaries of text with keys as text names and values as metadata of text.
        - Tables: Dictionaries of tables with keys as table names and values as the table.

        :params data: Dictionary of metrics, media, text, graph to log.
        :type data: dict[str, Any]
        :params step: Optional step to log.
        :type step: int
        :params include_context: Whether to include context in the log.
        :type step: bool or str

        .. code-block:: python

            # Log metrics
            logger.log({'acc': 0.314, 'p': 0.521})

            # Log image
            logger.log('./sample.jpg')
            logger.log({'img': './sample.jpg'})
            logger.log({'np_image': np.random.rand(32, 32, 3)})
            logger.log({'tensor_image': torch.rand(32, 32, 3)})
            logger.log({'pil_image': image})  # PIL.Image

            # Log text & its metadata
            logger.log('Hello world!')
            logger.log({'This is a text': {'dialog': 'id1', 'topic': 'test'}})

            # Log table
            logger.log('./table.csv')
            logger.log(data_frame)  # pandas.DataFrame
        """
        if not isinstance(data, (dict, set)):
            try:
                data = {data}  # assume data is a single string
            except TypeError:
                raise ValueError(f"Unsupported data type: {type(data)}.")

        if isinstance(data, set):
            if all(isinstance(item, str) for item in data):  # set of strings
                self._log_set(data, step)
            else:
                raise ValueError("Value passed must be a string or a pandas.DataFrame.")

        else:  # for dict
            if any(not isinstance(key, str) for key in data.keys()):
                raise ValueError("Key values passed must be a string.")
            self._log_dict(data, step)

    def _log_dict(self, data: dict, step: Optional[int] = None):
        if len(data) < 1:
            raise ValueError("Data is empty.")
        for key, val in data.items():
            if isinstance(val, (float, int)):
                #  metric value should be float or int
                self.log_metric(metric_name=key, metric_value=val, step=step)
            elif isinstance(val, (str, np.ndarray, Tensor, Image)):
                if isinstance(val, str) and imghdr.what(val) is None:
                    raise ValueError("Passed value is not a supported image type.")
                img = self._preprocess_image(image_data=val)
                self.log_image(name=key, image_data=img, step=step)
            elif isinstance(val, dict):
                self.log_text(text=key, step=step, metadata=val)
            else:
                raise ValueError(f"Unsupported data type: {type(val)}.")

    def _log_set(self, data: set, step: Optional[int] = None):
        if len(data) < 0:
            raise ValueError("Data is empty.")
        for item in data:
            try:
                if imghdr.what(item) is not None:  # valid image format
                    img = self._preprocess_image(item)
                    self.log_image(image_data=img, step=step)
                else:
                    raise ValueError("Passed file is not a supported image type.")
            except FileNotFoundError:
                self.log_text(text=item, step=step)

    # Override logging function ----------------------------------------------------------

    def log_metric(self, metric_name: str, metric_value: float, step: Optional[int] = None):
        """Log a single metric to Comet.ml.
        
        :params metric_name: Metric name.
        :type metric_name: str
        :params metric_value: Metric value.
        :type metric_value: float
        :params step: Optional step to log.
        :type step: int

        .. note::
            
            See more: https://www.comet.ml/docs/python-sdk/Experiment/#experimentlog_metric
        """
        self.experiment.log_metric(metric_name, metric_value, step=step)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log multiple metrics in form of dictionary to Comet.ml.
        
        :params metrics: Dictionary of metrics with keys as metric names and values as metric values.
        :type metrics: dict[str, Any]
        :params step: Optional step to log.
        :type step: int

        .. note::
            
            See more: https://www.comet.ml/docs/python-sdk/Experiment/#experimentlog_metrics
        """
        self.experiment.log_metrics(metrics, step=step)

    def log_parameters(self, params: Any, *args, **kwargs):
        """Log dictionary of parameters to Comet.ml.
        
        :params params: Parameters to log.
        :type params: dict[str, Any]
        :params args: Optional arguments.
        :type args: list

        .. note::
            
            See more: https://www.comet.ml/docs/python-sdk/Experiment/#experimentlog_parameters
        """
        self.experiment.log_parameters(params, *args, **kwargs)

    def log_text(self, text: str, step: Optional[int] = None, metadata: Any = None):
        """Log text into Text tab in Comet.ml dashboard.
        
        :params text: Text to log.
        :type text: str
        :params step: Optional step to log.
        :type step: int
        :params metadata: Optional metadata to log.
        :type metadata: dict[str, Any]
        
        .. note:: 
            
            See more: https://www.comet.ml/docs/python-sdk/Experiment/#experimentlog_text

        .. code-block:: python

            # Log text & its metadata
            logger.log_text('Hello world!', {'dialog': 'id1', 'topic': 'test'})
        """
        self.experiment.log_text(text, step, metadata)

    def log_histogram_3d(self, histogram: Any, step: Optional[int] = None, metadata: Any = None):
        """Log 3D histogram to Comet.ml."""
        self.experiment.log_histogram_3d(histogram, step, metadata)

    def log_html(self, html: str, clear=False):
        """Log HTML to Comet.ml.

        :params html: HTML to log.
        :type html: str
        :params clear: Optional flag to clear the HTML log.
        :type clear: bool

        .. note::
            
            See more: https://www.comet.ml/docs/python-sdk/Experiment/#experimentlog_html

        .. code-block:: python

            logger.log_html('<a href="www.comet.ml"> Test log html Comet.ml </a>')
        """
        self.experiment.log_html(html, clear)

    def log_image(self, image_data: Any, name: str = None, step: Optional[int] = None):
        """Log image to Comet.ml.
        
        :params image_data: Image data to log.
        :type image_data: str, np.ndarray, torch.Tensor, PIL.Image.Image
        :params name: Optional name of the image.
        :type name: str

        .. note::

            See more: https://www.comet.ml/docs/python-sdk/Experiment/#experimentlog_image

        .. code-block:: python

            # Log image
            logger.log_image(image_data=img, name='test_image', step=step)
        """
        self.experiment.log_image(image_data, name=name, step=step)

    # Custom logging function ----------------------------------------------------------

    @staticmethod
    def _preprocess_image(image_data: Union[str, Tensor, Image, np.ndarray]) -> Image:
        """Validating image before log it to Comet.ml.
        Image data can be a string, which direct points to an image file;
        or a numpy.ndarray or torch.Tensor, which be able to convert to a PIL.Image.

        .. note::
            A torch.tensor should be in the shape of (C, H, W).
            Otherwise, a numpy array should be in the shape of (H, W, C).
        """
        if isinstance(image_data, str):  # image path
            if not os.path.exists(image_data):
                raise FileNotFoundError("Image path is not a valid image file.")
            return PIL.Image.open(image_data)
        if isinstance(image_data, Tensor):
            if image_data.ndimension() != 3:
                raise ValueError("Tensor must be a 3D tensor.")
            # image shape must be (C, H, W) while C = 1, 3, 4 (RGBA)
            if image_data.size(0) not in [1, 3, 4]:
                raise ValueError(f"Shape {image_data.size()} is invalid dimensions"
                                 f"for Tensor image data")
            image_data = image_data.permute(1, 2, 0).cpu().detach().numpy()  # (H, W, C)
        else:  # np.ndarray (H, W, C) or (H, W)
            # check image shape is 2D or 3D array
            if image_data.ndim not in [2, 3]:
                raise ValueError("Numpy array must be a 2D or 3D array.")
            if image_data.shape[-1] not in [3, 4] and image_data.ndim == 3:
                raise ValueError(f"Shape {image_data.shape} is invalid dimensions"
                                 f"for Tensor image data")
        # squeeze channel if it is a single channel image
        image_data = np.squeeze(image_data) if image_data.ndim == 3 else image_data
        return PIL.Image.fromarray(image_data)  # convert to PIL.Image

    # Artifact functions -------------------------------------------------------------------

    # def log_artifact(
    #     self,
    #     artifact_path: str,
    #     artifact_name: str = None,
    #     artifact_type: str = None
    # ):
    #     """Log artifact to Comet.ml storage"""
    #     artifact = comet_ml.Artifact(name=artifact_name, artifact_type=artifact_type)
    #     artifact_metadata = {
    #         'artifact_name': artifact_name,
    #         'artifact_type': artifact_type,
    #         'artifact_path': artifact_path,
    #         'artifact_version': artifact.version,
    #         'experiment_name': self.experiment.get_name(),
    #         'experiment_id': self.experiment.id,
    #         'project_name': self.experiment.project_name,
    #         'project_id': self.experiment.project_id,
    #     }
    #     # save artifact metadata to file
    #     artifact_metadata_path = os.path.join(artifact_path, 'metadata.yaml')
    #     with open(artifact_metadata_path, 'w', encoding='utf8', errors='surrogateescape') as file:
    #         yaml.dump(artifact_metadata, file)
    #     if os.path.exists(artifact_path):
    #         artifact.add(local_path_or_data=artifact_path)
    #     elif artifact_path.startswith("s3://"):
    #         artifact.add_remote(uri=artifact_path)
    #     # elif artifact_path.startswith("https://drive.google.com/"):
    #     #     artifact.add_remote(uri=artifact_path)
    #     self.experiment.log_artifact(artifact)
    #     return artifact
    #
    # def download_artifact(self, artifact_name: str, save_path: str = None):
    #     """Download artifact from Comet.ml storage
    #         artifact_name format: "workspace/artifact-name:version_or_alias"
    #     """
    #     artifact = self.experiment.get_artifact(artifact_name)
    #     artifact.download(save_path)
    #     return save_path
