"""Comet Logger customize by UETAI"""
import argparse
import os
import getpass
import logging
from typing import Any, Dict, Optional, Union

import PIL
from PIL.Image import Image
from torch import Tensor

from uetai.logger.base import UetaiLoggerBase
from uetai.utilities import module_available

log = logging.getLogger(__name__)
_COMET_AVAILABLE = module_available("comet_ml")

if _COMET_AVAILABLE:
    import comet_ml
    from comet_ml import Experiment

    try:
        from comet_ml.api import API
    except ModuleNotFoundError:  # pragma: no-cover
        # For more information, see: https://www.comet.ml/docs/python-sdk/releases/#release-300
        from comet_ml.papi import API  # pragma: no-cover


class CometLogger(UetaiLoggerBase):
    """CometLogger."""
    def __init__(
        self,
        project_name: Optional[str] = 'UETAI_Project',
        workspace: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        :params project_name: Name of the project.
        :type project_name: Optional[str]
        :params workspace: Name of the workspace.
        :type workspace: Optional[str]
        :params api_key: Comet API key of the user.
        :type api_key: Optional[str]
        """
        super().__init__()
        # TODO: Init experiment and collect metadata
        self.project_name = project_name
        self.workspace = workspace
        self.api_key = self._check_api_key(api_key)
        self.experiment = self._init_experiment()  # Experiment object

    # Initialize experiment -----------------------------------------------------------
    def _init_experiment(self, ) -> Experiment:
        if not _COMET_AVAILABLE:
            raise ModuleNotFoundError(
                "Comet_ml is not available. Try install with `pip install comet_ml`.")
        experiment = comet_ml.Experiment(self.project_name, self.api_key, self.workspace)
        return experiment

    @staticmethod
    def _check_api_key(api_key: str) -> str:
        if api_key is None:
            if os.environ.get("COMET_API_KEY") is None:
                api_key = getpass.getpass("Please enter your Comet API key: ")
            else:
                api_key = os.environ.get("COMET_API_KEY")
        else:
            os.environ["COMET_API_KEY"] = api_key
            # TODO: save api_key to somewhere
        return api_key

    @property
    def name(self) -> str:
        """Get experiment name.
        :return: Experiment name.
        """
        return self.experiment.get_name()

    @name.setter
    def name(self, name: str):
        self.experiment.set_name(name)

    # Logging --------------------------------------------------------------------------
    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """Auto-detect data type (e.g: metrics, image, audio,) and log it to Comet.ml
        :params data: Dictionary of metrics, media, text, graph to log.
        :type data: dict
        :params step: Optional step to log.
        :type step: int
        :params include_context: Whether to include context in the log.
        :type step: bool or str

        ..note::
            * If the data is a dictionary of floats, it will be logged as metrics.
            * If the dictionary contains Tensor or PIL.Image, it will be logged as an image.
            * If the dictionary contains str or another sub-dictionary, it will be logged as a text.
        """
        # Log metrics
        all_metric = all(isinstance(value, float) for value in data.values())
        if all_metric:
            self.log_metrics(data, step=step)
        else:
            for key, val in data.items():
                if isinstance(val, float):
                    self.log_metric(key, val, step)
                elif isinstance(val, (Image, Tensor)):
                    self.log_image(val, key, step)
                elif isinstance(val, str):
                    self.log_text(val, step)
                elif isinstance(key, str) and isinstance(val, dict):
                    self.log_text(key, step, val)

    def log_metric(self, metric_name: str, metric_value: float, step: Optional[int] = None):
        """Log a single metric to Comet.ml."""
        self.experiment.log_metric(metric_name, metric_value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to Comet.ml."""
        self.experiment.log_metrics(metrics, step=step)

    def log_parameter(self, params: argparse.Namespace, *args, **kwargs):
        """Log parameters to Comet.ml."""
        self.experiment.log_parameters(params, *args, **kwargs)

    def log_text(self, text: str, step: Optional[int] = None, metadata: Any = None):
        """Log text to Comet.ml."""
        self.experiment.log_text(text, step, metadata)

    def log_image(
        self,
        image_data: Union[str, Image, Tensor],
        name: str = None,
        step: Optional[int] = None
    ):
        """Log image to Comet.ml."""
        if isinstance(image_data, str):
            try:
                image_data = PIL.Image.open(image_data)
            except ValueError:
                log.error("Image data is not a valid image file.")
        elif isinstance(image_data, Tensor):
            # Check if this tensor is a valid image
            if image_data.ndimension() != 3:
                log.error("Image data is not a valid image.")
            # Check if tensor's last dimension equal 1 or 3
            if image_data.size(-1) not in [1, 3]:
                log.error("Image data is not a valid image.")
        self.experiment.log_image(image_data, name=name, step=step)
