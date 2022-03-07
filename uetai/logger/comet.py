""""""
import argparse
import os
import sys
import getpass
import logging
from typing import Any, Dict, List, Optional, Union

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
    def __init__(self, project_name: str = None, workspace: str = None, api_key: str = None, ):
        super().__init__()
        # TODO: Init experiment and collect metadata
        self.project_name = project_name
        self.workspace = workspace
        self.api_key = self._check_api_key(api_key)
        self.experiment = self._init_experiment()

    # Initialize experiment -----------------------------------------------------------
    def _init_experiment(self,) -> Union[Experiment, None]:
        if not _COMET_AVAILABLE:
            raise ModuleNotFoundError("Comet_ml is not available. Try install with `pip install comet_ml`.")
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

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """Auto-detect data type (e.g: metrics, image, audio,) and log it to Comet.ml"""
        # if isinstance(data, dict):
        #     self.log_metrics(data, step)
        # self.experiment.log_metrics()
        pass

    def log_metric(self, metric_name: str, metric_value: float, step: Optional[int] = None):
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass

    def log_parameter(self, params: argparse.Namespace, *args, **kwargs):
        pass

    def log_text(self):
        pass

    def log_image(self):
        pass

    def log_graph(self):
        pass

    @property
    def name(self) -> str:
        return self.experiment.get_name()

    @name.setter
    def name(self, name: str):
        self.experiment.set_name(name)
