"""Abstracts base class to build loggers."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class UetaiLoggerBase(ABC):
    """Abstracts base class to build loggers."""

    def __init__(self):
        """Initialize logger."""
        # self.logger = logging.getLogger(name)
        # self.logger.setLevel(logging.DEBUG)
        # self.logger.propagate = False
        # self.logger.handlers = []

    @abstractmethod
    def log_metric(self, metric_name: str, metric_value: float, step: Optional[int] = None):
        """Log metric."""

    @abstractmethod
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log metrics."""

    @abstractmethod
    def log_parameters(self, params: Any, *args, **kwargs):
        """Record parameter.
        Args:
            params: :class:`~argparse.Namespace` containing the parameters
            args: Optional positional arguments, depends on the specific logger being used
            kwargs: Optional keywoard arguments, depends on the specific logger being used
        """

    @abstractmethod
    def log_text(self, text, step, metadata):
        """Log text."""

    @abstractmethod
    def log_image(self, image_data, name, step):
        """Log image."""

    # @abstractmethod
    # def log_graph(self):
    #     """Log graph."""

    @property
    def save_dir(self) -> Optional[str]:
        """Return the experiment save directory."""
        return

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the experiment name."""

    # @property
    # @abstractmethod
    # def version(self) -> Union[str, int]:
    #     """Return the experiment version."""
