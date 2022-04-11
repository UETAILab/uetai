"""init public class"""
from .base import UetaiLoggerBase
from .comet import _COMET_AVAILABLE, CometLogger


__all__ = ["UetaiLoggerBase"]

if _COMET_AVAILABLE:
    __all__.append("CometLogger")
