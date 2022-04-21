"""UETAI Data visualize function."""
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from ..utilities import module_available

log = logging.getLogger(__name__)
_COMET_AVAILABLE = module_available("comet_ml")
_SAVING_PATH = ".uetai"


class DataVisualize():
    def __init__(self):
        pass

    def visualize_data(self, data, labels, save_path=None):
        pass

