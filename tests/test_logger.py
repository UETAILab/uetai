"""Testing UETAI Logger"""
import os
import logging
import unittest

import torch
import numpy as np
import pandas as pd
from PIL import Image

from uetai.logger import CometLogger

_SAVING_PATH = os.path.join(".uetai")
debug_logger = logging.getLogger("debug-logger")


class TestCometLogger(unittest.TestCase):
    """Testing Comet.ml Logger"""

    def setUp(self):
        self.workspace = "uetai_tester"
        self.api_key = "Qd9kYrmr6gq4ouD4GG9TvuxJ6"

    def test_logger_init(self, ):
        """Test CometLogger init."""

        # Test init with api_key
        logger = CometLogger(workspace=self.workspace, api_key=self.api_key)
        self.assertEqual(logger.api_key, self.api_key)
        self.assertTrue(os.path.exists(os.path.join(_SAVING_PATH, "api_key.yaml")))

        # Test save experiment folder and save_dir
        self.assertTrue(os.path.exists(os.path.join(_SAVING_PATH, logger.experiment.id)))
        self.assertEqual(logger.save_dir, os.path.join(_SAVING_PATH, logger.experiment.id))

        # Test get experiment name
        self.assertTrue(isinstance(logger.name, str))
        logger.experiment.end()

    def test_logger_check_api_key(self, ):
        """Test CometLogger check api key."""

        class DummyCometLogger(CometLogger):
            """Dummy CometLogger"""

            def __init__(self, workspace, api_key=None):
                super().__init__(workspace=workspace, api_key=api_key)

            def check_api_key(self, api_key) -> str:
                """Test check_api_key to access protected method"""
                return self._check_api_key(api_key)

        # Test api function with environ variable
        api_save_path = os.path.join(_SAVING_PATH, "api_key.yaml")
        if os.path.exists(api_save_path):
            os.remove(api_save_path)  # clear api from previous test
        os.environ["COMET_API_KEY"] = self.api_key
        logger = DummyCometLogger(workspace=self.workspace)

        # Test api function with previous api key saver
        del os.environ["COMET_API_KEY"]
        self.assertTrue(os.path.exists(api_save_path))
        self.assertEqual(logger.check_api_key(api_key=None), self.api_key)

        logger.experiment.end()

    def test_logger_log(self, ):
        """Test CometLogger log function with with different types:
        * scalar/metric
        * image (image path, tensor or PIL.Image)
        * text (str or {str: metadata})
        * table (str or pandas.DataFrame)
        * combine metric and media
        """
        # Param list
        params = {'param_1': 1, 'param_2': 'nn.optim.Adam'}
        metrics = {'metric_1': 0.1, 'metric_2': 0.2}
        test_df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
        media = {
            'numpy_image': np.random.rand(16, 16) * 255,
            'torch_image': torch.randn(1, 16, 16) * 255,
            'image_path': './sample.jpg',
            'This is a sample text': {'topic': 'random'},
        }
        single_media = [
            test_df,
            './table.csv',
            './sample.jpg',  # image
            'This is a sample text',  # text
        ]

        # Setup
        Image.fromarray(media['numpy_image'].astype(np.uint8)).save(media['image_path'])
        test_df.to_csv('./table.csv')

        # Test logging function
        logger = CometLogger(workspace=self.workspace, api_key=self.api_key)

        logger.log_parameters(params)
        for param, val in params.items():
            self.assertEqual(logger.experiment.get_parameter(param), val)

        logger.log(metrics, step=1)
        for metric, val in metrics.items():
            self.assertEqual(logger.experiment.get_metric(metric), val)

        for test in [media, single_media]:
            for item in test:
                with self.assertLogs(logger='COMET', level='ERROR') as captured:
                    # Test
                    logger.log(item, step=1)
                self.assertLess(len(captured.output), 1)  # expect no ERROR at all

        # TODO: test with combine metric and media
        # TODO: test with wrong type

        logger.experiment.end()


if __name__ == '__main__':
    unittest.main(TestCometLogger())
