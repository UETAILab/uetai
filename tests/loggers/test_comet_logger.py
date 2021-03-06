"""Testing UETAI Logger"""
import os
import unittest

import torch
import numpy as np
from PIL import Image

from uetai.logger import CometLogger

_SAVING_PATH = os.path.join(".uetai")


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
        # Metrics + image + text
        metrics = {'metric_1': 0.1, 'metric_2': 0.2}
        media = {
            'numpy_image': np.random.rand(16, 16) * 255,
            'torch_image': torch.randn(1, 16, 16) * 255,
            'image_path': './sample.jpg',
            'This is a sample text': {'topic': 'random'},
        }
        single_media = [
            './sample.jpg',  # image
            'This is a sample text',  # text
        ]

        # Setup
        Image.fromarray(media['numpy_image'].astype(np.uint8)).save(media['image_path'])

        # Test logging function
        logger = CometLogger(workspace=self.workspace, api_key=self.api_key)

        logger.log(metrics, step=1)
        for metric, val in metrics.items():
            self.assertEqual(logger.experiment.get_metric(metric), val)

        for test in [media, single_media]:
            for item in test:
                try:
                    item = {item: test[item]}
                except TypeError:
                    pass
                # logger name is wrong
                # with self.assertLogs(logger=__name__, level='ERROR') as captured:
                    # Test
                logger.log(item, step=1)
                # self.assertLess(len(captured.output), 1)  # expect no ERROR at all

        # TODO: test with combine metric and media

        wrong_type = [
            [1],  # int
            {},  # empty dict
            {1, 2, 3},  # set
            {1: 'a', 2: 'b'}  # dict with non-str key
        ]
        for item in wrong_type:
            with self.assertRaises(ValueError):
                logger.log(item, step=1)

    def test_logger_preprocess_image(self, ):
        """Test CometLogger preprocess_image function."""
        pass

    def test_logger_override_function(self):
        """Test CometLogger override original function."""
        img = np.random.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
        params = {'lr': 1e-5, 'optim': 'nn.optim.Adam'}
        text = 'This is a sample text'
        html = '<h1>This is a sample html</h1>'
        hist = np.random.normal(170, 10, 250)

        logger = CometLogger(workspace=self.workspace, api_key=self.api_key)

        logger.log_parameters(params)
        for param, val in params.items():
            self.assertEqual(logger.experiment.get_parameter(param), val)

        logger.log_metric('recall', 0.1, step=1)
        logger.log_metrics({'loss': 0.1, 'acc': 0.9}, step=1)
        for metric, val in {'recall': 0.1, 'loss': 0.1, 'acc': 0.9}.items():
            self.assertEqual(logger.experiment.get_metric(metric), val)

        logger.log_text(text, step=1)
        logger.log_image(img, step=1)
        logger.log_html(html)
        logger.log_histogram_3d(hist)


if __name__ == '__main__':
    unittest.main()
