import unittest

import os
import csv
import torch
import wandb
import numpy as np

from uetai.logger import WandbLogger


class TestWandbLogger(unittest.TestCase):
    """Test WandbLogger."""

    def setUp(self) -> None:
        self.workspace = "uetai"
        os.environ['WANDB_API_KEY'] = '761fd4e45e2fc234ea9041dd463a98ab81979af1'
        self.logger = WandbLogger(workspace=self.workspace)

    def test_logger_log(self):
        logger = self.logger

        logger.log_metric(metric_name='metric_1', metric_value=0.2)
        metrics = {'metric_1': 0.1, 'metric_2': 0.2}
        logger.log_metrics(metrics)

        logger.log_text(text=[["I love my phone", "1", "1"],
                              ["My phone sucks", "0", "-1"]],
                        metadata=["Text", "Predicted Sentiment", "True Sentiment"])

        logger.log_image(np.random.rand(16, 16) * 255, 'numpy_image')
        imgs = [np.random.rand(16, 16) * 255,
                np.random.rand(16, 16) * 255,
                np.random.rand(16, 16) * 255]
        logger.log_images(imgs, name='images_test')

        my_data = [
            [0, 'text', 0, 0],
            [1, 'text', 8, 0],
            [2, 'text', 7, 1],
            [3, 'text', 1, 1]
        ]

        # create a wandb.Table() with corresponding columns
        columns = ['id', 'text', 'prediction', 'gt']
        test_table = wandb.Table(data=my_data, columns=columns)
        logger.log_table('table', test_table)

        config = {
            'lr': 0.01,
            'bs': 32,
            'epochs': 10,
        }
        logger.log_parameters(config)

    def test_logger_log_artifact(self):
        header = ['name', 'area', 'country_code2', 'country_code3']
        data = ['Afghanistan', 652090, 'AF', 'AFG']

        with open('countries.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(data)

        logger = self.logger
        logger.log_artifact(artifact_name='countries',
                            artifact_path='countries.csv',
                            auto_profiling=True)


if __name__ == '__main__':
    unittest.main()
