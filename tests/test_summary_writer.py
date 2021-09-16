import os
import shutil
from unittest import TestCase

import torch
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from uetai.logger import SummaryWriter


class TestSummaryWriter(TestCase):
    def test_instance(self):
        logger = SummaryWriter("uetai")
        assert isinstance(logger.logger, WandbLogger)

        api_key = os.environ.pop("WANDB_API_KEY", None)
        logger = SummaryWriter("uetai")
        os.environ["WANDB_API_KEY"] = api_key
        assert isinstance(logger.logger, TensorBoardLogger)

    def test_un_support_logger_type(self):
        with self.assertRaises(Exception):
            logger = SummaryWriter("uetai", log_tool="lcoal")

    def test_watch(self):
        logger = SummaryWriter("uetai")
        model = torch.nn.Conv2d(10, 10, 3)
        logger.watch(model)

    def test_data_path_local(self):
        api_key = os.environ.pop("WANDB_API_KEY", None)

        logger = SummaryWriter("uetai")
        os.makedirs("tmp/my_folder", exist_ok=True)
        torch.save(torch.randn((100, 100)), "tmp/my_folder/tmp.file")
        data_path = logger.data_path(path="tmp/my_folder")
        self.assertEqual(data_path, "tmp/my_folder")
        shutil.rmtree("tmp")
        os.environ["WANDB_API_KEY"] = api_key

    def test_data_path_prod(self):
        logger = SummaryWriter("uetai")
        data_dir = logger.data_path(
            path='./tmp/my_folder/',
            dataset_name='dummy-set',
            alias='latest')
        self.assertTrue(os.path.exists(data_dir))

    def test_data_path_url(self):
        logger = SummaryWriter("uetai")
        data_dir = logger.data_path("https://data.deepai.org/mnist.zip")
        self.assertTrue(os.path.exists(data_dir))
