__author__ = "Nguyen Van Phi"
__copyright__ = "Nguyen Van Phi"
__license__ = "MIT"

import os
import shutil
from unittest import TestCase

import torch
from uetai.logger.summary_writer import SummaryWriter


# from uetai.logger.summary_writer import SummaryWriter

class TestSummaryWriterWandb(TestCase):
    def test_artifact_control_function(self):
        pass
        # logger = SummaryWriter('experiment')
        # # upload (random) dataset as artifact
        # os.makedirs("./tmp/dataset", exist_ok=True)
        # torch.save(torch.randn((1000, 1000)), "./tmp/dataset/datapoint.tmp")
        # path = './tmp/dataset'
        # logger.log_dataset_artifact(path, dataset_name='demo-dataset')
        # shutil.rmtree('/tmp')
        # # delete local dataset
        # down_path, version = logger.download_dataset_artifact(dataset_name='demo-dataset')
        # assert os.path.exists(down_path)
        # assert os.path.exists(os.path.join(down_path, "datapoint.tmp"))
        #
        # # load dataset
        #
        # # setup a simple training run
        # model = torch.nn.Linear(10, 10)
        # for epoch in range(100):
        #     logger.save_model(model.state_dict(), 'model.pth')
        #
        # # attempt to download latest weight and continues train
        # weight_path = logger.log_model_artifact('experiment/model.pth')
        # assert down_path is not None
        # assert weight_path is not None
