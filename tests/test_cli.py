__author__ = "Nguyen Van Phi"
__copyright__ = "Nguyen Van Phi"
__license__ = "MIT"

from unittest import TestCase

import torch
from uetai.logger.summary_writer import SummaryWriter


# from uetai.logger.summary_writer import SummaryWriter

class TestSummaryWriterWandb(TestCase):
    def test_artifact_control_function(self):
        logger = SummaryWriter('experiment')
        # upload (random) dataset as artifact
        path = 'path/to/dataset'
        logger.log_dataset_artifact(path, dataset_name='demo-dataset')

        # delete local dataset
        down_path = logger.download_dataset_artifact(dataset_name='demo-dataset')

        # load dataset

        # setup a simple training run
        model = torch.nn.Linear(10, 10)
        for epoch in range(100):
            logger.save_model(model.state_dict(), 'model.pth')

        # delete current model weight

        # attempt to download latest weight and continues train
        weight_path = logger.log_model_artifact('experiment/model.pth')
        assert down_path is not None
        assert weight_path is not None
