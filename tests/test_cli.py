"""CLI unit-test"""
__author__ = "Nguyen Van Phi"
__copyright__ = "Nguyen Van Phi"
__license__ = "MIT"

import os
import shutil
from unittest import TestCase

import torch
from wandb.sdk.wandb_artifacts import Artifact
from uetai.logger.summary_writer import SummaryWriter


class TestSummaryWriterWandb(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSummaryWriterWandb, self).__init__(*args, **kwargs)
        self.logger = SummaryWriter('experiment')

    def test_dataset_artifact_upload(self):
        # create a (random) dataset
        path = './tmp/dataset'
        os.makedirs(path, exist_ok=True)
        torch.save(torch.randn((1000, 1000)), os.path.join(path, "datapoint.tmp"))

        # upload dataset as artifact
        artifact = self.logger.log_dataset_artifact(path, dataset_name="dummy-set")
        shutil.rmtree('./tmp')  # del tmpdir
        assert type(artifact) == Artifact

    def test_dataset_artifact_download(self):
        # download dataset
        down_path, dataset_artifact = self.logger.download_dataset_artifact(
            dataset_name='demo-dataset',
        )
        assert type(dataset_artifact) == Artifact
        assert os.path.exists(down_path)
        assert os.path.exists(os.path.join(down_path, "datapoint.tmp"))

    def test_model_artifact_upload(self):
        # setup a simple training run
        save_path = './tmp/model/'
        model = torch.nn.Linear(10, 10)
        # for epoch in range(5):
        model_artifact = self.logger.save_model(
            model.state_dict(),
            os.path.join(save_path, 'model.pth'),
        )
        assert type(model_artifact) == Artifact

    def test_model_artifact_download(self):
        # get model artifact
        model_name = "run_"+self.logger.wandb_run.id+'_model'

        # download a model weight
        path, model_artifact = self.logger.download_model_artifact(
            artifact_name=model_name,
        )
        assert type(model_artifact) == Artifact
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(path, "model.pth"))

    def test_watch_model(self):
        pass
