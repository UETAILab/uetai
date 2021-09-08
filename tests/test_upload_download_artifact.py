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
        self.logger = SummaryWriter('uetai')

    def test_dataset_artifact_upload_download(self):
        # create a (random) dataset
        path = './tmp/dataset'
        os.makedirs(path, exist_ok=True)
        torch.save(torch.randn((1000, 1000)), os.path.join(path, "datapoint.tmp"))

        # upload dataset as artifact
        artifact = self.logger.log_dataset_artifact(path, dataset_name="dummy-set")
        shutil.rmtree('./tmp')  # del tmpdir
        assert type(artifact) == Artifact

        down_path, dataset_artifact = self.logger.download_dataset_artifact(
                                                dataset_name='dummy-set')
        assert type(dataset_artifact) == Artifact
        assert os.path.exists(down_path)
        assert os.path.exists(os.path.join(down_path, "datapoint.tmp"))
