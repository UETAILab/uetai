"""CLI unit-test"""
import os
import shutil
import time
from unittest import TestCase

import torch
from wandb.sdk.wandb_artifacts import Artifact

from uetai.logger.summary_writer import SummaryWriter


class TestSummaryWriterWandb(TestCase):
    """artifact upload/download tests"""
    def __init__(self, *args, **kwargs):
        super(TestSummaryWriterWandb, self).__init__(*args, **kwargs)
        self.logger = SummaryWriter('uetai', log_tool='wandb')

    def test_dataset_artifact_upload_download(self):
        # create a (random) dataset
        path = './tmp/dataset'
        os.makedirs(path, exist_ok=True)
        torch.save(torch.randn((1000, 1000)), os.path.join(path, "datapoint.tmp"))

        # upload dataset as artifact
        artifact = self.logger.log_dataset_artifact(path, dataset_name="dummy-set")
        shutil.rmtree('./tmp')  # del tmpdir
        assert isinstance(artifact, Artifact)
        time.sleep(5)
        down_path, _ = self.logger.download_dataset_artifact(
            dataset_name='dummy-set'
        )
        assert os.path.exists(down_path)
        assert os.path.exists(os.path.join(down_path, "datapoint.tmp"))
