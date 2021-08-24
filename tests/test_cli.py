__author__ = "Nguyen Van Phi"
__copyright__ = "Nguyen Van Phi"
__license__ = "MIT"

import torch
from uetai.logger.wandb.wandb_logger import WandbLogger, create_dataset_artifact, download_model_artifact


# from uetai.logger.summary_writer import SummaryWriter


def test_artifact_control_function():
    wb_logger = WandbLogger('experiment')
    # upload (random) dataset as artifact
    path = 'path/to/dataset'
    create_dataset_artifact(path, dataset_name='demo-dataset')

    # delete local dataset
    down_path = wb_logger.download_dataset_artifact(dataset_name='demo-dataset')

    # load dataset

    # setup a simple training run
    model = torch.nn.Linear(10, 10)
    for epoch in range(100):
        torch.save(model.state_dict(), 'model.pth')
        wb_logger.log_model('model.pth')

    # delete current model weight

    # attempt to download latest weight and continues train
    weight_path = download_model_artifact('experiment/model.pth')
    assert down_path is not None
    assert weight_path is not None
    pass


def test_main():
    assert 1 == 1
