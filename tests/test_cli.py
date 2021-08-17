__author__ = "Nguyen Van Phi"
__copyright__ = "Nguyen Van Phi"
__license__ = "MIT"

# from uetai.logger.summary_writer import SummaryWriter
import torch
from uetai.logger.wandb.wandb_logger import WandbLogger, create_dataset_artifact, download_model_artifact

def test_artifact_control_function():
    wb_logger = WandbLogger('experiment')
    # upload (random) dataset as artifact
    path = 'path/to/dataset'
    create_dataset_artifact(path, dataset_name='demo-dataset')

    # delete local dataset
    down_path = wb_logger.download_dataset_artifact(dataset_name='demo-dataset')

    # load dataset

    # setup a simple training run
    model = Net()
    for epoch in range(100):
        torch.save(model.state_dict(), 'model.pth')
        wb_logger.log_model('model.pth')
    
    # delete current model weight

    # attempt to download latest weight and continues train
    weight_path = download_model_artifact('experiment/model.pth')

    pass

def test_main():
    assert 1 == 1
