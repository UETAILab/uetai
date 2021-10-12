import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_warn

from uetai.logger import SummaryWriter


def check_logger(logger: LightningLoggerBase) -> bool:
    available = True
    msg = None
    if not logger:
        msg = "Trainer has no logger. Cannot log media."
        available = False
    elif isinstance(logger, SummaryWriter):
        if 'wandb' != logger.log_tool:
            msg = ("`Summary_Writer` is not logging with `wandb`."
                   "Please set `WANDB_API_KEY` to start using `wandb`")
            available = False
    else:
        msg = f"Except logger is `SummaryWriter` type, receive {logger.__class__.__name__}"
        available = False

    if msg is not None:
        rank_zero_warn(msg)
    return available


def trainer_finish_run(trainer: "pl.Trainer") -> None:
    logger = trainer.logger
    if isinstance(logger, SummaryWriter):
        if 'wandb' in logger.log_tool:
            logger.wandb_run.finish()
