"""DiffCSE Trainer"""

import os

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping
)

from utils import get_config
from model import SimCSEModel
from dataset import SimCSEDataModule

CONFIG_PATH = "./config_roberta.yaml"
TRAINER_CONFIG = get_config(CONFIG_PATH, "trainer")
os.environ['TOKENIZERS_PARALLELISM'] = "false"


logger = TensorBoardLogger(
    save_dir=TRAINER_CONFIG.get("log_dir"),
    name=TRAINER_CONFIG.get("name"),
    default_hp_metric=False
)


early_stopping = EarlyStopping(
    monitor="SpearmanR",
    min_delta=1e-4,
    patience=20,
    verbose=True,
    mode="max"
)

model_checkpoint = ModelCheckpoint(
    monitor="SpearmanR",
    dirpath=TRAINER_CONFIG.get("ckpt_dir"),
    filename=TRAINER_CONFIG.get("name"),
    mode="max"
)

trainer = Trainer(
    max_epochs=TRAINER_CONFIG.get("max_epochs"),
    logger=logger,
    devices=TRAINER_CONFIG.get("devices"),
    accelerator=TRAINER_CONFIG.get("accelerator"),
    strategy="ddp",
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)


if __name__ == "__main__":
    model = SimCSEModel(config_path=CONFIG_PATH)
    datamodule = SimCSEDataModule(config_path=CONFIG_PATH)
    trainer.fit(model=model, datamodule=datamodule)
