"""
DiffCSE Model

Author: DaeHyeon Gi
"""

from pathlib import Path

import yaml
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel


class Encoder(nn.Module):
    """DiffCSE's Encoder"""
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.get("weight"))

        if config.get("freeze"):
            for name, child in self.encoder.named_children():
                if name != "pooler":
                    for param in child.parameters():
                        param.requires_grad = False

    def forward(self, inputs):
        return self.model(**inputs)


class SimCSEModel(pl.LightningModule):
    def __init__(self, config_path: str) -> None:
        super().__init__()
        with Path(config_path).open("r", encoding="utf8") as cf:
            self.config = yaml.load(stream=cf, Loader=yaml.FullLoader)
        self.encoder = Encoder(
            config=self.config.get("SimCSE")
        )

    def configure_optimizers(self):
        return 
