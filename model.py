"""
DiffCSE Model

Author: DaeHyeon Gi
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from scipy.stats import spearmanr, pearsonr

from utils import (
    get_sts_dataset,
    get_config
    )


class Encoder(nn.Module):
    """DiffCSE's Encoder"""
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.get("weight"))

    def forward(self, tokens):
        return self.encoder(
                input_ids=tokens.get("input_ids"),
                token_type_ids=tokens.get("token_type_ids"),
                attention_mask=tokens.get("attention_mask")
            )


class SimCSELoss(nn.Module):
    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.metric = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, query, pos, neg):
        pos_sim = self.cos(query.unsqueeze(1), pos.unsqueeze(0)) / self.temperature
        neg_sim = self.cos(query.unsqueeze(1), neg.unsqueeze(0)) / self.temperature
        cosine_sim = torch.cat([pos_sim, neg_sim], dim=1).cuda()
        labels = torch.arange(cosine_sim.size(0)).long().cuda()
        return self.metric(cosine_sim, labels)


class SimCSEModel(pl.LightningModule):
    def __init__(self, config_path: str) -> None:
        super().__init__()
        self.max_length = get_config(config_path, "datamodule").get("max_length")
        self.hparam = get_config(config_path, "hparam")
        self.encoder = Encoder(get_config(config_path, "encoder"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            get_config(config_path, "datamodule").get("tokenizer")
            )
        self.loss_fn = SimCSELoss(self.hparam.get("temperature"))

    def forward(self, tokens):
        return self.encoder(tokens)["last_hidden_state"][:, 0, :]

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparam.get("optimizer"))
        return optimizer(
            self.encoder.parameters(), lr=float(self.hparam.get("learning_rate"))
            )

    def _step(self, batch):
        embeddings = [self(b) for b in batch]
        return self.loss_fn(*embeddings)

    def tokenize(self, text) -> dict:
        return self.tokenizer(
            text=text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length
        )

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)

        if batch_idx % 100 == 1:
            self.sts_validate()

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        val_loss = self._step(batch)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True,
                 batch_size=self.hparam.get("batch_size"), sync_dist=True)
        return {"val_loss": val_loss}

    def sts_validate(self, *args, **kwargs) -> None:
        # STS Dataset Validate
        self.freeze()
        sts_data = get_sts_dataset("./data/korSTS/sts-test.tsv")
        y_true = []
        y_hat = torch.Tensor()
        for d in sts_data:
            y_true.append(d[0])
            emb_1 = self({k: v.cuda() for k, v in self.tokenize(d[1]).items()})
            emb_2 = self({k: v.cuda() for k, v in self.tokenize(d[2]).items()})
            y_hat = torch.concat([y_hat, torch.cosine_similarity(emb_1, emb_2).cpu()])
        spearman, _ = spearmanr(np.array(y_true), y_hat.cpu().numpy())
        pearson, _ = pearsonr(np.array(y_true), y_hat.cpu().numpy())
        self.log("SpearmanR", spearman, sync_dist=True)
        self.log("PearsonR", pearson, sync_dist=True)
        self.unfreeze()
