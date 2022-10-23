"""
DiffCSE dataset

Author: DaeHyeon Gi
"""

from random import choice

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl

from utils import (
    get_config,
    load_multinli_data,
    load_snli_data
    )


class UnsupervisedDataset(Dataset):
    def __init__(self, is_train: bool, config: dict) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("tokenizer"))
        if is_train:
            multinli = load_multinli_data("./data/korNLI/multinli.train.ko.tsv", is_sup=False)
            snli = load_snli_data("./data/korNLI/snli_1.0_train.ko.tsv", is_sup=False)
            self.data = multinli + snli
            del multinli, snli
        else:
            dev_data = load_snli_data("./data/korNLI/xnli.dev.ko.tsv", is_sup=False)
            test_data = load_snli_data("./data/korNLI/xnli.test.ko.tsv", is_sup=False)
            self.data = dev_data + test_data
            del dev_data, test_data

    def tokenize(self, text) -> dict:
        return self.tokenizer(
            text=text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.config.get("max_length"),
            truncation=True
        )

    def __getitem__(self, idx) -> tuple:
        return (
            {k: v.squeeze() for k, v in self.tokenize(self.data[idx]).items()},
            {k: v.squeeze() for k, v in self.tokenize(self.data[idx]).items()},
            {k: v.squeeze() for k, v in self.tokenize(choice(self.data)).items()}
            )

    def __len__(self) -> int:
        return len(self.data)


class SupervisedDataset(Dataset):
    def __init__(self, is_train: bool, config: dict) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("tokenizer"))
        if is_train:
            multinli = load_multinli_data("./data/korNLI/multinli.train.ko.tsv", is_sup=True)
            snli = load_snli_data("./data/korNLI/snli_1.0_train.ko.tsv", is_sup=True)
            self.data = multinli + snli
            del multinli, snli
        else:
            dev_data = load_snli_data("./data/korNLI/xnli.dev.ko.tsv", is_sup=True)
            test_data = load_snli_data("./data/korNLI/xnli.test.ko.tsv", is_sup=True)
            self.data = dev_data + test_data
            del dev_data, test_data

    def tokenize(self, text) -> dict:
        return self.tokenizer(
            text=text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.config.get("max_length"),
            truncation=True
        )

    def __getitem__(self, idx) -> tuple:
        return (
            {k: v.squeeze() for k, v in self.tokenize(self.data[idx][0]).items()},
            {k: v.squeeze() for k, v in self.tokenize(self.data[idx][1]).items()},
            {k: v.squeeze() for k, v in self.tokenize(self.data[idx][2]).items()}
            )

    def __len__(self) -> int:
        return len(self.data)


class SimCSEDataModule(pl.LightningDataModule):
    def __init__(self, config_path: str) -> None:
        super().__init__()
        self.config = get_config(config_path, "datamodule")
        if self.config.get("is_sup"):
            self.train_dataset = SupervisedDataset(
                is_train=True, config=self.config
                )
            self.val_dataset = SupervisedDataset(
                is_train=False, config=self.config
                )
        else:
            self.train_dataset = UnsupervisedDataset(
                is_train=True, config=self.config
                )
            self.val_dataset = UnsupervisedDataset(
                is_train=False, config=self.config
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.get("batch_size"),
            shuffle=True,
            num_workers=self.config.get("num_workers"),
            drop_last=self.config.get("drop_last"),
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.get("batch_size") * 2,
            shuffle=False,
            num_workers=self.config.get("num_workers"),
            drop_last=False,
            pin_memory=True
        )
