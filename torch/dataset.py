"""
DiffCSE dataset

Author: DaeHyeon Gi
"""

from pathlib import Path

from torch.utils.data import Dataset


class UnsupervisedDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = []

    def __getitem__(self, index) -> tuple:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return len(self.data)


class SupervisedDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
