from typing import Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, Trainer, LightningModule
from pytorch_lightning.tuner import Tuner
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    def __init__(self, size, length: int):
        self.len = length
        self.data = torch.randn(length, *size)

    def __getitem__(self, index: int):
        return {"data": self.data[index], "targets": index % 3, "session_idx": index % 2}

    def __len__(self) -> int:
        return self.len


class DummyDataModule(LightningDataModule):
    def __init__(self, data_set):
        super().__init__()
        self.data_set = data_set
        self.batch_size = 1

    def train_dataloader(self):
        return DataLoader(self.data_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_set, batch_size=self.batch_size)


def custom_batch_size_tuner(sample_shape: Tuple[int, ...], initial_batch_size: int, mode: str, max_trials: int, tuner: Tuner, lightning_model: LightningModule, max_batch_size: Optional[int] = None):
    dummy_ds = DummyDataset(sample_shape, length=max_batch_size or 100_000)

    lightning_model.batch_tuning_mode = True
    tune_results = tuner.scale_batch_size(model=lightning_model, datamodule=DummyDataModule(dummy_ds),
                                init_val=initial_batch_size, mode=mode, max_trials=max_trials)
    lightning_model.batch_tuning_mode = False

    new_batch_size = tune_results

    return min(new_batch_size, max_batch_size or np.inf)
