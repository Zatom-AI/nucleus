"""Lightweight datasets used by multiprocessing checkpoint tests.

Keeping worker-side fixtures outside the trainer test module avoids importing
Lightning and Foundry in every spawned data-loader process.
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset


class WorkerRandomDataset(Dataset):
    """Draw from every RNG used by ordinary worker transforms."""

    def __len__(self) -> int:
        # Four samples exercise both workers before and after the checkpoint.
        return 4

    def __getitem__(self, index: int) -> tuple[int, float, float, float]:
        return (
            index,
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
        )
