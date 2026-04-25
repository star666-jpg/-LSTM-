"""
TIFS 图像 Dataset
加载 generate_tifs.py 预计算的 .npy 文件，
配合 CS-ACNN 支路使用。

典型用法：
    from src.data.tifs_dataset import TIFSDataset, load_tifs_splits

    train_ds, val_ds, test_ds = load_tifs_splits("data/processed")
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    # batch: (images, labels_price, labels_dir)
    #         (B,3,H,W)  (B,)           (B,)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class TIFSDataset(Dataset):
    def __init__(
        self,
        images:       np.ndarray,   # (N, 3, H, W)
        labels_price: np.ndarray,   # (N,)
        labels_dir:   np.ndarray,   # (N,)
    ):
        self.images = torch.tensor(images,       dtype=torch.float32)
        self.lp     = torch.tensor(labels_price, dtype=torch.float32)
        self.ld     = torch.tensor(labels_dir,   dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.lp[idx], self.ld[idx]


def load_tifs_splits(
    processed_dir: str = "data/processed",
) -> tuple[TIFSDataset, TIFSDataset, TIFSDataset]:
    """
    读取预计算文件，返回 (train_ds, val_ds, test_ds)。
    """
    d = Path(processed_dir)
    images  = np.load(d / "tifs_images.npy")    # (N, 3, H, W)
    labels  = np.load(d / "tifs_labels.npy")    # (N,)
    dirs    = np.load(d / "tifs_dirs.npy")      # (N,)
    splits  = np.load(d / "tifs_splits.npz")

    def _ds(key):
        idx = splits[key]
        return TIFSDataset(images[idx], labels[idx], dirs[idx])

    return _ds("train"), _ds("val"), _ds("test")
