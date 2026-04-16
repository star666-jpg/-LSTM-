"""
训练入口脚本（仅时序支路 — 阶段一）
运行：python scripts/train.py

阶段规划：
  阶段一：仅训练支路1 (TCNLSTMTransformer) + H-MoE，验证时序基线
  阶段二：加入支路4 (情感得分)
  阶段三：加入支路2/3 (图像+文本)，完整四模态
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.data.indicators import build_indicator_dataset
from src.data.preprocess  import split_dataset
import pandas as pd


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict):
    df = pd.read_csv(cfg["data"]["indicator_csv"], parse_dates=["date"])
    feature_cols = [c for c in df.columns if c != "date"]

    splits = split_dataset(
        df, feature_cols,
        window=cfg["data"]["window"],
        horizon=cfg["data"]["horizon"],
        ratio=tuple(cfg["data"]["split_ratio"]),
    )

    def to_loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.float32).unsqueeze(-1))
        return DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=shuffle)

    return (
        to_loader(splits["X_train"], splits["y_train"], shuffle=True),
        to_loader(splits["X_val"],   splits["y_val"],   shuffle=False),
        to_loader(splits["X_test"],  splits["y_test"],  shuffle=False),
        splits["scaler"],
    )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        total_loss += criterion(pred, y_batch).item()
    return total_loss / len(loader)


def main():
    cfg    = load_config()
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # 阶段一：仅使用时序支路，轻量化验证
    from src.models.tcn_lstm_transformer import TCNLSTMTransformer

    train_loader, val_loader, test_loader, scaler = build_dataloaders(cfg)
    sample_X, _ = next(iter(train_loader))
    in_features = sample_X.shape[-1]

    # 阶段一用简单线性输出头
    backbone = TCNLSTMTransformer(in_features=in_features, out_dim=64).to(device)
    head     = nn.Linear(64, 1).to(device)
    model    = nn.Sequential(backbone, head)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    criterion = nn.MSELoss()

    best_val = float("inf")
    patience_counter = 0
    patience = cfg["training"]["early_stopping_patience"]

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = eval_epoch(model, val_loader,   criterion, device)
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_stage1.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Early Stop] 连续 {patience} 轮无改善，停止训练。")
                break

    test_loss = eval_epoch(model, test_loader, criterion, device)
    print(f"\n[Test] MSE = {test_loss:.6f}")


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()
