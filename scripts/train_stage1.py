"""
阶段一训练脚本：TCN-LSTM-Transformer 时序基线
================================================
只用支路一（技术指标时序），验证模型能否学到茅台价格规律。
达标后再逐步加入情感、图像、文本（阶段二/三/四）。

用法：
    python scripts/train_stage1.py
    python scripts/train_stage1.py --epochs 100 --window 30
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
from sklearn.metrics import r2_score

from src.data.indicators import load_price, compute_indicators
from src.data.preprocess  import split_dataset
from src.models.tcn_lstm_transformer import TCNLSTMTransformer


# ─────────────────────────────────────────────
# 命令行参数（可覆盖 config.yaml 中的设置）
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="configs/config.yaml")
    p.add_argument("--epochs",  type=int,   default=None)
    p.add_argument("--window",  type=int,   default=None)
    p.add_argument("--lr",      type=float, default=None)
    p.add_argument("--batch",   type=int,   default=None)
    p.add_argument("--device",  default=None)
    return p.parse_args()


# ─────────────────────────────────────────────
# 评估指标
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, scaler, n_features, device):
    """
    返回：MSE, MAE, R²（反归一化到真实价格，与论文对比口径一致）
    """
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())

    preds = np.concatenate(preds).reshape(-1, 1)
    trues = np.concatenate(trues).reshape(-1, 1)

    # 反归一化：只还原 close 列（第4列，index=3）
    dummy = np.zeros((len(preds), n_features))
    dummy[:, 3] = preds[:, 0]
    preds_real = scaler.inverse_transform(dummy)[:, 3]

    dummy[:, 3] = trues[:, 0]
    trues_real = scaler.inverse_transform(dummy)[:, 3]

    mse = np.mean((preds_real - trues_real) ** 2)
    mae = np.mean(np.abs(preds_real - trues_real))
    r2  = r2_score(trues_real, preds_real)
    return mse, mae, r2


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 命令行参数覆盖配置文件
    epochs     = args.epochs or cfg["training"]["epochs"]
    window     = args.window or cfg["data"]["window"]
    lr         = args.lr     or cfg["training"]["lr"]
    batch_size = args.batch  or cfg["training"]["batch_size"]
    patience   = cfg["training"]["early_stopping_patience"]
    device     = torch.device(
        args.device or (cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    )

    print(f"\n{'='*50}")
    print(f"  阶段一训练：TCN-LSTM-Transformer 时序基线")
    print(f"  设备={device}  窗口={window}  epochs={epochs}  lr={lr}")
    print(f"{'='*50}\n")

    # ── 数据准备
    print("[1/4] 加载数据...")
    df = load_price(cfg["data"]["price_csv"])
    df = compute_indicators(df)
    feature_cols = [c for c in df.columns if c != "date"]
    n_features   = len(feature_cols)
    print(f"      特征维度: {n_features}  总交易日: {len(df)}")

    splits = split_dataset(
        df, feature_cols,
        window=window,
        horizon=cfg["data"]["horizon"],
        ratio=tuple(cfg["data"]["split_ratio"]),
    )

    def to_loader(X, y, shuffle):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    train_loader = to_loader(splits["X_train"], splits["y_train"], shuffle=True)
    val_loader   = to_loader(splits["X_val"],   splits["y_val"],   shuffle=False)
    test_loader  = to_loader(splits["X_test"],  splits["y_test"],  shuffle=False)
    scaler       = splits["scaler"]

    print(f"      训练样本: {len(splits['X_train'])}  "
          f"验证: {len(splits['X_val'])}  "
          f"测试: {len(splits['X_test'])}")

    # ── 模型
    print("\n[2/4] 构建模型...")
    backbone = TCNLSTMTransformer(
        in_features=n_features,
        out_dim=cfg["model"]["feat_dim"],
    ).to(device)
    head  = nn.Linear(cfg["model"]["feat_dim"], 1).to(device)
    model = nn.Sequential(backbone, head)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      可训练参数: {n_params:,}")

    # ── 优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()

    # ── 训练循环
    print(f"\n[3/4] 开始训练（最多 {epochs} 轮，early stop patience={patience}）\n")
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # 训练
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"train={train_loss:.6f}  val={val_loss:.6f} | "
              f"{elapsed:.1f}s", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_stage1.pt")
            print("  ← 最佳")
        else:
            patience_counter += 1
            print()
            if patience_counter >= patience:
                print(f"\n  [Early Stop] 连续 {patience} 轮无改善，停止。")
                break

    # ── 测试评估
    print("\n[4/4] 测试集评估（加载最佳权重）...")
    model.load_state_dict(torch.load("checkpoints/best_stage1.pt", map_location=device))
    mse, mae, r2 = evaluate(model, test_loader, scaler, n_features, device)

    print(f"\n{'='*50}")
    print(f"  测试集结果（真实价格，元）")
    print(f"  MSE : {mse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  R²  : {r2:.4f}")
    print(f"{'='*50}")
    print(f"\n  权重已保存 → checkpoints/best_stage1.pt")

    # 孙惠莹论文对比基准（第三章，孙的单体LSTM R²=0.42，TCN-LSTM-Transformer R²=0.68）
    if r2 >= 0.68:
        print(f"  达到论文 TCN-LSTM-Transformer 基准（R²≥0.68）✓")
    elif r2 >= 0.42:
        print(f"  超过 LSTM 基线（R²≥0.42），但未达到论文完整架构水平")
    else:
        print(f"  低于 LSTM 基线，建议检查数据预处理或增加训练轮数")


if __name__ == "__main__":
    main()
