"""
第一阶段训练脚本
================
按老师要求：
  - 特征：开盘、收盘、最高、最低、成交量（5维原始数据）
  - 划分：70% 训练集 / 30% 测试集
  - 模型：TCN-LSTM-Transformer + H-MoE

用法：
    python scripts/train_phase1.py
    python scripts/train_phase1.py --epochs 100 --window 30
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from src.models.tcn_lstm_transformer import TCNLSTMTransformer
from src.models.hmoe import HMoE


# ─────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/raw/moutai_price.csv")
    p.add_argument("--window",  type=int,   default=30,    help="滑窗长度（天）")
    p.add_argument("--epochs",  type=int,   default=50)
    p.add_argument("--batch",   type=int,   default=32)
    p.add_argument("--lr",      type=float, default=0.001)
    p.add_argument("--device",  default="cpu")
    return p.parse_args()


# ─────────────────────────────────────────────
# 数据准备
# ─────────────────────────────────────────────

FEATURE_COLS = ["open", "close", "high", "low", "volume"]   # 老师要求的5个特征
TARGET_COL   = "close"                                        # 预测目标：次日收盘价

COL_RENAME = {
    "日期": "date", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low",  "成交量": "volume",
}


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.rename(columns=COL_RENAME)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS)
    return df


def make_windows(data: np.ndarray, window: int):
    """切成 (样本数, window, 5) 的滑窗，y 为下一个时间步的 close（index=1）"""
    X, y = [], []
    close_idx = FEATURE_COLS.index(TARGET_COL)
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window, close_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_datasets(csv_path: str, window: int, batch_size: int):
    df    = load_data(csv_path)
    values = df[FEATURE_COLS].values.astype(np.float32)

    # 70 / 30 划分
    n       = len(values)
    n_train = int(n * 0.7)
    train_raw = values[:n_train]
    test_raw  = values[n_train:]

    # MinMaxScaler 只在训练集上 fit
    scaler      = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled  = scaler.transform(test_raw)

    X_train, y_train = make_windows(train_scaled, window)
    X_test,  y_test  = make_windows(test_scaled,  window)

    print(f"  总交易日: {n}  训练: {n_train}({n_train/n:.0%})  测试: {n-n_train}({(n-n_train)/n:.0%})")
    print(f"  训练样本: {len(X_train)}  测试样本: {len(X_test)}  特征维度: {X_train.shape[-1]}")

    def to_loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(-1))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (
        to_loader(X_train, y_train, shuffle=True),
        to_loader(X_test,  y_test,  shuffle=False),
        scaler,
    )


# ─────────────────────────────────────────────
# 评估（反归一化后计算指标）
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, scaler, device):
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        out = model(X.to(device))
        preds.append(out["price"].cpu().numpy())
        trues.append(y.cpu().numpy())

    preds = np.concatenate(preds).ravel()
    trues = np.concatenate(trues).ravel()

    # 反归一化：只还原 close 列（index=1）
    close_idx = FEATURE_COLS.index(TARGET_COL)
    dummy_p = np.zeros((len(preds), len(FEATURE_COLS)))
    dummy_t = np.zeros((len(trues), len(FEATURE_COLS)))
    dummy_p[:, close_idx] = preds
    dummy_t[:, close_idx] = trues
    preds_real = scaler.inverse_transform(dummy_p)[:, close_idx]
    trues_real = scaler.inverse_transform(dummy_t)[:, close_idx]

    mse = np.mean((preds_real - trues_real) ** 2)
    mae = np.mean(np.abs(preds_real - trues_real))
    r2  = r2_score(trues_real, preds_real)
    return mse, mae, r2, preds_real, trues_real


# ─────────────────────────────────────────────
# 模型
# ─────────────────────────────────────────────

class Phase1Model(nn.Module):
    def __init__(self, in_features: int = 5, feat_dim: int = 64):
        super().__init__()
        self.backbone = TCNLSTMTransformer(
            in_features=in_features,
            tcn_hidden=32,       # 5个特征，模型适当缩小
            lstm_hidden=128,
            n_lstm_layers=2,
            out_dim=feat_dim,
        )
        self.hmoe = HMoE(in_dim=feat_dim, expert_hidden=64, out_dim=32)

    def forward(self, x):
        return self.hmoe(self.backbone(x))


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"\n{'='*52}")
    print(f"  第一阶段训练：5维原始特征 × TCN-LSTM-Transformer")
    print(f"  特征: {FEATURE_COLS}")
    print(f"  划分: 70% 训练 / 30% 测试  窗口: {args.window}天")
    print(f"{'='*52}\n")

    # ── 数据
    print("[1/4] 加载数据...")
    train_loader, test_loader, scaler = prepare_datasets(
        args.data, args.window, args.batch
    )

    # ── 模型
    print("\n[2/4] 构建模型...")
    model     = Phase1Model(in_features=len(FEATURE_COLS)).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # ── 训练
    print(f"\n[3/4] 开始训练（{args.epochs} 轮）\n")
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out["price"], y) + out["aux_loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/phase1_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}  train_loss={avg_loss:.6f}"
                  + (" ← 最佳" if avg_loss == best_loss else ""))

    # ── 评估
    print("\n[4/4] 测试集评估（最佳权重）...")
    model.load_state_dict(torch.load("checkpoints/phase1_best.pt", map_location=device))
    mse, mae, r2, preds, trues = evaluate(model, test_loader, scaler, device)

    print(f"\n{'='*52}")
    print(f"  测试集结果（真实价格，单位：元）")
    print(f"  MSE  : {mse:.4f}")
    print(f"  MAE  : {mae:.4f}  元")
    print(f"  R²   : {r2:.4f}")
    print(f"{'='*52}")

    # 保存预测结果对比
    result_df = pd.DataFrame({"真实收盘价": trues, "预测收盘价": preds})
    result_df.to_csv("data/processed/phase1_predictions.csv",
                     index=False, encoding="utf-8-sig")
    print(f"\n  预测结果已保存 → data/processed/phase1_predictions.csv")
    print(f"  模型权重已保存 → checkpoints/phase1_best.pt")


if __name__ == "__main__":
    main()
