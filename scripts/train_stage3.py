"""
阶段三训练：支路1（技术指标时序）+ 支路3（BERT文本）→ H-MoE
================================================================
依赖 precompute_bert.py 已生成 data/processed/bert_cls_daily.npy

数据流：
    ts_window   (B, T, ts_dim)   → TCNLSTMTransformer → ts_feat (B, 128)
    bert_window (B, T, 768)      → GFM 逐步融合      → text_feat (B, 128)
    concat [ts_feat, text_feat]  → (B, 256) → HMoE(in_dim=256) → price / direction

用法：
    python scripts/train_stage3.py
    python scripts/train_stage3.py --epochs 80 --window 30 --device cuda
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, accuracy_score

from src.data.indicators   import load_price, compute_indicators
from src.data.preprocess   import split_dataset
from src.models.tcn_lstm_transformer import TCNLSTMTransformer
from src.models.bert_gfm   import GFM
from src.models.hmoe       import HMoE


# ─────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="configs/config.yaml")
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--window",       type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--batch",        type=int,   default=None)
    p.add_argument("--device",       default=None)
    p.add_argument("--bert_cls",     default="data/processed/bert_cls_daily.npy")
    p.add_argument("--bert_dates",   default="data/processed/bert_dates.csv")
    p.add_argument("--feat_dim",     type=int,   default=None)
    return p.parse_args()


# ─────────────────────────────────────────────
# Dataset：时序 + BERT CLS 序列
# ─────────────────────────────────────────────

class TextStockDataset(Dataset):
    """
    每个样本：
        ts_window   (T, ts_dim)  技术指标滑窗（已 MinMax 归一化）
        bert_window (T, 768)     对应日期的 BERT CLS 序列（未归一化，已 L2 norm）
        label_price float        下一日收盘价（原始尺度，用 scaler 还原后算 R²）
        label_dir   int          1=涨 0=跌
    """
    def __init__(
        self,
        ts_windows:   np.ndarray,   # (N, T, ts_dim)
        bert_windows: np.ndarray,   # (N, T, 768)
        labels_price: np.ndarray,   # (N,) 归一化后的收盘价
        labels_dir:   np.ndarray,   # (N,) 0/1
    ):
        self.ts   = torch.tensor(ts_windows,   dtype=torch.float32)
        self.bert = torch.tensor(bert_windows, dtype=torch.float32)
        self.lp   = torch.tensor(labels_price, dtype=torch.float32)
        self.ld   = torch.tensor(labels_dir,   dtype=torch.long)

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, idx):
        return self.ts[idx], self.bert[idx], self.lp[idx], self.ld[idx]


def make_direction_labels(prices: np.ndarray) -> np.ndarray:
    """归一化收盘价序列 → 涨跌标签（1=涨，0=跌/平）。"""
    dirs = np.zeros(len(prices), dtype=np.int64)
    dirs[1:] = (prices[1:] > prices[:-1]).astype(np.int64)
    return dirs


def build_bert_windows(
    bert_cls:    np.ndarray,       # (n_days, 768)
    bert_dates:  pd.DatetimeIndex, # 与 bert_cls 对应的日期
    ts_dates:    pd.DatetimeIndex, # 技术指标的全量日期（与 bert_cls 等长）
    window:      int,
    split_idx:   tuple[int, int],  # (n_train_rows, n_val_rows)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 bert_cls 按同样的滑窗逻辑切分成 (N, T, 768)，
    与 split_dataset 的划分方式保持一致。
    """
    n = len(bert_cls)
    n_train, n_val = split_idx

    def _windows(arr):
        X = []
        for i in range(len(arr) - window):
            X.append(arr[i: i + window])
        return np.array(X, dtype=np.float32)

    train_cls = bert_cls[:n_train]
    val_cls   = bert_cls[n_train: n_train + n_val]
    test_cls  = bert_cls[n_train + n_val:]

    # L2 归一化（稳定训练，不破坏方向信息）
    def _l2_norm(x):
        norms = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)
        return x / norms

    return (
        _windows(_l2_norm(train_cls)),
        _windows(_l2_norm(val_cls)),
        _windows(_l2_norm(test_cls)),
    )


# ─────────────────────────────────────────────
# Stage3 模型
# ─────────────────────────────────────────────

class Stage3Model(nn.Module):
    """
    支路1 + 支路3 双支路模型。
    ts_feat  (B,128) + text_feat (B,128) → concat (B,256) → HMoE(in_dim=256)
    """
    def __init__(
        self,
        ts_in_features: int,
        bert_hidden:    int = 768,
        feat_dim:       int = 128,
    ):
        super().__init__()
        self.ts_branch = TCNLSTMTransformer(in_features=ts_in_features, out_dim=feat_dim)
        self.gfm       = GFM(input_size=bert_hidden, hidden_size=feat_dim)
        self.gfm_proj  = nn.Linear(feat_dim, feat_dim)
        self.hmoe      = HMoE(in_dim=feat_dim * 2)   # 128+128=256

    def forward(
        self,
        ts_x:   torch.Tensor,   # (B, T, ts_dim)
        bert_x: torch.Tensor,   # (B, T, 768)
    ) -> dict:
        ts_feat = self.ts_branch(ts_x)   # (B, feat_dim)

        # GFM 逐步处理 T 个时间步的 BERT CLS
        h = None
        for t in range(bert_x.size(1)):
            gfm_out, h = self.gfm(bert_x[:, t, :], h)
        text_feat = self.gfm_proj(gfm_out)   # (B, feat_dim)

        fused = torch.cat([ts_feat, text_feat], dim=-1)   # (B, 256)
        return self.hmoe(fused)


# ─────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, scaler, n_ts_features, device):
    model.eval()
    preds_norm, labels_norm, dirs_pred, dirs_true = [], [], [], []

    for ts_x, bert_x, lp, ld in loader:
        ts_x   = ts_x.to(device)
        bert_x = bert_x.to(device)
        out    = model(ts_x, bert_x)

        price_pred = out["price"].squeeze(-1).cpu().numpy()
        dir_pred   = (torch.sigmoid(out["direction"].squeeze(-1)).cpu().numpy() > 0.5).astype(int)

        preds_norm.extend(price_pred.tolist())
        labels_norm.extend(lp.numpy().tolist())
        dirs_pred.extend(dir_pred.tolist())
        dirs_true.extend(ld.numpy().tolist())

    # 反归一化（close 在第4列）
    def _denorm(vals):
        dummy = np.zeros((len(vals), n_ts_features), dtype=np.float32)
        dummy[:, 3] = vals
        return scaler.inverse_transform(dummy)[:, 3]

    p = _denorm(np.array(preds_norm))
    l = _denorm(np.array(labels_norm))

    mse = float(np.mean((p - l) ** 2))
    mae = float(np.mean(np.abs(p - l)))
    r2  = float(r2_score(l, p))
    acc = float(accuracy_score(dirs_true, dirs_pred))
    return {"mse": mse, "mae": mae, "r2": r2, "acc": acc}


# ─────────────────────────────────────────────
# 主训练循环
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── 超参数（命令行优先于 config）──
    epochs    = args.epochs    or cfg["training"]["epochs"]
    window    = args.window    or cfg["data"]["window"]
    lr        = args.lr        or cfg["training"]["lr"]
    batch_sz  = args.batch     or cfg["training"]["batch_size"]
    feat_dim  = args.feat_dim  or cfg["model"]["feat_dim"]
    device    = args.device    or cfg["training"]["device"]
    w_decay   = cfg["training"]["weight_decay"]
    patience  = cfg["training"]["early_stopping_patience"]
    lw        = cfg["training"]["loss_weights"]
    device    = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    print(f"device={device}  epochs={epochs}  window={window}  lr={lr}")

    # ── 加载技术指标 ──
    df = compute_indicators(load_price(cfg["data"]["price_csv"]))
    feature_cols = [c for c in df.columns if c not in ("date",)]
    splits = split_dataset(df, feature_cols, window=window)

    n_ts = splits["X_train"].shape[-1]
    n    = len(df)
    n_train = int(n * 0.60)
    n_val   = int(n * 0.15)

    # ── 涨跌标签 ──
    all_prices_norm = np.concatenate([
        splits["y_train"], splits["y_val"], splits["y_test"]
    ])
    def _make_dir(y_norm, split_prices_norm):
        dirs = np.zeros(len(split_prices_norm), dtype=np.int64)
        dirs[1:] = (split_prices_norm[1:] > split_prices_norm[:-1]).astype(np.int64)
        return dirs

    dir_train = _make_dir(None, splits["y_train"])
    dir_val   = _make_dir(None, splits["y_val"])
    dir_test  = _make_dir(None, splits["y_test"])

    # ── 加载预计算 BERT CLS ──
    print(f"加载 BERT CLS：{args.bert_cls}")
    bert_cls   = np.load(args.bert_cls).astype(np.float32)    # (n_days, 768)
    bert_dates = pd.read_csv(args.bert_dates)["date"].values

    if len(bert_cls) != n:
        raise ValueError(
            f"bert_cls 行数 ({len(bert_cls)}) 与技术指标行数 ({n}) 不匹配，"
            "请重新运行 precompute_bert.py"
        )

    bert_tr, bert_va, bert_te = build_bert_windows(
        bert_cls, pd.to_datetime(bert_dates),
        pd.to_datetime(df["date"].values),
        window, (n_train, n_val),
    )

    # ── 构建 Dataset / DataLoader ──
    ds_train = TextStockDataset(splits["X_train"], bert_tr, splits["y_train"], dir_train)
    ds_val   = TextStockDataset(splits["X_val"],   bert_va, splits["y_val"],   dir_val)
    ds_test  = TextStockDataset(splits["X_test"],  bert_te, splits["y_test"],  dir_test)

    dl_train = DataLoader(ds_train, batch_size=batch_sz, shuffle=True,  drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_sz, shuffle=False)
    dl_test  = DataLoader(ds_test,  batch_size=batch_sz, shuffle=False)

    # ── 模型 ──
    model = Stage3Model(ts_in_features=n_ts, feat_dim=feat_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数：{n_params:,}")

    # 对 BERT（已冻结）不传给优化器，减少内存
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, verbose=True)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_r2  = -np.inf
    best_state   = None
    no_improve   = 0

    # ── 训练 ──
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0

        for ts_x, bert_x, lp, ld in dl_train:
            ts_x   = ts_x.to(device)
            bert_x = bert_x.to(device)
            lp     = lp.to(device)
            ld     = ld.float().to(device)

            out = model(ts_x, bert_x)
            loss_reg = mse_loss(out["price"].squeeze(-1), lp)
            loss_cls = bce_loss(out["direction"].squeeze(-1), ld)
            loss     = lw["regression"] * loss_reg + lw["classification"] * loss_cls + out["aux_loss"]

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dl_train)
        val_met  = evaluate(model, dl_val, splits["scaler"], n_ts, device)
        sched.step(-val_met["r2"])   # ReduceLROnPlateau 最小化 -R²

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss={avg_loss:.4f} | "
            f"val R²={val_met['r2']:.4f}  acc={val_met['acc']:.4f}  "
            f"MAE={val_met['mae']:.2f}  [{time.time()-t0:.1f}s]"
        )

        if val_met["r2"] > best_val_r2:
            best_val_r2 = val_met["r2"]
            best_state  = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping（{patience} 轮无提升）")
                break

    # ── 测试集评估 ──
    model.load_state_dict(best_state)
    model.to(device)
    test_met = evaluate(model, dl_test, splits["scaler"], n_ts, device)

    print("\n" + "=" * 50)
    print("阶段三测试集结果：")
    print(f"  MSE  = {test_met['mse']:.2f}")
    print(f"  MAE  = {test_met['mae']:.2f} 元")
    print(f"  R²   = {test_met['r2']:.4f}")
    print(f"  Acc  = {test_met['acc']:.4f}")
    print("=" * 50)

    # 对比
    print("\n与论文基准对比：")
    print(f"  单体 LSTM（论文基线）   R²=0.42")
    print(f"  TCN-LSTM-Transformer    R²=0.68")
    print(f"  阶段一（5维）          R²=0.70")
    print(f"  阶段三（本次+BERT）    R²={test_met['r2']:.4f}  ← 当前")
    print(f"  论文完整模型            R²=0.71")

    # 保存模型
    os.makedirs("data/processed", exist_ok=True)
    ckpt_path = "data/processed/stage3_model.pt"
    torch.save({"model_state": best_state, "metrics": test_met}, ckpt_path)
    print(f"\n模型已保存：{ckpt_path}")


if __name__ == "__main__":
    main()
