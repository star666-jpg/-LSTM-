"""
TIFS（时序图像特征序列）生成脚本（只需运行一次）

编码方式（参照昝泓含2024）：
  对每个 T 天滑窗，将 (T, F) 特征矩阵按语义分三组，
  每组独立归一化后 resize 到固定尺寸，构成 RGB 图像的三个通道：

    Channel R（价格/趋势，11维）：open high low close volume
                                   SMA EMA MACD MACD_signal CCI ADX
    Channel G（动量，3维）：        RSI Stochastic ROC
    Channel B（波动/成交量，5维）：  BB_high BB_low ATR MFI OBV

  每通道 (T, n_group) → 窗口内逐列 MinMax → bilinear resize → (H, W)
  最终图像尺寸：(3, IMG_H, IMG_W)，默认 64×64

输出：
  data/processed/tifs_images.npy   float32  (n_windows, 3, H, W)
  data/processed/tifs_splits.npz            train/val/test 的窗口索引
  data/processed/tifs_labels.npy   float32  (n_windows,)  归一化收盘价
  data/processed/tifs_dirs.npy     int64    (n_windows,)  涨跌标签

用法：
  python scripts/generate_tifs.py
  python scripts/generate_tifs.py --window 30 --img_size 64
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

from src.data.indicators import load_price, compute_indicators


# ─────────────────────────────────────────────
# 特征分组（与 CORE_FEATURES 顺序对应）
# ─────────────────────────────────────────────

# 去掉 "date" 后的特征列顺序：
# 0:open 1:high 2:low 3:close 4:volume
# 5:trend_sma_fast 6:trend_ema_fast 7:trend_macd 8:trend_macd_signal 9:trend_cci 10:trend_adx
# 11:momentum_rsi 12:momentum_stoch 13:momentum_roc
# 14:volatility_bbh 15:volatility_bbl 16:volatility_atr
# 17:volume_mfi 18:volume_obv

FEATURE_GROUPS = {
    "price_trend":    list(range(0, 11)),   # Channel R
    "momentum":       list(range(11, 14)),  # Channel G
    "vol_volatility": list(range(14, 19)),  # Channel B（实际列数可能<5，容错处理）
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default="configs/config.yaml")
    p.add_argument("--out_dir",   default="data/processed")
    p.add_argument("--window",    type=int, default=None)
    p.add_argument("--img_size",  type=int, default=64,  help="输出图像的 H 和 W")
    return p.parse_args()


# ─────────────────────────────────────────────
# 核心编码函数
# ─────────────────────────────────────────────

def _encode_channel(mat: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """
    (T, n_feat) → 窗口内逐列归一化 → bilinear resize → (img_h, img_w) float32 [0,1]
    """
    # 逐列 MinMax
    vmin = mat.min(axis=0, keepdims=True)
    vmax = mat.max(axis=0, keepdims=True)
    denom = np.where(vmax - vmin < 1e-8, 1.0, vmax - vmin)
    mat = (mat - vmin) / denom   # [0,1]

    # bilinear resize via torch（已是项目依赖，无需额外安装）
    t = torch.tensor(mat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,T,F)
    t = F.interpolate(t, size=(img_h, img_w), mode="bilinear", align_corners=False)
    return t.squeeze().numpy().astype(np.float32)   # (img_h, img_w)


def encode_window(
    window: np.ndarray,     # (T, n_features)
    img_h: int,
    img_w: int,
    groups: dict,
) -> np.ndarray:
    """
    将单个滑窗编码为 (3, H, W) RGB 图像。
    groups 中超出实际列数的索引会被自动截断。
    """
    n_feat = window.shape[1]
    channels = []
    for name, idxs in groups.items():
        valid = [i for i in idxs if i < n_feat]
        if not valid:
            channels.append(np.zeros((img_h, img_w), dtype=np.float32))
            continue
        channels.append(_encode_channel(window[:, valid], img_h, img_w))
    return np.stack(channels, axis=0)   # (3, H, W)


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    window   = args.window  or cfg["data"]["window"]
    img_h    = img_w = args.img_size
    ratio    = cfg["data"]["split_ratio"]   # [0.60, 0.15, 0.25]
    out_dir  = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── 加载并归一化技术指标 ──
    print("加载技术指标...")
    df = compute_indicators(load_price(cfg["data"]["price_csv"]))
    feature_cols = [c for c in df.columns if c != "date"]
    values = df[feature_cols].values.astype(np.float32)   # (n_days, n_feat)
    n, n_feat = values.shape

    # 整体 MinMax（与 train_stage1 保持一致）
    scaler = MinMaxScaler()
    n_train = int(n * ratio[0])
    n_val   = int(n * ratio[1])
    values[:n_train] = scaler.fit_transform(values[:n_train])
    values[n_train:n_train+n_val] = scaler.transform(values[n_train:n_train+n_val])
    values[n_train+n_val:]        = scaler.transform(values[n_train+n_val:])

    # ── 滑窗生成图像 & 标签 ──
    print(f"生成 TIFS 图像（window={window}, img={img_h}×{img_w}）...")
    images, labels_price, labels_dir = [], [], []

    for i in range(n - window):
        win   = values[i : i + window]             # (T, F)
        img   = encode_window(win, img_h, img_w, FEATURE_GROUPS)
        label = values[i + window, 3]              # 下一日 close（归一化后）
        images.append(img)
        labels_price.append(label)

    images       = np.stack(images, axis=0).astype(np.float32)   # (N, 3, H, W)
    labels_price = np.array(labels_price, dtype=np.float32)       # (N,)
    labels_dir   = np.zeros(len(labels_price), dtype=np.int64)
    labels_dir[1:] = (labels_price[1:] > labels_price[:-1]).astype(np.int64)

    # ── 划分 train/val/test 索引 ──
    n_windows  = len(images)
    n_tr = int(n_windows * ratio[0])
    n_va = int(n_windows * ratio[1])
    splits = {
        "train": np.arange(0,        n_tr),
        "val":   np.arange(n_tr,     n_tr + n_va),
        "test":  np.arange(n_tr + n_va, n_windows),
    }

    # ── 保存 ──
    out_img    = os.path.join(out_dir, "tifs_images.npy")
    out_labels = os.path.join(out_dir, "tifs_labels.npy")
    out_dirs   = os.path.join(out_dir, "tifs_dirs.npy")
    out_splits = os.path.join(out_dir, "tifs_splits.npz")

    np.save(out_img,    images)
    np.save(out_labels, labels_price)
    np.save(out_dirs,   labels_dir)
    np.savez(out_splits, **splits)

    print(f"已保存：{out_img}  shape={images.shape}")
    print(f"        train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")


if __name__ == "__main__":
    main()
