"""
数据预处理模块
负责：滑窗切分、MinMax 标准化、训练/验证/测试集划分。
划分比例参照孙惠莹(2025) §3.2：6 : 1.5 : 2.5
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


SPLIT_RATIO = (0.60, 0.15, 0.25)   # train / val / test


def make_windows(
    data: np.ndarray,
    window: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """将时序数据切成 (样本数, window, 特征数) 的滑窗张量。"""
    X, y = [], []
    for i in range(len(data) - window - horizon + 1):
        X.append(data[i : i + window])
        y.append(data[i + window + horizon - 1, 3])  # 第4列 = close
    return np.array(X), np.array(y)


def split_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    window: int = 30,
    horizon: int = 1,
    ratio: tuple[float, float, float] = SPLIT_RATIO,
) -> dict:
    """
    Returns:
        dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
                        scaler, dates_test
    """
    values = df[feature_cols].values.astype(np.float32)

    n = len(values)
    n_train = int(n * ratio[0])
    n_val   = int(n * ratio[1])

    train_raw = values[:n_train]
    val_raw   = values[n_train : n_train + n_val]
    test_raw  = values[n_train + n_val :]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled   = scaler.transform(val_raw)
    test_scaled  = scaler.transform(test_raw)

    X_tr, y_tr = make_windows(train_scaled, window, horizon)
    X_va, y_va = make_windows(val_scaled,   window, horizon)
    X_te, y_te = make_windows(test_scaled,  window, horizon)

    dates = df["date"].values if "date" in df.columns else None

    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val":   X_va, "y_val":   y_va,
        "X_test":  X_te, "y_test":  y_te,
        "scaler":  scaler,
        "dates_test": dates[n_train + n_val + window:] if dates is not None else None,
    }
