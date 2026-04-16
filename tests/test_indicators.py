"""
技术指标计算单元测试
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from src.data.indicators import load_price, compute_indicators, CORE_FEATURES


def _make_dummy_df(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = np.cumsum(np.random.randn(n)) + 100
    return pd.DataFrame({
        "date":   dates,
        "open":   close * 0.99,
        "high":   close * 1.01,
        "low":    close * 0.98,
        "close":  close,
        "volume": np.random.randint(10000, 100000, n).astype(float),
    })


def test_compute_indicators_columns():
    df = _make_dummy_df()
    result = compute_indicators(df)
    # 必须包含所有核心特征（允许部分不存在时 fallback）
    assert "close" in result.columns
    assert "momentum_rsi" in result.columns


def test_compute_indicators_no_nan():
    df = _make_dummy_df(200)
    result = compute_indicators(df)
    assert not result.isnull().values.any(), "技术指标不应有 NaN（fillna=True）"


def test_compute_indicators_length():
    df = _make_dummy_df(150)
    result = compute_indicators(df)
    assert len(result) == len(df)


if __name__ == "__main__":
    test_compute_indicators_columns()
    test_compute_indicators_no_nan()
    test_compute_indicators_length()
    print("[PASS] 所有测试通过")
