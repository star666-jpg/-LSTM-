"""
TCN-LSTM-Transformer 单元测试
覆盖：形状、因果性、梯度、真实数据前向传播
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pandas as pd
import numpy as np
import pytest

from src.models.tcn_lstm_transformer import TCN, TCNLSTMTransformer
from src.data.indicators import load_price, compute_indicators
from src.data.preprocess import split_dataset


# ─────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────

def _make_batch(B=4, T=64, F=20):
    return torch.randn(B, T, F)


# ─────────────────────────────────────────────
# TCN 单独测试
# ─────────────────────────────────────────────

def test_tcn_output_shape():
    model = TCN(in_features=20, hidden=32, n_levels=4)
    x = _make_batch(4, 64, 20)
    out = model(x)
    assert out.shape == (4, 64, 32), f"期望 (4,64,32)，得到 {out.shape}"


def test_tcn_causality():
    """改变未来时间步，不影响过去时间步的输出。"""
    model = TCN(in_features=20, hidden=32, n_levels=4)
    model.eval()

    x = _make_batch(2, 32, 20)
    x2 = x.clone()
    x2[:, 20:, :] += 1000   # 修改 t>=20 的输入

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x2)

    diff = (out1[:, :20, :] - out2[:, :20, :]).abs().max().item()
    assert diff < 1e-5, f"因果性违反！前20步输出差异={diff:.6f}"


def test_tcn_gradient_flow():
    """梯度能正常反向传播到输入层。"""
    model = TCN(in_features=10, hidden=16, n_levels=3)
    x = _make_batch(2, 30, 10).requires_grad_(True)
    out = model(x).sum()
    out.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


# ─────────────────────────────────────────────
# TCNLSTMTransformer 测试
# ─────────────────────────────────────────────

def test_full_model_output_shape():
    model = TCNLSTMTransformer(in_features=20, out_dim=128)
    x = _make_batch(8, 64, 20)
    out = model(x)
    assert out.shape == (8, 128), f"期望 (8,128)，得到 {out.shape}"


def test_full_model_different_batch_sizes():
    model = TCNLSTMTransformer(in_features=15, out_dim=64)
    model.eval()
    for B in [1, 4, 16]:
        x = _make_batch(B, 30, 15)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, 64)


def test_full_model_gradient_flow():
    model = TCNLSTMTransformer(in_features=20, out_dim=64)
    x = _make_batch(4, 64, 20).requires_grad_(True)
    loss = model(x).sum()
    loss.backward()
    assert x.grad is not None


def test_full_model_param_count():
    """参数量应在合理范围内（不应超过 5M，避免过拟合720样本）。"""
    model = TCNLSTMTransformer(in_features=20)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  可训练参数量: {n_params:,}")
    assert n_params < 5_000_000, f"参数量过大: {n_params:,}"


# ─────────────────────────────────────────────
# 真实数据端到端测试
# ─────────────────────────────────────────────

PRICE_CSV = "data/raw/moutai_price.csv"
SKIP_NO_DATA = not os.path.exists(PRICE_CSV)

@pytest.mark.skipif(SKIP_NO_DATA, reason="本地无行情数据")
def test_real_data_forward():
    """用真实茅台数据做一次前向传播，验证整条数据管道。"""
    df = load_price(PRICE_CSV)
    df_feat = compute_indicators(df)

    feature_cols = [c for c in df_feat.columns if c != "date"]
    splits = split_dataset(df_feat, feature_cols, window=64)

    X_train = torch.tensor(splits["X_train"], dtype=torch.float32)
    print(f"\n  训练集形状: {X_train.shape}")   # (samples, 64, n_features)

    model = TCNLSTMTransformer(in_features=X_train.shape[-1], out_dim=128)
    model.eval()
    with torch.no_grad():
        # 取前4个样本跑一次
        out = model(X_train[:4])
    assert out.shape == (4, 128), f"真实数据输出形状错误: {out.shape}"
    assert not torch.isnan(out).any(), "输出中含有 NaN"
    print(f"  输出形状: {out.shape}  ✓")


if __name__ == "__main__":
    # 不安装 pytest 也可以直接运行
    tests = [
        test_tcn_output_shape,
        test_tcn_causality,
        test_tcn_gradient_flow,
        test_full_model_output_shape,
        test_full_model_different_batch_sizes,
        test_full_model_gradient_flow,
        test_full_model_param_count,
    ]
    if not SKIP_NO_DATA:
        tests.append(test_real_data_forward)

    os.chdir(os.path.dirname(os.path.dirname(__file__)))  # 切到项目根目录
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 通过")
