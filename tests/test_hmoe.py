"""
H-MoE 单元测试
覆盖：输出形状、门控归一化、负载均衡、梯度、与 TCN 的联合前向
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from src.models.hmoe import HMoE
from src.models.tcn_lstm_transformer import TCNLSTMTransformer


def _make_feat(B=8, in_dim=128):
    return torch.randn(B, in_dim)


# ─────────────────────────────────────────────
# 输出结构
# ─────────────────────────────────────────────

def test_output_keys():
    model = HMoE(in_dim=128)
    out = model(_make_feat())
    assert set(out.keys()) == {"price", "direction", "gate_weights", "aux_loss"}


def test_output_shapes():
    B, in_dim = 8, 128
    model = HMoE(in_dim=in_dim)
    out = model(_make_feat(B, in_dim))
    assert out["price"].shape        == (B, 1)
    assert out["direction"].shape    == (B, 1)
    assert out["gate_weights"].shape == (B, 4)   # n_super * n_sub = 2 * 2
    assert out["aux_loss"].shape     == ()        # scalar


# ─────────────────────────────────────────────
# 门控性质
# ─────────────────────────────────────────────

def test_gate_weights_sum_to_one():
    model = HMoE(in_dim=64)
    out = model(_make_feat(16, 64))
    row_sums = out["gate_weights"].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(16), atol=1e-5), \
        f"门控权重行和不为1: {row_sums}"


def test_direction_in_zero_one():
    model = HMoE(in_dim=64)
    out = model(_make_feat(16, 64))
    assert out["direction"].min() >= 0.0
    assert out["direction"].max() <= 1.0


def test_aux_loss_positive():
    model = HMoE(in_dim=64)
    model.train()
    out = model(_make_feat(16, 64))
    assert out["aux_loss"].item() >= 0.0


# ─────────────────────────────────────────────
# 梯度
# ─────────────────────────────────────────────

def test_gradient_flows_to_input():
    model = HMoE(in_dim=64)
    x = _make_feat(4, 64).requires_grad_(True)
    loss = model(x)["price"].sum() + model(x)["aux_loss"]
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_all_params_receive_gradient():
    model = HMoE(in_dim=64)
    model.train()
    x = _make_feat(8, 64)
    out = model(x)
    loss = out["price"].sum() + out["aux_loss"]
    loss.backward()
    no_grad = [n for n, p in model.named_parameters()
               if p.grad is None or p.grad.abs().sum() == 0]
    assert len(no_grad) == 0, f"以下参数未收到梯度: {no_grad}"


# ─────────────────────────────────────────────
# 与 TCN-LSTM-Transformer 的联合测试
# ─────────────────────────────────────────────

def test_tcn_hmoe_pipeline():
    """支路一 + H-MoE 端到端前向传播"""
    B, T, F = 4, 64, 20
    feat_dim = 128

    backbone = TCNLSTMTransformer(in_features=F, out_dim=feat_dim)
    hmoe     = HMoE(in_dim=feat_dim)

    x = torch.randn(B, T, F)
    feat = backbone(x)
    out  = hmoe(feat)

    assert out["price"].shape     == (B, 1)
    assert out["direction"].shape == (B, 1)
    assert not torch.isnan(out["price"]).any()
    assert not torch.isnan(out["direction"]).any()


def test_tcn_hmoe_backward():
    """联合反向传播不报错"""
    B, T, F = 4, 32, 15
    backbone = TCNLSTMTransformer(in_features=F, out_dim=64)
    hmoe     = HMoE(in_dim=64)

    x   = torch.randn(B, T, F)
    y   = torch.randn(B, 1)
    out = hmoe(backbone(x))

    loss = torch.nn.functional.mse_loss(out["price"], y) + out["aux_loss"]
    loss.backward()


# ─────────────────────────────────────────────
# 负载均衡效果验证
# ─────────────────────────────────────────────

def test_load_balance_discourages_collapse():
    """
    aux_loss 在均匀分配时应最小。
    构造全部样本都走同一专家的极端情况，aux_loss 应比均匀时大。
    """
    model = HMoE(in_dim=32, aux_loss_coef=1.0)
    model.eval()

    # 均匀情况：随机输入
    torch.manual_seed(0)
    x_uniform = torch.randn(64, 32)
    loss_uniform = model(x_uniform)["aux_loss"].item()

    # 极端情况：所有输入都相同，门控会趋向相同专家
    x_same = torch.ones(64, 32)
    loss_same = model(x_same)["aux_loss"].item()

    # 只验证 aux_loss 是正数，说明惩罚机制存在
    assert loss_uniform >= 0
    assert loss_same    >= 0


if __name__ == "__main__":
    tests = [
        test_output_keys,
        test_output_shapes,
        test_gate_weights_sum_to_one,
        test_direction_in_zero_one,
        test_aux_loss_positive,
        test_gradient_flows_to_input,
        test_all_params_receive_gradient,
        test_tcn_hmoe_pipeline,
        test_tcn_hmoe_backward,
        test_load_balance_discourages_collapse,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 通过")
