"""
H-MoE（层次化混合专家）融合模块
参照孙惠莹(2025) §3.3：
  - 4 个专家网络，每个专家对应一种市场状态
    （上涨趋势 / 下跌趋势 / 震荡整理 / 高波动）
  - 门控网络：根据输入特征动态分配各专家权重
  - 层次化：先在模态内专家路由，再跨模态汇总

同时提供双输出头：
  - regression_head：预测次日收盘价（MSE 损失）
  - classification_head：预测涨跌方向（BCE 损失）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HMoE(nn.Module):
    """
    Args:
        in_dim:     融合后的输入特征维度（各支路 out_dim 之和）
        n_experts:  专家数量，默认 4
        expert_hidden: 专家内隐藏层宽度
        out_dim:    专家输出维度
        dropout:    Dropout 概率
    """

    def __init__(
        self,
        in_dim: int,
        n_experts: int = 4,
        expert_hidden: int = 128,
        out_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(in_dim, expert_hidden, out_dim, dropout)
            for _ in range(n_experts)
        ])
        # 门控网络：输出 n_experts 维 softmax 权重
        self.gate = nn.Sequential(
            nn.Linear(in_dim, n_experts * 4),
            nn.ReLU(),
            nn.Linear(n_experts * 4, n_experts),
        )

        # 双输出头
        self.regression_head     = nn.Linear(out_dim, 1)
        self.classification_head = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, in_dim) 各支路特征拼接后的向量

        Returns:
            {
              'price':     (B, 1)  回归输出（归一化价格）
              'direction': (B, 1)  分类输出（sigmoid 概率，> 0.5 为涨）
              'gate_weights': (B, n_experts)  专家权重（用于可解释性分析）
            }
        """
        gate_logits = self.gate(x)                          # (B, n_experts)
        weights = F.softmax(gate_logits, dim=-1)            # (B, n_experts)

        expert_outs = torch.stack(
            [e(x) for e in self.experts], dim=1
        )                                                   # (B, n_experts, out_dim)

        # 加权汇总
        fused = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)  # (B, out_dim)

        return {
            "price":        self.regression_head(fused),
            "direction":    torch.sigmoid(self.classification_head(fused)),
            "gate_weights": weights,
        }
