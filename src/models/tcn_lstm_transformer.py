"""
支路一：TCN-LSTM-Transformer 时序建模模块
参照孙惠莹(2025) §3.1：
  - TCN：膨胀因子指数增长，卷积核尺寸=3，捕捉长程时序依赖
  - 4 层堆叠 LSTM（隐藏维度 256），捕捉短期序列模式
  - Transformer Encoder（4 头自注意力），捕捉全局位置关系

输出：特征向量（供 H-MoE 融合层使用），不直接预测价格。
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# ---------------------------------------------------------------------------
# TCN 组件
# ---------------------------------------------------------------------------

class _CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=self.padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)[:, :, : -self.padding] if self.padding else self.conv(x)


class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(in_ch, out_ch, kernel, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            _CausalConv1d(out_ch, out_ch, kernel, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, n_levels: int = 4,
                 kernel: int = 3, dropout: float = 0.3):
        super().__init__()
        layers = []
        for i in range(n_levels):
            in_ch  = in_features if i == 0 else hidden
            layers.append(_TCNBlock(in_ch, hidden, kernel, dilation=2 ** i, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T) for Conv1d
        return self.network(x.permute(0, 2, 1)).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------

class TCNLSTMTransformer(nn.Module):
    """
    Args:
        in_features:  输入特征维度（技术指标数 + 情感得分数）
        hidden:       LSTM 隐藏层维度，默认 256
        n_lstm_layers:LSTM 层数，默认 4
        nhead:        Transformer 注意力头数，默认 4
        dropout:      Dropout 概率，默认 0.3
        out_dim:      输出特征向量维度（用于 H-MoE 融合）
    """

    def __init__(
        self,
        in_features: int,
        hidden: int = 256,
        n_lstm_layers: int = 4,
        nhead: int = 4,
        dropout: float = 0.3,
        out_dim: int = 128,
    ):
        super().__init__()
        self.tcn = TCN(in_features, hidden=hidden // 4, dropout=dropout)

        self.lstm = nn.LSTM(
            input_size=hidden // 4,
            hidden_size=hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead,
            dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, in_features)
        Returns:
            feat: (B, out_dim) — 最后一个时间步的特征向量
        """
        x = self.tcn(x)                        # (B, T, hidden//4)
        x, _ = self.lstm(x)                    # (B, T, hidden)
        x = self.transformer(x)                # (B, T, hidden)
        feat = self.proj(x[:, -1, :])          # (B, out_dim)
        return feat
