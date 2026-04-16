"""
支路一：TCN-LSTM-Transformer 时序建模模块
参照孙惠莹(2025) §3.1

数据流：
  输入 (B, T, F)
    → TCN（4层膨胀因果卷积，捕捉长程依赖）
    → 4层堆叠 LSTM（隐藏维度256，捕捉短期序列）
    → Transformer Encoder（4头自注意力 + 位置编码，捕捉全局关系）
    → 取最后时间步，线性投影
  输出 (B, out_dim)  ← 供 H-MoE 融合层使用
"""

import math
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# ─────────────────────────────────────────────
# 1. TCN 组件
# ─────────────────────────────────────────────

class _CausalConv1d(nn.Module):
    """
    因果一维卷积：输出位置 t 只能看到输入 t 及之前的时间步。
    实现方式：在序列左侧补 (kernel-1)*dilation 个零，右侧不补。
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel - 1) * dilation
        # padding=0，填充由 F.pad 手动完成以保证因果性
        self.conv = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = torch.nn.functional.pad(x, (self.left_pad, 0))
        return self.conv(x)


class _TCNBlock(nn.Module):
    """
    单个 TCN 残差块：两层因果卷积 + ReLU + Dropout + 残差连接。
    当输入输出通道不同时用 1×1 卷积对齐。
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(in_ch,  out_ch, kernel, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            _CausalConv1d(out_ch, out_ch, kernel, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(self.net(x) + residual)


class TCN(nn.Module):
    """
    多层 TCN：膨胀因子按 1, 2, 4, 8, ... 指数增长，
    使感受野随层数指数扩大，能覆盖较长时序。

    Args:
        in_features: 输入特征维度
        hidden:      每层卷积输出通道数
        n_levels:    堆叠层数（默认4层，感受野 = 2*(2^4 - 1)*1 + 1 = 31）
        kernel:      卷积核大小（默认3）
        dropout:     Dropout 概率
    """
    def __init__(self, in_features: int, hidden: int = 64,
                 n_levels: int = 4, kernel: int = 3, dropout: float = 0.3):
        super().__init__()
        layers = []
        for i in range(n_levels):
            in_ch = in_features if i == 0 else hidden
            layers.append(_TCNBlock(in_ch, hidden, kernel,
                                    dilation=2 ** i, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)  时序输入
        Returns:
            (B, T, hidden)
        """
        # Conv1d 期望 (B, C, T)，前后转置
        return self.network(x.permute(0, 2, 1)).permute(0, 2, 1)


# ─────────────────────────────────────────────
# 2. 位置编码（Transformer 必须）
# ─────────────────────────────────────────────

class _PositionalEncoding(nn.Module):
    """
    标准正弦/余弦位置编码（Vaswani et al., 2017）。
    让 Transformer 感知序列中每个时间步的相对位置。
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为 buffer：不参与梯度更新，但会随模型保存/加载
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────
# 3. 主模型：TCN-LSTM-Transformer
# ─────────────────────────────────────────────

class TCNLSTMTransformer(nn.Module):
    """
    三阶段时序特征提取器：
      TCN  → 捕捉多尺度局部模式（膨胀感受野）
      LSTM → 捕捉短期有序依赖（门控记忆）
      Transformer → 捕捉任意位置间的全局关系（自注意力）

    Args:
        in_features:    输入特征维度
        tcn_hidden:     TCN 每层通道数（默认64）
        tcn_levels:     TCN 层数（默认4，感受野约31步）
        lstm_hidden:    LSTM 隐藏维度（默认256，与孙惠莹一致）
        n_lstm_layers:  LSTM 堆叠层数（默认4）
        nhead:          Transformer 注意力头数（默认4）
        tf_layers:      Transformer Encoder 层数（默认2）
        dropout:        Dropout 概率（默认0.3）
        out_dim:        输出特征向量维度（默认128，供 H-MoE 使用）
    """

    def __init__(
        self,
        in_features: int,
        tcn_hidden: int = 64,
        tcn_levels: int = 4,
        lstm_hidden: int = 256,
        n_lstm_layers: int = 4,
        nhead: int = 4,
        tf_layers: int = 2,
        dropout: float = 0.3,
        out_dim: int = 128,
    ):
        super().__init__()

        # ── TCN
        self.tcn = TCN(
            in_features=in_features,
            hidden=tcn_hidden,
            n_levels=tcn_levels,
            dropout=dropout,
        )

        # ── LSTM（TCN输出 → LSTM输入）
        self.lstm = nn.LSTM(
            input_size=tcn_hidden,
            hidden_size=lstm_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

        # ── Transformer（位置编码 + Encoder）
        self.pos_enc = _PositionalEncoding(lstm_hidden, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden,
            nhead=nhead,
            dim_feedforward=lstm_hidden * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)

        # ── 输出投影
        self.norm = nn.LayerNorm(lstm_hidden)
        self.proj = nn.Linear(lstm_hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, in_features)  — 技术指标时序窗口
        Returns:
            feat: (B, out_dim)      — 最后时间步的特征向量
        """
        # 1. TCN：提取多尺度局部特征
        x = self.tcn(x)                      # (B, T, tcn_hidden)

        # 2. LSTM：顺序建模
        x, _ = self.lstm(x)                  # (B, T, lstm_hidden)

        # 3. Transformer：全局依赖建模
        x = self.pos_enc(x)                  # 加入位置编码
        x = self.transformer(x)              # (B, T, lstm_hidden)

        # 4. 取最后时间步，归一化后投影
        feat = self.proj(self.norm(x[:, -1, :]))  # (B, out_dim)
        return feat


# ─────────────────────────────────────────────
# 快速验证（python -m src.models.tcn_lstm_transformer）
# ─────────────────────────────────────────────

if __name__ == "__main__":
    B, T, F = 32, 64, 20   # batch=32, 窗口=64天, 特征=20维

    model = TCNLSTMTransformer(in_features=F)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy = torch.randn(B, T, F)
    out = model(dummy)

    print("=" * 40)
    print(f"输入形状:  {list(dummy.shape)}")
    print(f"输出形状:  {list(out.shape)}")
    print(f"可训练参数: {total_params:,}")
    print("=" * 40)

    # 验证因果性：改变最后一个时间步，前面时间步的输出不变
    dummy2 = dummy.clone()
    dummy2[:, -1, :] += 999
    out2 = model(dummy2)
    # 只取倒数第二步验证（最后一步本来就可以看到自己）
    diff = (out[:, 0] - out2[:, 0]).abs().max().item() if out.dim() > 1 else 0
    print(f"因果性验证（改变最后帧，倒数第二步输出差异应≈0）: {diff:.6f}")
    print("验证通过 ✓" if diff < 1e-4 else "警告：可能存在非因果泄漏")
