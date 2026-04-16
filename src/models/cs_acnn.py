"""
支路二：CS-ACNN 图像特征提取模块
参照昝泓含(2024) §3.3：
  - 输入：TIFS（时序图像特征序列）蜡烛图，尺寸 (B, 3, H, W)
  - 3 层 CNN + 通道注意力（Channel & Spatial Attention）
  - 同时输出多尺度特征（供 INSA 早期融合使用）和深层特征（供 INTA 晚期融合）

注：TIFS 图像生成见 scripts/generate_tifs.py（待实现）。
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc(self.avg_pool(x).flatten(1))
        mx  = self.fc(self.max_pool(x).flatten(1))
        scale = self.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.ca(x)
        x = self.sa(x)
        return x


class CSACNN(nn.Module):
    """
    Args:
        in_channels: 输入通道数，默认 3（RGB 蜡烛图）
        out_dim:     输出特征维度
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 128):
        super().__init__()
        self.layer1 = _ConvBlock(in_channels, 32)   # 多尺度特征1
        self.layer2 = _ConvBlock(32, 64)             # 多尺度特征2
        self.layer3 = _ConvBlock(64, 128)            # 多尺度特征3（深层特征）
        self.pool   = nn.AdaptiveAvgPool2d((4, 4))
        self.proj   = nn.Linear(128 * 4 * 4, out_dim)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Returns:
            multi_scale: [feat1, feat2, feat3] 三个中间层特征，用于 INSA
            deep_feat:   (B, out_dim)，深层特征用于 INTA
        """
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        deep = self.proj(self.pool(f3).flatten(1))
        return [f1, f2, f3], deep
