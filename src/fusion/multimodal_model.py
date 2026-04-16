"""
四模态融合总模型
整合四条支路 + H-MoE 融合层，
复现报名表 §5.3 描述的系统架构。

数据流：
  支路1 (TCNLSTMTransformer) ─────────────────────────────────┐
  支路2 (CSACNN) ──── INSA(图文早期融合) ─────────────────────┤
  支路3 (BertGFM) ─────────────────────────────────────────── ┤→ concat → H-MoE → price / direction
  支路4 (SnowNLP情感) ─────────────────────────────────────────┘
"""

import torch
import torch.nn as nn

from src.models.tcn_lstm_transformer import TCNLSTMTransformer
from src.models.cs_acnn import CSACNN
from src.models.bert_gfm import BertGFM
from src.models.hmoe import HMoE


class INSA(nn.Module):
    """图文跨模态早期融合（图像多尺度特征 × 文本语义向量）。"""

    def __init__(self, img_ch: int, text_dim: int, out_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj_img  = nn.Linear(img_ch, out_dim)
        self.proj_text = nn.Linear(text_dim, out_dim)
        self.attn      = nn.MultiheadAttention(out_dim, num_heads=4, batch_first=True)

    def forward(
        self, img_feat: torch.Tensor, text_feat: torch.Tensor
    ) -> torch.Tensor:
        img_vec  = self.proj_img(self.pool(img_feat).flatten(1)).unsqueeze(1)
        text_vec = self.proj_text(text_feat).unsqueeze(1)
        out, _ = self.attn(img_vec, text_vec, text_vec)
        return out.squeeze(1)


class MultiModalStockModel(nn.Module):
    """
    Args:
        ts_in_features:  支路1输入特征数（技术指标 + 情感得分列数）
        bert_name:       支路3 BERT 模型名
        feat_dim:        各支路统一输出维度，默认 128
    """

    def __init__(
        self,
        ts_in_features: int = 21,
        bert_name: str = "bert-base-chinese",
        feat_dim: int = 128,
    ):
        super().__init__()

        # 支路1：时序建模
        self.branch_ts = TCNLSTMTransformer(
            in_features=ts_in_features, out_dim=feat_dim
        )

        # 支路2：图像特征
        self.branch_img = CSACNN(in_channels=3, out_dim=feat_dim)

        # 支路3：文本特征
        self.branch_text = BertGFM(bert_name=bert_name, out_dim=feat_dim)

        # INSA：图文早期融合（使用 layer3 特征，128通道）
        self.insa = INSA(img_ch=128, text_dim=feat_dim, out_dim=feat_dim)

        # 情感得分映射（标量 → 向量）
        self.sentiment_proj = nn.Sequential(
            nn.Linear(1, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, feat_dim),
        )

        # H-MoE 融合层（4路特征拼接后输入）
        self.hmoe = HMoE(in_dim=feat_dim * 4, out_dim=feat_dim)

    def forward(
        self,
        ts_x: torch.Tensor,
        img_x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentiment_score: torch.Tensor,
        gfm_h: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            ts_x:           (B, T, ts_in_features)  技术指标时序
            img_x:          (B, 3, H, W)             TIFS蜡烛图
            input_ids:      (B, seq_len)             新闻token ids
            attention_mask: (B, seq_len)
            sentiment_score:(B, 1)                   日级SnowNLP情感均值
            gfm_h:          GFM上一步隐藏状态

        Returns:
            HMoE 输出字典: price, direction, gate_weights
        """
        # 支路1
        f_ts = self.branch_ts(ts_x)                                    # (B, feat_dim)

        # 支路2
        multi_scale, f_img_deep = self.branch_img(img_x)               # 3个尺度 + 深层

        # 支路3
        f_text, gfm_h_new = self.branch_text(input_ids, attention_mask, gfm_h)

        # INSA：使用最深层图像特征 (layer3, 128ch) 与文本做早期融合
        f_insa = self.insa(multi_scale[2], f_text)                     # (B, feat_dim)

        # 支路4：情感得分
        f_sent = self.sentiment_proj(sentiment_score)                  # (B, feat_dim)

        # 拼接四路特征 → H-MoE
        fused = torch.cat([f_ts, f_insa, f_img_deep, f_sent], dim=-1) # (B, feat_dim*4)
        out = self.hmoe(fused)
        out["gfm_h"] = gfm_h_new
        return out
