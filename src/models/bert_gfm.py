"""
支路三：BERT + GFM 新闻文本特征提取模块
参照昝泓含(2024) §3.2：
  - BERT-Base-Chinese：冻结主干，仅微调最后 2 层
  - GFM（门控融合模块）：简化版 LSTM（存储门+输出门），
    将 BERT [CLS] 向量与历史文本特征做时序融合
    参数量比标准 LSTM 少约 1/3
"""

import torch
import torch.nn as nn
from transformers import BertModel


class GFM(nn.Module):
    """
    Gate Fusion Module — 存储门 + 输出门（无遗忘门）。
    用于对同一股票多天新闻的 BERT 特征做序列融合。

    Args:
        input_size:  每个时间步特征维度（= bert_hidden）
        hidden_size: GFM 隐藏状态维度
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # 存储门
        self.storage_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # 输出门
        self.output_gate   = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_update   = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_size) 当前时间步特征
            h: (B, hidden_size) 上一时间步隐藏状态，默认全零

        Returns:
            (out, h_new)
        """
        if h is None:
            h = torch.zeros(x.size(0), self.storage_gate.out_features, device=x.device)

        combined = torch.cat([x, h], dim=-1)
        s = torch.sigmoid(self.storage_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))
        c = torch.tanh(self.cell_update(combined))
        h_new = s * c + (1 - s) * h
        out   = o * torch.tanh(h_new)
        return out, h_new


class BertGFM(nn.Module):
    """
    BERT-Base-Chinese + GFM 文本特征提取器。

    Args:
        bert_name:   HuggingFace 模型名，默认 'bert-base-chinese'
        freeze_layers: 冻结前 N 层（0 = 不冻结，10 = 冻结前10层）
        gfm_hidden:  GFM 隐藏维度
        out_dim:     最终输出特征维度
    """

    def __init__(
        self,
        bert_name: str = "bert-base-chinese",
        freeze_layers: int = 10,
        gfm_hidden: int = 128,
        out_dim: int = 128,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        bert_hidden = self.bert.config.hidden_size  # 768

        # 冻结 BERT 前 N 层
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        self.gfm  = GFM(bert_hidden, gfm_hidden)
        self.proj = nn.Linear(gfm_hidden, out_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:      (B, seq_len)
            attention_mask: (B, seq_len)
            h:              GFM 上一步隐藏状态

        Returns:
            feat:  (B, out_dim) 文本特征向量（供 INSA / INTA 使用）
            h_new: 更新后的 GFM 隐藏状态
        """
        cls = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        gfm_out, h_new = self.gfm(cls, h)
        feat = self.proj(gfm_out)
        return feat, h_new
