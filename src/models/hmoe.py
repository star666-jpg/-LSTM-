"""
H-MoE：层次化混合专家融合模块
参照孙惠莹(2025) §3.3

层次化结构（两级路由）：
  第一级（粗粒度）：2 个超级专家，对应宏观市场方向（多头 / 空头）
  第二级（细粒度）：每个超级专家下有 2 个子专家，对应精细状态
    超级专家0（多头）→ 子专家 [强趋势上涨 / 震荡上涨]
    超级专家1（空头）→ 子专家 [强趋势下跌 / 震荡下跌]
  共 4 个叶节点专家，与论文一致。

关键机制：
  - Top-K 稀疏路由（K=2）：每个样本只激活权重最大的 K 个专家
  - 负载均衡辅助损失：防止所有样本塌缩到同一专家
  - 双输出头：回归（收盘价）+ 分类（涨跌方向）

使用方式：
    hmoe = HMoE(in_dim=128)
    out = hmoe(feat)
    # out["price"]        (B, 1)
    # out["direction"]    (B, 1)  sigmoid 概率
    # out["gate_weights"] (B, 4)  4个叶专家权重（可解释性）
    # out["aux_loss"]     scalar  负载均衡辅助损失，加入总 loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 叶节点专家
# ─────────────────────────────────────────────

class _Expert(nn.Module):
    """单个专家网络：两层 MLP + LayerNorm"""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),                    # GELU 比 ReLU 在 Transformer 系任务上更常用
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# Top-K 稀疏门控
# ─────────────────────────────────────────────

def _topk_gate(
    logits: torch.Tensor,
    k: int,
    training: bool,
    noise_std: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Top-K 稀疏路由：保留每个样本权重最大的 K 个专家，其余置零。

    训练时加入少量高斯噪声，鼓励门控探索不同专家（避免过早收敛）。

    Returns:
        weights:   (B, n_experts) 稀疏权重（只有 K 个非零）
        aux_loss:  负载均衡辅助损失（scalar）
    """
    if training and noise_std > 0:
        logits = logits + torch.randn_like(logits) * noise_std

    # 只保留 Top-K
    topk_vals, topk_idx = logits.topk(k, dim=-1)
    sparse_logits = torch.full_like(logits, float("-inf"))
    sparse_logits.scatter_(-1, topk_idx, topk_vals)
    weights = F.softmax(sparse_logits, dim=-1)   # (B, n_experts)

    # ── 负载均衡辅助损失（Shazeer et al., Switch Transformer）
    # 鼓励每个专家被均等使用
    # f_i = 样本中选择专家 i 的比例
    # p_i = 门控分配给专家 i 的平均概率
    # loss = n_experts * sum(f_i * p_i)，均匀时最小
    n_experts = logits.size(-1)
    # f_i：one-hot 选择（top-1 用于计数）
    top1_idx = logits.argmax(dim=-1)
    f = torch.zeros(n_experts, device=logits.device)
    for i in range(n_experts):
        f[i] = (top1_idx == i).float().mean()
    # p_i：完整 softmax 的均值（不用稀疏版）
    p = F.softmax(logits, dim=-1).mean(dim=0)    # (n_experts,)
    aux_loss = n_experts * (f * p).sum()

    return weights, aux_loss


# ─────────────────────────────────────────────
# H-MoE 主模块
# ─────────────────────────────────────────────

class HMoE(nn.Module):
    """
    层次化混合专家（Hierarchical Mixture of Experts）

    Args:
        in_dim:        输入特征维度（支路一单独使用时 = feat_dim = 128）
        n_super:       第一级超级专家数（默认 2：多头/空头）
        n_sub:         每个超级专家下的子专家数（默认 2）
        expert_hidden: 叶专家内隐藏层宽度
        out_dim:       专家输出维度（接双头之前）
        top_k:         每次激活的专家数（默认 2）
        dropout:       Dropout 概率
        aux_loss_coef: 负载均衡损失系数（总 loss = task_loss + coef * aux_loss）
    """

    def __init__(
        self,
        in_dim: int,
        n_super: int = 2,
        n_sub: int = 2,
        expert_hidden: int = 128,
        out_dim: int = 64,
        top_k: int = 2,
        dropout: float = 0.3,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.n_experts    = n_super * n_sub     # 总叶专家数 = 4
        self.n_super      = n_super
        self.n_sub        = n_sub
        self.top_k        = top_k
        self.aux_loss_coef = aux_loss_coef

        # ── 第一级门控（粗粒度：2个超级专家）
        self.coarse_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, n_super),
        )

        # ── 第二级门控（细粒度：每个超级专家各自路由 2 个子专家）
        # 每个超级专家有独立的子门控，输入是 in_dim + 第一级上下文
        self.fine_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim + n_super, n_sub),
            )
            for _ in range(n_super)
        ])

        # ── 叶节点专家（4个）
        self.experts = nn.ModuleList([
            _Expert(in_dim, expert_hidden, out_dim, dropout)
            for _ in range(self.n_experts)
        ])

        # ── 双输出头
        self.regression_head     = nn.Linear(out_dim, 1)
        self.classification_head = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, in_dim)

        Returns:
            price:        (B, 1)         次日收盘价（归一化）
            direction:    (B, 1)         涨跌概率（sigmoid）
            gate_weights: (B, n_experts) 4个叶专家的最终权重（可解释性）
            aux_loss:     scalar         负载均衡损失，需加入总 loss
        """
        B = x.size(0)

        # ── 第一级：粗粒度路由
        coarse_logits = self.coarse_gate(x)                  # (B, n_super)
        coarse_w, coarse_aux = _topk_gate(
            coarse_logits, k=self.n_super,                   # 第一级全激活（权重软分配）
            training=self.training
        )

        # ── 第二级：每个超级专家的细粒度路由
        # 将第一级 softmax 权重拼入输入，作为市场状态上下文
        x_aug = torch.cat([x, coarse_w.detach()], dim=-1)   # (B, in_dim + n_super)

        fine_weights_list = []
        fine_aux_list     = []
        for s in range(self.n_super):
            fine_logits = self.fine_gates[s](x_aug)          # (B, n_sub)
            fw, fa = _topk_gate(fine_logits, k=1, training=self.training)
            fine_weights_list.append(fw)                     # (B, n_sub)
            fine_aux_list.append(fa)

        # ── 组合叶专家权重
        # 最终权重 = 第一级权重 × 第二级权重
        # 专家排列：[super0_sub0, super0_sub1, super1_sub0, super1_sub1]
        leaf_weights = torch.cat([
            coarse_w[:, s:s+1] * fine_weights_list[s]       # (B, n_sub)
            for s in range(self.n_super)
        ], dim=-1)                                           # (B, n_experts=4)

        # ── Top-K 稀疏化（只保留权重最大的 K 个叶专家）
        if self.top_k < self.n_experts:
            topk_vals, topk_idx = leaf_weights.topk(self.top_k, dim=-1)
            sparse = torch.zeros_like(leaf_weights)
            sparse.scatter_(-1, topk_idx, topk_vals)
            # 重新归一化
            leaf_weights = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)

        # ── 专家前向传播（只计算权重非零的专家）
        expert_outs = torch.stack(
            [self.experts[i](x) for i in range(self.n_experts)], dim=1
        )                                                    # (B, n_experts, out_dim)

        fused = (leaf_weights.unsqueeze(-1) * expert_outs).sum(dim=1)  # (B, out_dim)

        # ── 输出
        aux_loss = self.aux_loss_coef * (
            coarse_aux + sum(fine_aux_list) / self.n_super
        )

        return {
            "price":        self.regression_head(fused),
            "direction":    torch.sigmoid(self.classification_head(fused)),
            "gate_weights": leaf_weights,
            "aux_loss":     aux_loss,
        }


# ─────────────────────────────────────────────
# 快速验证
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    B, in_dim = 16, 128

    model = HMoE(in_dim=in_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.randn(B, in_dim)
    out = model(x)

    print("=" * 45)
    print(f"输入形状:       {list(x.shape)}")
    print(f"price 形状:     {list(out['price'].shape)}")
    print(f"direction 形状: {list(out['direction'].shape)}")
    print(f"gate_weights:   {list(out['gate_weights'].shape)}")
    print(f"aux_loss:       {out['aux_loss'].item():.6f}")
    print(f"可训练参数:     {n_params:,}")
    print("=" * 45)

    # 验证门控权重每行和为1
    w_sum = out["gate_weights"].sum(dim=-1)
    assert torch.allclose(w_sum, torch.ones(B), atol=1e-5), "门控权重未归一化"

    # 验证 direction 在 [0,1]
    assert out["direction"].min() >= 0 and out["direction"].max() <= 1

    # 验证负载均衡：打印每个专家的平均激活权重
    avg_w = out["gate_weights"].mean(dim=0)
    print("\n各专家平均激活权重（均匀时各为0.25）：")
    labels = ["强趋势上涨", "震荡上涨", "强趋势下跌", "震荡下跌"]
    for i, (label, w) in enumerate(zip(labels, avg_w.tolist())):
        bar = "█" * int(w * 40)
        print(f"  专家{i}({label}): {w:.3f}  {bar}")

    print("\n验证通过 ✓")
