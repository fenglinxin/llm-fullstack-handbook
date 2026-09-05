# -*- coding: utf-8 -*-
"""
MoE 变体（MiniMind 式：把 FFN 换成“多个专家 + 路由器”）

【定位】展示主干第 00 章 MoE 的核心：每个 token 经 Router 打分，
只激活 Top-k 个专家，稀疏计算换取更大参数量。

【环境依赖】Python 3.10+, PyTorch 2.x（无第三方库）

【关键参数】
- n_experts：专家数量（本实现 4，真实系统可达 8-256）；
- top_k：每个 token 激活几个专家（本实现 2）；
- aux_loss：负载均衡辅助损失，乘以小系数加入总 loss，
  防止所有 token 都涌向同一个专家（“路由坍塌”）。

【避坑】
1. Router 是稠密计算，MoE 的省算力收益来自“专家前向只算 top-k”，
   参数量会变大，小模型 CPU 训练反而更慢；
2. top-k 权重要在选中的专家上重新归一化，否则概率和不为 1；
3. 辅助 loss 系数要小（0.001-0.01），否则会干扰主任务。

【输出解读】router probs 分布：若长期集中在一个专家，
说明路由未学会分化，需要调大 aux_loss 权重。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TinyGPT  # 复用 TinyGPT，只替换其中的 FFN


class ExpertFFN(nn.Module):
    """单个专家 = 一个 SiLU FFN（等价于原 FeedForward）。"""

    def __init__(self, dim):
        super().__init__()
        self.up = nn.Linear(dim, 4 * dim, bias=False)
        self.down = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.up(x)))


class MoEFFN(nn.Module):
    def __init__(self, dim, n_experts=4, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(dim) for _ in range(n_experts)])

    def forward(self, x):
        # 注意：为兼容 TinyGPT 的 Block（ffn(x) 需返回张量），这里只返回主输出，
        # aux_loss 挂在 self.aux_loss 上供训练循环读取（保持计算图可反传）
        # x: [b, s, dim]
        b, s, d = x.shape
        flat = x.reshape(-1, d)                    # [b*s, dim]
        logits = self.router(flat)                 # [b*s, n_experts]
        probs = F.softmax(logits, dim=-1)

        top_probs, top_idx = torch.topk(probs, self.top_k, dim=-1)
        # 重新归一化：只让选中的专家分摊 1.0 的概率
        top_probs = top_probs / top_probs.sum(-1, keepdim=True)

        out = torch.zeros_like(flat)
        hit_counts = torch.zeros(self.n_experts, device=x.device)
        for e in range(self.n_experts):
            mask = top_idx == e                   # [b*s, top_k] 命中位
            hits = mask.any(dim=-1)               # 该 token 是否选中专家 e
            hit_counts[e] = hits.sum()
            if hits.any():
                w = top_probs[mask].sum(-1)       # 该 token 给专家 e 的权重
                expert_out = self.experts[e](flat[hits])
                out[hits] += w.unsqueeze(-1) * expert_out

        # 负载均衡辅助 loss：f_i=实际被选中占比，p_i=平均路由概率，
        # 均匀路由时 f_i 与 p_i 都接近 1/n，aux 最小（防“路由坍塌”）
        f_i = hit_counts / flat.shape[0]
        p_i = probs.mean(dim=0)
        self.aux_loss = self.n_experts * (f_i * p_i).sum()

        return out.view(b, s, d)


def build_moe_model(cfg, n_experts=4, top_k=2):
    """基于 TinyGPT 构建 MoE 版：只替换每个 Block 的 ffn。"""
    model = TinyGPT(cfg)
    for block in model.blocks:
        block.ffn = MoEFFN(cfg.dim, n_experts=n_experts, top_k=top_k)
    return model


"""
进阶改造 Prompt
1. 加 Router noise（训练期扰动）与大模型常用的 load balancing 版本；
2. top-k 换成“阈值路由”，比较吞吐与效果；
3. 记录每个专家被选中的比例曲线，观察路由是否分化；
4. 对比同参数预算下 Dense vs MoE 的 loss 与生成质量；
5. 思考：为什么推理部署要 EP（专家并行）而不是 TP（主线 00/33 章）。
"""
