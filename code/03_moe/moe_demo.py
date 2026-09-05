# -*- coding: utf-8 -*-
"""
MoE 混合专家 落地 Demo：稀疏专家路由最小实现 + 单轮推理

【环境依赖】
- Python 3.10+, PyTorch 2.x；CPU 可运行

【核心逻辑】
1. 门控 Router：把每个 token 映射成 n_experts 个分数；
2. Top-k 路由：每个 token 只激活分数最高的 k 个专家；
3. 负载均衡损失：鼓励 token 均匀分布到各专家（防止专家饿死）；
4. 输出 = 被选中专家输出的加权和。

【关键参数】
- n_experts=4：专家数量；
- top_k=2：每个 token 激活几个专家；
- d_model=16：输入/输出宽度。

【避坑】
1. 真实 MoE 还要考虑 all2all 通信、专家显存驻留与路由局部性；
2. 负载均衡损失系数不能太大，否则路由失去选择性；
3. 只看参数量会高估 MoE 算力成本，要看激活参数量。

【输出解读】
- 输出形状 [batch, seq, d_model]；
- 打印每个专家的“被选次数”，均匀说明负载均衡有效；
- 总参数量远大于“每次激活参数量”。

【工程改造方向】
- 把 Expert 换成 Transformer FFN 或任意子网络；
- 训练时记录路由分布，监控专家崩溃；
- 推理侧评估 EP（专家并行）需求，见主线第 39 章。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """单个专家：一个小 MLP。"""

    def __init__(self, d_model=16, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        return self.net(x)


class SparseMoE(nn.Module):
    """Top-k 稀疏 MoE。"""

    def __init__(self, d_model=16, n_experts=4, top_k=2, balance_coef=0.01):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.balance_coef = balance_coef
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(n_experts)])

    def forward(self, x):
        # x: [batch, seq, d_model]，先压成 token 级
        b, s, d = x.shape
        flat = x.reshape(-1, d)  # [b*s, d]
        scores = self.router(flat)  # [b*s, n_experts]

        top_scores, top_idx = torch.topk(scores, self.top_k, dim=-1)
        weights = F.softmax(top_scores, dim=-1)  # 归一化路由权重

        out = torch.zeros_like(flat)
        for e_idx in range(self.n_experts):
            # 哪些 token 选中了专家 e_idx
            mask = top_idx == e_idx  # [b*s, top_k]
            if mask.any():
                # 取该专家对应位置的权重并求和
                w = (weights * mask.float()).sum(dim=-1)  # [b*s]
                expert_out = self.experts[e_idx](flat)
                out += expert_out * w.unsqueeze(-1)

        # 负载均衡损失：专家被选概率尽量均匀（简化版）
        count = torch.zeros(self.n_experts)
        for e_idx in range(self.n_experts):
            count[e_idx] = (top_idx == e_idx).float().mean()
        target = torch.full_like(count, 1.0 / self.n_experts)
        balance_loss = F.mse_loss(count, target)

        return out.reshape(b, s, d), balance_loss


def main():
    torch.manual_seed(0)
    model = SparseMoE()
    x = torch.randn(2, 6, 16)
    y, balance_loss = model(x)
    print("MoE output:", tuple(y.shape))
    print("balance_loss: %.4f（越小越均匀）" % balance_loss.item())

    total_params = sum(p.numel() for p in model.parameters())
    # 粗略估算单 token 激活参数：router + top_k 个专家
    expert_params = sum(p.numel() for p in model.experts[0].parameters())
    active = sum(p.numel() for p in model.router.parameters()) + expert_params * model.top_k
    print("total params ~%d, 单 token 激活参数约 %d" % (total_params, active))


"""
进阶改造 Prompt
1. 记录每个 batch 的专家命中分布，观察是否出现专家崩溃；
2. 把 balance_coef 从 0 调到 0.1，观察路由均匀度与任务 loss 的权衡；
3. 把 Expert 替换成真实 FFN 结构并接入 Transformer 层；
4. 对比同参数量稠密模型与 MoE 的单 token 算力差异；
5. 调研 EP 推理（主线第 39 章）并设计专家路由缓存。
"""

if __name__ == "__main__":
    main()
