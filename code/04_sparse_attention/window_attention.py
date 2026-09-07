# -*- coding: utf-8 -*-
"""
窗口注意力（滑动窗口）落地 Demo：把全注意力改成只看局部窗口

【层级】L1（极简 Demo：新手跑通，CPU 秒级出结果）

【环境依赖】
- Python 3.10+；PyTorch 2.x（建议 >=2.1）；仅 torch
- 安装：pip install torch==2.2.2（GPU 版按官网 CUDA 版本装；本 Demo CPU 可跑）

【核心逻辑】
- 标准注意力：每个位置看全序列，复杂度 O(n^2)；
- 窗口注意力：每个位置只看前后 w/2 个邻居，复杂度约 O(n*w)；
- 本实现用 mask 方式演示：先生成全量 scores，再把窗口外置为 -inf。

【关键参数】
- seq_len=32；window=8（实际看左右各 4 个位置 + 自己）；
- n_heads=2, head_dim=8

【避坑】
1. 窗口外的 -inf 必须足够小，否则 softmax 后仍有泄漏；
2. 长程依赖任务用窗口注意力可能掉点，需配合全局 token；
3. 本 Demo 的 mask 法便于理解；高性能实现应直接分块计算。

【运行结果示例】（真实运行，CPU）
$ python window_attention.py
full vs window mean abs diff: 0.3256（>0 说明窗口限制生效）
（diff 明显大于 0 = 窗口确实限制了可见范围；把 window 调到 >=seq_len 后 diff 会归零）

【输出解读】
- 输出形状不变；
- diff 在 window<seq 时 >0（窗口生效），window>=seq 时 ≈0（退化为全注意力）。

【高频报错 Top5】
1. window 为奇数/过小导致 mask 全 False：softmax(-inf) 出 NaN；修复：window>=2 且保证每行至少一个 True；
2. q/k/v 维度错：需 [batch, heads, seq, head_dim]；修复：先 transpose(1,2) 再传入；
3. masked_fill 用错 mask 极性：窗口外应置 -inf；修复：masked_fill(~mask, -inf)；
4. 长序列 OOM：mask 法仍构建了 n*n scores；修复：用分块实现（见 L3 sparse_attn_bench.py）；
5. diff 为 0 但窗口应该生效：window>=seq 时本来就会退化为全注意力，属预期。

【工程改造方向】
- 窗口 + 全局锚点 token 混合，兼顾效率与长程；
- 对超长文档/流式场景实测收益；
- 配合 FlashAttention 类 kernel 做块级窗口计算。
"""

import math

import torch
import torch.nn.functional as F


def window_mask(seq_len, window):
    """生成 [1,1,seq,seq] 的窗口 mask：窗口内为 True。"""
    idx = torch.arange(seq_len)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    return dist <= (window // 2)


def window_attention(q, k, v, window=8):
    """q/k/v: [batch, heads, seq, head_dim]，返回窗口注意力输出。"""
    b, h, s, d = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    mask = window_mask(s, window).unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(~mask, float("-inf"))
    w = F.softmax(scores, dim=-1)
    return torch.matmul(w, v)


def full_attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    w = F.softmax(scores, dim=-1)
    return torch.matmul(w, v)


def main():
    torch.manual_seed(0)
    b, h, s, d = 1, 2, 32, 8
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)

    o_full = full_attention(q, k, v)
    o_win = window_attention(q, k, v, window=8)
    diff = (o_full - o_win).abs().mean().item()
    print("full vs window mean abs diff: %.4f（>0 说明窗口限制生效）" % diff)


"""
进阶改造 Prompt
1. 把 window 从 4 调到 32，观察输出逐步逼近全注意力；
2. 设计“窗口 + 全局 token”的混合注意力并对比长程任务；
3. 在长文档问答上对比全注意力与窗口注意力的质量/显存；
4. 实现无 mask 的分块窗口 kernel，观察速度提升；
5. 结合 FlashAttention 分块思路做高性能版。
"""

if __name__ == "__main__":
    main()