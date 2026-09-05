# -*- coding: utf-8 -*-
"""
Transformer 落地代码 1/2：手写注意力 + 多头注意力 + 位置编码

【环境依赖】
- Python 3.10+, PyTorch 2.x
- CPU 即可运行

【本文件做什么】
- 从零实现 Scaled Dot-Product Attention：softmax(QK^T/sqrt(d))V
- 从零实现多头注意力（Linear 投影 -> 切头 -> 注意力 -> 拼接 -> 输出投影）
- 实现正弦位置编码
- 用随机数据前向一遍，验证维度与数值范围

【关键参数】
- d_model=32：模型宽度；
- n_heads=4：头数，需能被 d_model 整除；
- seq_len=10：序列长度。

【避坑】
1. 注意力分数要除以 sqrt(d_k)，否则 softmax 会饱和、梯度消失；
2. 多头切分后 head_dim = d_model // n_heads，不能有余数；
3. 训练/推理都要做 mask（padding/因果），本 Demo 演示基础版。

【输出解读】
- 输出形状为 [batch, seq_len, d_model]；
- 数值范围比输入更“集中”，说明注意力在做加权平均。

【工程改造方向】
- 加 causal mask 即可变成 decoder-only 自回归注意力；
- 把 Linear 换成 8bit/4bit 版本即可做量化注意力（见主线 31 章）；
- 换 RoPE 位置编码可提升外推能力（见主线 33 章）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """手写缩放点积注意力。
    q/k/v: [batch, heads, seq, head_dim]
    """
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights


class MultiHeadAttentionManual(nn.Module):
    """手写多头注意力。"""
    def __init__(self, d_model=32, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # x: [b, s, d] -> [b, heads, s, head_dim]
        b, s, d = x.shape
        x = x.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        return x

    def forward(self, x, mask=None):
        q = self.split_heads(self.wq(x))
        k = self.split_heads(self.wk(x))
        v = self.split_heads(self.wv(x))
        out, _ = scaled_dot_product_attention(q, k, v, mask)
        b, _, s, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.wo(out)


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码：偶数为 sin，奇数为 cos。"""
    def __init__(self, d_model=32, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: [b, s, d]
        return x + self.pe[:, :x.shape[1]]


def main():
    torch.manual_seed(0)
    batch, seq, d_model, heads = 2, 10, 32, 4
    x = torch.randn(batch, seq, d_model)

    attn = MultiHeadAttentionManual(d_model, heads)
    out = attn(x)
    print("MHA output:", tuple(out.shape))

    pe = SinusoidalPositionalEncoding(d_model)
    out2 = pe(x)
    print("After PE:", tuple(out2.shape), "std=%.3f" % out2.std().item())

    # 因果 mask 演示：每个位置只能看自己及之前
    causal = torch.tril(torch.ones(seq, seq)).unsqueeze(0).unsqueeze(0)
    out3 = attn(x, mask=causal)
    print("Causal MHA:", tuple(out3.shape))


"""进阶改造 Prompt
1. 加上 padding mask + causal mask，做真正的 decoder；
2. 把正弦位置编码替换成 RoPE，比较外推长度；
3. 输出加权注意力权重，可视化“模型在看谁”；
4. 用 FlashAttention（见 04 目录）替换本实现并对比速度/显存；
5. 对 d_model/heads 做消融，观察表达能力与显存。
"""

if __name__ == "__main__":
    main()
