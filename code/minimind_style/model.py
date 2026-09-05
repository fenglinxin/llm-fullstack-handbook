# -*- coding: utf-8 -*-
"""
TinyGPT：MiniMind 式纯 PyTorch 微型 GPT（Decoder-only）

【定位】参考 MiniMind“从 0 教学”思路，用最少代码实现：
RMSNorm + RoPE + 因果多头注意力 + FFN + 残差，不依赖 transformers。

【环境依赖】Python 3.10+, PyTorch 2.x；CPU 可运行。

【关键参数】见 TinyConfig：dim/n_layers/n_heads/max_seq。

【避坑】
1. 因果 mask 必须用三角矩阵，否则训练时看到未来；
2. RoPE 只在 q/k 上做，v 不做；
3. 小模型建议 lr=1e-3 级别，dim 别小于 32。

【输出解读】forward 返回 [batch, seq, vocab] logits。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyConfig:
    def __init__(
        self,
        vocab_size=1000,
        dim=64,
        n_layers=2,
        n_heads=2,
        max_seq=64,
        dropout=0.0,
        rope_base=10000.0,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq = max_seq
        self.dropout = dropout
        self.rope_base = rope_base


class RMSNorm(nn.Module):
    """RMSNorm：比 LayerNorm 少算均值，速度更快，Qwen/Llama 常用。"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_rope(dim, max_seq, base=10000.0):
    """预计算 RoPE 的 cos/sin：dim 需为偶数。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)  # [max_seq, dim/2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """把旋转位置编码作用到 q/k：x 形状 [b, h, s, d]。"""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


class CausalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        b, s, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # [b, s, heads, head_dim] -> [b, heads, s, head_dim]
        q = q.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos[:s], sin[:s])
        k = apply_rope(k, cos[:s], sin[:s])

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(s, s, device=x.device)).view(1, 1, s, s)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)
        o = torch.matmul(w, v)
        o = o.transpose(1, 2).contiguous().view(b, s, d)
        return self.out(o)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.up = nn.Linear(cfg.dim, 4 * cfg.dim, bias=False)
        self.down = nn.Linear(4 * cfg.dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = CausalAttention(cfg)
        self.norm2 = RMSNorm(cfg.dim)
        self.ffn = FeedForward(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        cos, sin = precompute_rope(cfg.dim // cfg.n_heads, cfg.max_seq, cfg.rope_base)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, idx):
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, idx, max_new=40, temperature=0.8, top_k=20):
        """自回归生成：idx 形状 [1, s]。"""
        self.eval()
        for _ in range(max_new):
            x = idx[:, -self.cfg.max_seq :]
            logits = self(x)[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


"""
进阶改造 Prompt
1. 把正弦/绝对位置换成你喜欢的 RoPE 变体，对比外推；
2. 加 KV Cache 让生成提速（主线 33 章）；
3. 把 FFN 换成 SwiGLU / MoE（主线 00 章 MoE Demo 可拼装）；
4. 把注意力替换成 FlashAttention / 窗口注意力（04 目录）；
5. 调 dim/layers/heads 并记录 loss 与生成质量的关系。
"""
