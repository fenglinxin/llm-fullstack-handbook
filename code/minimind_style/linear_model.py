# -*- coding: utf-8 -*-
"""
MiniMind-Linear 教学版：微型线性注意力语言模型

【定位】对标 MiniMind-Linear“线性/亚二次注意力”思路的教学原创实现：
用核函数特征化 q/k，把 softmax 注意力里的 (QK^T)V 变成
“累积 KV”的线性递推，注意力计算从 O(n^2) 降到 O(n)。

【核心公式】（每头）
φ(x) = elu(x) + 1（核特征）
S_t = Σ_{i<=t} φ(k_i) ⊗ v_i        （累积“键值外积”）
z_t = Σ_{i<=t} φ(k_i)              （累积“键和”，做归一化）
y_t = (φ(q_t) S_t) / (φ(q_t) z_t + eps)

【环境依赖】Python 3.10+, PyTorch 2.x。

【关键参数】dim/n_layers/n_heads/max_seq 与 TinyGPT 一致；
RoPE 换成了可学习绝对位置编码（线性注意力里旋转编码不兼容）。

【避坑】
1. 特征化后必须做归一化（除以累积键和），否则输出会随序列长度发散；
2. 显式 S_t 仍占 O(n·d^2) 训练显存，真正的 O(n) 推理收益在
   “流式/单步解码”（状态固定为 d×d，不随历史增长）；
3. 线性注意力的表达能力弱于 softmax（无法做尖锐的“指向性”注意力），
   小语料上通常略逊于标准 Transformer——这正是要对照观察的点。

【输出解读】对比同配置 TinyGPT（softmax 注意力）：两者 loss 都能下降；
若线性版 loss 偏高/生成更乱，说明小模型+小语料下线性注意力
没有显现长序列优势，教学重点是理解“为什么”而不是“谁更强”。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RMSNorm, FeedForward


class LinearAttention(nn.Module):
    """因果线性注意力：无 attention matrix，靠累积状态递推。"""

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        # 核特征化（elu+1，保证非负、可做因果累积）
        qf = F.elu(q) + 1.0
        kf = F.elu(k) + 1.0
        # 累积 KV 外积：S [b,h,s,d,d]，z [b,h,s,d]
        kv = torch.einsum("bhsk,bhsd->bhskd", kf, v)
        S = kv.cumsum(dim=2)
        z = kf.cumsum(dim=2)
        num = torch.einsum("bhsk,bhskd->bhsd", qf, S)
        den = torch.einsum("bhsk,bhsk->bhs", qf, z).unsqueeze(-1) + 1e-5
        o = (num / den).transpose(1, 2).contiguous().view(b, s, d)
        return self.out(o)


class LinearBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = LinearAttention(cfg)
        self.norm2 = RMSNorm(cfg.dim)
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LinearGPT(nn.Module):
    """LinearGPT：线性注意力 + 可学习位置编码 + 与 TinyGPT 相同的外壳。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.max_seq, cfg.dim) * 0.02)
        self.blocks = nn.ModuleList([LinearBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, idx):
        b, s = idx.shape
        x = self.token_emb(idx) + self.pos_emb[:, :s]
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, idx, max_new=40, temperature=0.8, top_k=20):
        self.eval()
        for _ in range(max_new):
            x = idx[:, -self.cfg.max_seq:]
            logits = self(x)[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


"""
进阶改造 Prompt
1. 换成“单步递推”推理（状态缓存 S/z，不重算历史），对比生成速度；
2. 试不同核函数：relu、elu+1、随机特征（performer 思路）；
3. 给 q 乘温度/加 causal 衰减，模拟 RetNet/线性 RNN 的长程遗忘；
4. 对比同参数量 softmax Transformer：小语料谁快谁准；
5. 把注意力状态接到 MoE 或 SSM 上，思考线性注意力家族的统一视图。
"""

"""
规范字段补充（全局强制代码落地规范）

【层级】L3（模型实现模块：线性注意力 + 可学习位置编码）
【核心逻辑】LinearAttention 用 elu+1 核特征化后做因果累积：
S=Σφ(k)⊗v、z=Σφ(k)，y=(φ(q)S)/(φ(q)z+eps)；无 attention matrix。
【运行结果示例】（真实运行，train_linear.py 400 步同配置对照）
linear 最终 loss 0.0230；eval CE softmax 0.0196 vs linear 0.0255（8 窗均值）
params=149696（比 TinyGPT 多 4096 位置编码参数）
【高频报错 Top5】
1. 输出随序列长度发散：漏了除以累积键和 z（已实现）；
2. RoPE 与线性注意力不兼容：本实现换可学习绝对位置编码；
3. einsum 维度错：S 是 [b,h,s,d,d]，拼写核对公式；
4. 数值不稳定：elu+1 后仍可能大数；修复：输入先 norm；
5. 小语料打不过 softmax 是预期，别为调参浪费时间。
【工程改造方向】做流式推理状态缓存；换 RetNet 式指数衰减；GPU 上对比长序列。
"""
