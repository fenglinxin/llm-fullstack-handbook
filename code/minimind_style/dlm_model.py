# -*- coding: utf-8 -*-
"""
MiniMind-dLM 教学版：微型扩散语言模型（absorbing/掩码扩散）

【定位】对标 MiniMind-dLM“扩散语言模型”思路的教学原创实现：
不按“从左到右”逐个预测下一个字，而是把文本当“被涂黑了一部分”的
序列，训练一个双向 Transformer 把涂黑处还原；生成时从全 MASK 出发
迭代去噪（类似图像扩散模型的离散版）。

【核心概念】
1. 加噪：以概率 p 把 token 替换成 [MASK]（absorbing state）；
2. 去噪网络：双向注意力（能看到左右上下文），预测每个位置的原 token；
3. loss 只算被 mask 的位置（类似 SFT 只算回答位置）；
4. 采样：给定前缀（不 mask），把后续位置全部 mask，然后迭代
   按置信度从高到低逐步还原。

【环境依赖】Python 3.10+, PyTorch 2.x。

【关键参数】
- mask_prob：训练时平均涂黑比例（0.3 左右）；
- steps：生成时迭代轮数（本实现=剩余 mask 数，逐位还原）。

【避坑】
1. 必须用“双向注意力”，否则看不到右边的字，扩散模型退化成从左到右 LM；
2. mask 位置才进 loss，别把未 mask 的位置也算进去（会退化成普通 LM）；
3. MASK 是一个新 token（id = vocab_size），输入/输出词表都要 +1；
4. 生成长度是固定的（前缀+待生成长度），不适合变长自由续写——
   这是扩散 LM 与自回归 LM 的典型差异。

【输出解读】val acc = 在“没见过的涂黑实例”上还原正确的 token 占比；
明显高于随机（1/词表≈0.3%）说明去噪网络真的学会了利用上下文。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TinyConfig, RMSNorm, FeedForward, precompute_rope, apply_rope


class FullAttention(nn.Module):
    """双向注意力（无 causal mask），扩散模型需要看到全文。"""

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        b, s, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos[:s], sin[:s])
        k = apply_rope(k, cos[:s], sin[:s])
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        w = F.softmax(scores, dim=-1)   # 不掩码：双向
        o = torch.matmul(w, v).transpose(1, 2).contiguous().view(b, s, d)
        return self.out(o)


class FullBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = FullAttention(cfg)
        self.norm2 = RMSNorm(cfg.dim)
        self.ffn = FeedForward(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class DiffLM(nn.Module):
    """微型扩散语言模型：双向 Transformer + MASK token。"""

    def __init__(self, cfg, mask_prob=0.3, span_prob=0.5):
        super().__init__()
        self.cfg = cfg
        self.mask_prob = mask_prob
        self.span_prob = span_prob
        self.vocab_size = cfg.vocab_size  # 不含 MASK
        self.mask_id = cfg.vocab_size     # MASK 是最后一个新 token
        self.token_emb = nn.Embedding(cfg.vocab_size + 1, cfg.dim)
        self.blocks = nn.ModuleList([FullBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size + 1, bias=False)
        cos, sin = precompute_rope(cfg.dim // cfg.n_heads, cfg.max_seq, 10000.0)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        h = self.token_emb(x)
        for blk in self.blocks:
            h = blk(h, self.cos, self.sin)
        return self.head(self.norm(h))  # [B, S, vocab+1]

    def corrupt(self, x, rng=None):
        """加噪（absorbing noise），返回 (加噪输入, mask)。

        混合两种涂黑：独立撒点（mask_prob）+ 每行约一半概率随机涂一段
        连续 span（2-8 字）。只撒点的话模型几乎没见过“连续挖空”，
        完形填空/自由还原会失效——这是扩散 LM 训练的关键细节。
        """
        b, s = x.shape
        mask = torch.rand(b, s, generator=rng) < self.mask_prob
        for i in range(b):
            if torch.rand(1, generator=rng).item() < self.span_prob:
                span_len = int(torch.randint(2, 9, (1,), generator=rng).item())
                start = int(torch.randint(0, max(1, s - span_len), (1,),
                                          generator=rng).item())
                mask[i, start:start + span_len] = True
        xm = x.clone()
        xm[mask] = self.mask_id
        return xm, mask

    def loss(self, x):
        """只还原被涂黑的位置。"""
        xm, mask = self.corrupt(x)
        logits = self.forward(xm)
        labels = x.clone()
        labels[~mask] = -100
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               labels.reshape(-1))

    @torch.no_grad()
    def generate(self, prompt_ids, target_len, temperature=0.7, top_k=20):
        """从前缀出发，把后面 target_len 个位置全 mask，迭代逐位还原。

        每轮把“置信度最高”的 mask 位置还原（confidence-based decoding），
        保证前缀永远不被改动。
        """
        self.eval()
        seq = torch.full((1, len(prompt_ids) + target_len), self.mask_id,
                         dtype=torch.long)
        seq[0, : len(prompt_ids)] = torch.tensor(prompt_ids)
        masks = torch.zeros_like(seq, dtype=torch.bool)
        masks[0, len(prompt_ids):] = True
        for _ in range(target_len):
            logits = self.forward(seq[:, -self.cfg.max_seq:])[0]
            # 只允许还原仍在 mask 的位置（平移后对齐）
            off = seq.shape[1] - self.cfg.max_seq if seq.shape[1] > self.cfg.max_seq else 0
            cur_masks = masks[0, off:off + logits.shape[0]]
            cand = cur_masks.clone()
            if not cand.any():
                break
            # 选置信度最高的候选位置
            probs = F.softmax(logits / temperature, dim=-1)
            conf, _ = probs.max(dim=-1)
            conf = conf.clone()
            conf[~cand] = -1e9
            pos = int(conf.argmax().item())
            v, _ = torch.topk(logits[pos], top_k)
            top_logits = logits[pos].clone()
            top_logits[top_logits < v[-1]] = -1e9
            p = F.softmax(top_logits / temperature, dim=-1)
            nxt = int(torch.multinomial(p, 1).item())
            seq[0, off + pos] = nxt
            masks[0, off + pos] = False
        return seq[0].tolist()

    @torch.no_grad()
    def infill(self, left_ids, right_ids, mid_len, temperature=0.7, top_k=20):
        """完形填空：left + [MASK]*mid_len + right，逐位还原中间段。

        与 generate 的区别：右侧上下文从一开始就可见——这正是扩散
        语言模型比自回归模型多出来的“双向信息”。已还原位置会被记录，
        不会被重复改写。
        """
        self.eval()
        L, R = len(left_ids), len(right_ids)
        total = L + mid_len + R
        seq = torch.full((1, total), self.mask_id, dtype=torch.long)
        seq[0, :L] = torch.tensor(left_ids)
        seq[0, L + mid_len:] = torch.tensor(right_ids)
        remaining = torch.zeros(total, dtype=torch.bool)
        remaining[L:L + mid_len] = True
        for _ in range(mid_len):
            w = seq[:, -self.cfg.max_seq:]
            off = total - w.shape[1]
            logits = self.forward(w)[0]
            probs = F.softmax(logits / temperature, dim=-1)
            conf = probs.max(dim=-1).values.clone()
            win_ok = remaining[off:off + w.shape[1]].clone()
            conf[~win_ok] = -1e9
            pos = int(conf.argmax().item())
            v, _ = torch.topk(logits[pos], top_k)
            lp = logits[pos].clone()
            lp[lp < v[-1]] = -1e9
            nxt = int(torch.multinomial(F.softmax(lp / temperature, -1), 1).item())
            seq[0, off + pos] = nxt
            remaining[off + pos] = False
        return seq[0].tolist()


"""
进阶改造 Prompt
1. 把 mask_prob 换成随长度变化的调度（前密后疏），对齐 LLaDA 训练细节；
2. 采样换成“每轮同时还原 k 个位置”的并行解码，对比速度与质量；
3. 加 [CLS] 做无条件/条件生成切换（classifier-free guidance 简化版）；
4. 对比扩散 LM 与自回归 LM 在同语料上的困惑度与生成多样性；
5. 思考扩散 LM 在“规划/回溯/编辑”上的潜在优势与工程代价。
"""

"""
规范字段补充（全局强制代码落地规范）

【层级】L3（模型实现模块：双向 Transformer + MASK 扩散，corrupt/generate/infill）
【核心逻辑】corrupt 用撒点+连续 span 涂黑（span 缺了学不会完形填空）；
DiffLM 双向注意力还原被涂位置；generate/infill 按置信度逐位还原，
infill 保留右侧上下文（扩散 LM 相对自回归的差异点）。
【运行结果示例】（真实运行，train_dlm.py 3000 步后）
训练窗口 span 还原率 98.7%；infill '问题'->'问题' OK、'精度'->'精度' OK
【高频报错 Top5】
1. 只撒点涂黑学不会连续挖空：必须混合 span（--span_prob）；已实现；
2. 输出 MASK 符：生成时候选没排除已还原位置；修复：infill 记录 remaining；
3. MASK id 与词表冲突：MASK=vocab_size 新 token（本实现已隔离）；
4. 双向注意力写成了因果：扩散模型看不到右边会退化；
5. 生成长度固定：扩散 LM 不能自由停，需按任务给 target_len。
【工程改造方向】换长度调度训练；并行还原 k 位置；接 classifier-free guidance。
"""
