# -*- coding: utf-8 -*-
"""
Transformer 落地代码 2/2：手写注意力的最小 Encoder-Decoder（序列反转任务）

【环境依赖】
- Python 3.10+, PyTorch 2.x；CPU 可运行
- 复用同目录 attention_from_scratch.py 的组件

【任务】给定一串数字，输出它的反转序列。
- 例：输入 [3, 1, 4] -> 输出 [4, 1, 3]
- 任务简单但能验证：编码器理解全序列、解码器自回归生成

【参数调优模板】
- d_model=32, n_heads=4, max_len=20, epochs=200, lr=1e-3
- 调参顺序：先调 lr，再调 d_model/层数，最后动 head 数

【避坑】
1. 解码器必须用因果 mask，否则训练时“看到未来答案”；
2. 训练用 teacher forcing，推理要逐 token 生成并拼接；
3. 数字要 pad 到同长并给 padding mask（本 Demo 简化固定长度）。

【输出解读】
- 训练 loss 下降；
- 最终 accuracy 接近 1.0 说明学会了反转规则；
- 若 accuracy 为 0，多半是 mask/位置编码写错。

【工程改造方向】
- 换成真正文本：用 tokenizer 替代整数词表；
- 加深 encoder/decoder 层、加 FFN；
- 训练完接主线 15 章 SFT 流程，替换为开源底座。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_from_scratch import MultiHeadAttentionManual, SinusoidalPositionalEncoding


def causal_mask(seq):
    return torch.tril(torch.ones(seq, seq)).unsqueeze(0).unsqueeze(0)


class CrossAttentionManual(nn.Module):
    """手写交叉注意力：Q 来自解码器，K/V 来自编码器。"""
    def __init__(self, d_model=32, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def split(self, x):
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, q, kv):
        q = self.split(self.wq(q))
        k = self.split(self.wk(kv))
        v = self.split(self.wv(kv))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        w = F.softmax(scores, dim=-1)
        out = torch.matmul(w, v)
        b, _, s, _ = out.shape
        return self.wo(out.transpose(1, 2).contiguous().view(b, s, -1))


class MiniSeq2Seq(nn.Module):
    """最小编解码：编码器自注意力 + 解码器（自注意力+交叉注意力）。"""
    def __init__(self, vocab=20, d_model=32, n_heads=4, max_len=20):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pe = SinusoidalPositionalEncoding(d_model, max_len)
        self.enc_attn = MultiHeadAttentionManual(d_model, n_heads)
        self.dec_attn = MultiHeadAttentionManual(d_model, n_heads)
        self.cross = CrossAttentionManual(d_model, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model))
        self.out = nn.Linear(d_model, vocab)

    def encode(self, src):
        x = self.pe(self.embed(src))
        return self.ffn(self.enc_attn(x)) + x

    def decode_step(self, tgt_embed, enc_out):
        # 自注意力（因果）
        seq = tgt_embed.shape[1]
        h = self.dec_attn(tgt_embed, mask=causal_mask(seq))
        h = self.cross(h, enc_out)
        return self.out(self.ffn(h) + h)

    def forward(self, src, tgt):
        enc = self.encode(src)
        tgt_embed = self.pe(self.embed(tgt))
        logits = self.decode_step(tgt_embed, enc)
        return logits

    @torch.no_grad()
    def generate(self, src, max_len=10):
        self.eval()
        enc = self.encode(src)
        out = torch.full((src.shape[0], 1), 0, dtype=torch.long)  # SOS=0
        for _ in range(max_len):
            tgt_embed = self.pe(self.embed(out))
            logits = self.decode_step(tgt_embed, enc)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
        return out


def make_batch(batch_size, max_src=6, vocab=20):
    src = torch.randint(2, vocab, (batch_size, max_src))
    tgt = torch.flip(src, dims=[1])  # 反转任务
    sos = torch.zeros(batch_size, 1, dtype=torch.long)
    tgt_in = torch.cat([sos, tgt[:, :-1]], dim=1)  # teacher forcing 输入
    return src, tgt_in, tgt


def main():
    torch.manual_seed(0)
    model = MiniSeq2Seq()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(300):
        src, tgt_in, tgt = make_batch(32)
        logits = model(src, tgt_in)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print("step %d loss %.4f" % (step, loss.item()))

    # 验证
    src, _, tgt = make_batch(8)
    pred = model.generate(src, max_len=tgt.shape[1])[:, 1:]
    acc = (pred == tgt).float().mean().item()
    print("valid accuracy = %.3f" % acc)


"""进阶改造 Prompt
1. 加 FFN/层数/多头数并复现“更深更好”的边界；
2. 把反转任务换成加法/翻译小词表，观察任务难度变化；
3. 对比正弦 PE 与 RoPE 的外推；
4. 对 src 加 padding mask，支持变长 batch；
5. 训练后接主线 SFT/部署流程，把它换成真实 LLM 底座。
"""

if __name__ == "__main__":
    main()
