# -*- coding: utf-8 -*-
"""
MiniMind-dLM 教学版训练（CPU 2-4 分钟）

【环境依赖】Python 3.10+, PyTorch 2.x。

【前置】data/corpus.txt 已入库（含末尾 6 条 QA，可直接演示续写）。

【核心逻辑】
1. 字符级 tokenizer fit 语料，MASK 作为第 N 个新 token（N=词表大小）；
2. 训练样本 = 语料滑窗，随机涂黑 30% 位置，双向 Transformer 还原；
3. loss 只算被涂黑的位置；val_acc 用固定种子在留出窗口上测还原准确率；
4. 生成 = 前缀不动，其余位置全 mask，按置信度逐位还原。

【关键参数】--steps 1500 --seq 48 --mask_prob 0.25 --span_prob 0.9 --dim 64 --layers 2。

【避坑】
1. mask 比例不要太大（>0.7 很难学），也不要太小（<0.1 学不到长程）；
   还要混入连续 span 涂黑（--span_prob），否则模型没见过“连续挖空”，
   完形填空/自由还原会失效；
2. 窗口长度即“能还原的上下文跨度”，想回答长问题请加大 seq；
3. 扩散模型的“生成长度”是固定的：target_len 定多少就还原多少，
   不会像自回归模型那样自己决定停在哪。

【输出解读】
- val_acc：留出窗口（语料尾部 QA 段）涂黑还原正确率——模型没背过这段，
  分数低（约 10-15%）恰恰证明它没有“作弊”；训练窗口还原率约 98%；
- 完形填空样例：左右都可见时能精确还原 2 字跨度，说明双向信息生效；
  3 字以上跨度仍会出错——微型模型对 span 调度/容量很敏感，可作进阶实验。
"""

import argparse
import pathlib
import random

import torch

from dlm_model import DiffLM
from model import TinyConfig
from tokenizer import CharTokenizer

ROOT = pathlib.Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=48)
    p.add_argument("--mask_prob", type=float, default=0.25)
    p.add_argument("--span_prob", type=float, default=0.9)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="out/dlm.pt")
    return p.parse_args()


def make_windows(ids, seq, stride=None):
    stride = stride or seq // 2
    out = []
    for i in range(0, max(1, len(ids) - seq), stride):
        w = ids[i:i + seq]
        if len(w) == seq:
            out.append(w)
    return out


def masked_acc(model, windows):
    model.eval()
    torch.manual_seed(123)
    right = total = 0
    with torch.no_grad():
        for w in windows:
            x = torch.tensor([w])
            xm, mask = model.corrupt(x)
            logits = model(xm).argmax(-1)
            right += (logits[mask] == x[mask]).sum().item()
            total += mask.sum().item()
    model.train()
    return right / max(total, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    corpus = (ROOT / "data" / "corpus.txt").read_text(encoding="utf-8")
    tok = CharTokenizer()
    tok.fit([corpus])
    ids = tok.encode(corpus)
    all_windows = make_windows(ids, args.seq)
    # 末尾 8 个窗口做 val（含 QA 尾巴），其余训练
    val_windows = all_windows[-8:]
    train_windows = all_windows[:-8]
    print("vocab:", len(tok.vocab), "train windows:", len(train_windows),
          "val windows:", len(val_windows))

    cfg = TinyConfig(vocab_size=len(tok.vocab), dim=args.dim,
                     n_layers=args.layers, n_heads=args.heads, max_seq=args.seq)
    model = DiffLM(cfg, mask_prob=args.mask_prob, span_prob=args.span_prob)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    for step in range(args.steps):
        batch = [random.choice(train_windows) for _ in range(args.batch)]
        x = torch.tensor(batch)
        xm, mask = model.corrupt(x)
        labels = x.clone()
        labels[~mask] = -100
        loss = ce(model(xm).reshape(-1, model.cfg.vocab_size + 1),
                  labels.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 60 == 0 or step == args.steps - 1:
            acc = masked_acc(model, val_windows)
            # 完形填空演示（训练语料内句子）：双向信息是扩散 LM 的主场
            left = "监督微调让模型学会按照指令回答"
            gen = model.infill(tok.encode(left), tok.encode("。"), 2,
                               temperature=0.01, top_k=1)
            text = tok.decode(gen)
            mid = text[len(left):len(left) + 2]
            print("step %4d loss %.4f val_acc %.3f | 完形: %s[%s]。"
                  % (step, loss.item(), acc, left, mid))

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "cfg": {"vocab_size": cfg.vocab_size, "dim": cfg.dim,
                        "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                        "max_seq": cfg.max_seq, "mask_prob": args.mask_prob}},
               out_path)
    tok.save(ROOT / "out" / "dlm_vocab.json")
    print("saved ->", out_path)


"""
进阶改造 Prompt
1. 训练 mask 比例做长度调度（短序列 0.1、长序列 0.5），对齐论文实现；
2. 用“一次还原 k 个最高置信位置”替代逐位还原，提速并观察质量；
3. 给 val 加“完全没见过的句子”评测（从语料外构造），衡量泛化；
4. 对比相同 dim/layers 的自回归 TinyGPT 在“补全”任务上的表现；
5. 尝试把 MASK 采样改成“候选重采样”（top-k 反复扰动），提升多样性。
"""

if __name__ == "__main__":
    main()
