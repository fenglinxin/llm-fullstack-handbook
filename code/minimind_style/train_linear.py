# -*- coding: utf-8 -*-
"""
MiniMind-Linear 教学版训练（CPU 1-3 分钟）

【环境依赖】Python 3.10+, PyTorch 2.x。

【前置】data/corpus.txt 已入库；语料/窗口/超参与 pretrain.py 对齐，
方便和 softmax 版 TinyGPT（out/pretrain.pt）做同配置对照。

【核心逻辑】
1. 相同滑窗 next-token 预训练循环（标签左移一位）；
2. 模型换成 LinearGPT（线性注意力 + 可学习位置编码）；
3. 打印 loss 与生成样例，供与 softmax 基线对比。

【关键参数】--steps 400 --dim 64 --layers 2 --seq 64（与 pretrain.py 一致）。

【避坑】
1. 位置编码从 RoPE 换成可学习绝对位置，不同 seq 长不能直接套用旧词表；
2. einsum 的 S 矩阵让训练显存仍是 O(n·d^2)，别误以为训练一定更省；
3. 对比时保持 steps/seed/batch 完全一致，只换注意力才算公平。

【输出解读】
- loss 从约 ln(vocab)≈5.9 下降即训练正常；
- 与 softmax 版对照：若两者 loss 接近，说明线性注意力在小语料上
  损失不大；若偏高，正是“线性注意力适合超长序列”的学费。
"""

import argparse
import pathlib
import random

import torch

from linear_model import LinearGPT
from model import TinyConfig
from tokenizer import CharTokenizer
import pretrain  # 复用 make_windows / sample_text

ROOT = pathlib.Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="out/linear.pt")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    corpus = (ROOT / "data" / "corpus.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer()
    tokenizer.fit([corpus])
    ids = tokenizer.encode(corpus)
    windows = pretrain.make_windows(ids, args.seq)
    print("vocab:", len(tokenizer.vocab), "windows:", len(windows))

    cfg = TinyConfig(vocab_size=len(tokenizer.vocab), dim=args.dim,
                     n_layers=args.layers, n_heads=args.heads, max_seq=args.seq)
    model = LinearGPT(cfg)
    print("linear params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for step in range(args.steps):
        batch_ids = [random.choice(windows) for _ in range(args.batch)]
        x = torch.tensor(batch_ids, dtype=torch.long)
        logits = model(x[:, :-1])
        loss = loss_fn(logits.view(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == args.steps - 1:
            text = pretrain.sample_text(model, tokenizer, "cpu")
            print("step %4d loss %.4f | %s" % (step, loss.item(), text[:40]))

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "cfg": {"vocab_size": cfg.vocab_size, "dim": cfg.dim,
                        "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                        "max_seq": cfg.max_seq}},
               out_path)
    print("saved ->", out_path)


"""
进阶改造 Prompt
1. 跑同一份语料 100/200/400 步，画 softmax vs linear 的 loss 曲线；
2. 把 seq 拉到 128/256 对比训练速度与显存（线性注意力才开始显优势）；
3. 给线性注意力加指数衰减（RetNet 式），观察长程依赖是否改善；
4. 试 relu 核 vs elu+1 核的数值稳定性；
5. 流式解码场景（一次只来一个新 token）对比重算 vs 状态递推的耗时。
"""

if __name__ == "__main__":
    main()

"""
规范字段补充（全局强制代码落地规范）

【层级】L2（工程训练：线性注意力与 softmax 基线的同配置对照）
【核心逻辑】与 pretrain.py 相同语料/窗口/种子/步数，只换模型为 LinearGPT；
对比目的：线性注意力在小语料上的精度代价与机制完整性。
【运行结果示例】（真实运行，CPU，400 步）
step 399 loss 0.0230 | 人工智能正在改变世界。大语言模型通过预测下一个词来学习
saved -> .../out/linear.pt
（生成与 Dense 几乎一致；同配置 eval CE 略高 0.0196->0.0255，属预期）
【高频报错 Top5】
1. 想对比必须同 seed/steps/batch，否则结论无效；
2. 位置编码参数不同导致 state_dict 不兼容：别混用 checkpoint；
3. CPU 下 400 步约 1-2 分钟，加大 seq 后线性收益才明显；
4. 误以为训练更快：本实现显式 S 仍是 O(n*d^2)；
5. 长序列评测请上 GPU 或截断窗口。
【工程改造方向】流式状态推理封装；performer 随机特征核；因果衰减变体。
"""
