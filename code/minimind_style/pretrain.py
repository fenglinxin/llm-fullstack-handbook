# -*- coding: utf-8 -*-
"""
TinyGPT 微型预训练（MiniMind 式最小闭环，CPU 可跑）

【环境依赖】Python 3.10+, PyTorch 2.x；CPU 1-3 分钟可完成默认配置。

【运行】python pretrain.py --steps 200 --dim 64 --layers 2 --heads 2 --seq 64

【核心逻辑】
1. 读语料 -> CharTokenizer -> token id 序列；
2. 滑窗切样本，随机采样 batch；
3. TinyGPT 前向算全序列交叉熵——注意标签要左移一位（next-token），
   否则模型会退化成“抄当前字符”，生成时只会无限复读最后一个字；
4. 反向更新，周期性打印 loss 与生成样例。

【关键参数】见 argparse：steps/dim/layers/heads/seq/lr/batch/seed。

【避坑】
1. 语料小、模型小时 loss 会快速下降，但别期待“像人话”；
2. loss 从 ln(vocab) 附近开始下降即为正常；忘记左移标签会让 loss 假性归零，生成却只会复读；
3. 想复现请固定 --seed 0。

【输出解读】
- loss 从约 ln(vocab)≈5.9 降到 1-2 说明学到字符/词搭配；
- 生成样例会出现语料中的词片段即说明学习生效。
"""

import argparse
import pathlib
import random

import torch

from model import TinyConfig, TinyGPT
from tokenizer import CharTokenizer

ROOT = pathlib.Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="out/pretrain.pt")
    return p.parse_args()


def make_windows(ids, seq, stride=None):
    stride = stride or seq // 2
    windows = []
    for i in range(0, max(1, len(ids) - seq), stride):
        windows.append(ids[i : i + seq])
    return [w for w in windows if len(w) == seq]


def sample_text(model, tokenizer, device, prompt="人工智能", max_new=24):
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new=max_new, temperature=0.8, top_k=20)
    return tokenizer.decode(out[0].tolist())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    corpus = (ROOT / "data" / "corpus.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer()
    tokenizer.fit([corpus])
    ids = tokenizer.encode(corpus)
    windows = make_windows(ids, args.seq)
    print("vocab:", len(tokenizer.vocab), "windows:", len(windows))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TinyConfig(
        vocab_size=len(tokenizer.vocab),
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        max_seq=args.seq,
    )
    model = TinyGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for step in range(args.steps):
        batch_ids = [random.choice(windows) for _ in range(args.batch)]
        x = torch.tensor(batch_ids, dtype=torch.long, device=device)
        logits = model(x[:, :-1])              # 用前 seq-1 个位置预测
        labels = x[:, 1:]                      # 标签整体左移一位 = next-token
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 20 == 0 or step == args.steps - 1:
            text = sample_text(model, tokenizer, device)
            print("step %4d loss %.4f | %s" % (step, loss.item(), text[:40]))

    # 保存模型与分词器
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": {
                "vocab_size": cfg.vocab_size,
                "dim": cfg.dim,
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "max_seq": cfg.max_seq,
            },
        },
        out_path,
    )
    tokenizer.save(ROOT / "out" / "vocab.json")
    print("saved ->", out_path)


"""
进阶改造 Prompt
1. 换更大/真实语料（主线 04-10 章数据管道思路），观察 loss 与生成提升；
2. 加入学习率 warmup/cosine（主线 12 章）；
3. 加 checkpoint/断点续训（主线 13 章）；
4. 加 wandb/swanlab 可视化；
5. 用真实 tokenizer（BPE）替换字符级。
"""

if __name__ == "__main__":
    main()
