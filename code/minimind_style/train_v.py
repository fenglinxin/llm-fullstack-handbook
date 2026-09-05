# -*- coding: utf-8 -*-
"""
MiniMind-V 教学版训练（CPU 2-4 分钟）

【环境依赖】Python 3.10+, PyTorch 2.x；无 torchvision 依赖。

【前置】先跑 data/make_vision_data.py 生成 data/vision.jsonl。

【核心逻辑】
1. 字符级 tokenizer 在“基础语料 + 看图问答文本”上 fit（保证无 UNK）；
2. MiniMindV：12x12 图 -> 36 个图像 token，与“问：...答：”文本拼接；
3. loss 只统计文本段（图像 token 不产生预测目标）；
4. 训练/测试按图像实例划分：8 种图案都训练，但测试用的是
   “没见过的噪声实例”，只有学到图案概念而不是死记图像的模型能答对。

【关键参数】--epochs 60 --batch 4 --dim 64 --layers 2 --seq 64。

【避坑】
1. 图像 token 数(36) + 文本长度要小于 max_seq，否则生成被截断；
2. 只看 train loss 会骗人：小模型背训练样本很容易，
   必须看“未见实例”的 val acc；
3. 回答比较用前缀匹配（模型可能生成答案后又继续输出别的内容）。

【输出解读】val acc 明显高于 12.5%（8 类随机）即说明视觉通路
真的把图案信息传给了文本解码器，且学到的是可泛化的图案概念。
"""

import argparse
import json
import pathlib
import random

import torch

from model import TinyConfig
from tokenizer import CharTokenizer
from vision_model import MiniMindV, CAPTIONS, render_pattern

ROOT = pathlib.Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="out/v.pt")
    return p.parse_args()


def load_rows():
    rows = [json.loads(line) for line in open(ROOT / "data" / "vision.jsonl", encoding="utf-8")]
    train = [r for r in rows if r["noise_seed"] < 3]
    test = [r for r in rows if r["noise_seed"] >= 3]
    return train, test


def build_tokenizer(train_rows, test_rows):
    texts = [open(ROOT / "data" / "corpus.txt", encoding="utf-8").read()]
    for r in train_rows + test_rows:
        texts.append("问：" + r["q"] + "答：" + r["a"])
    tok = CharTokenizer()
    tok.fit(texts)
    return tok


def make_sample(tok, row, max_seq):
    prompt = "问：" + row["q"] + "答："
    ids = tok.encode(prompt + row["a"])[:max_seq - 36]  # 给 36 个图像 token 留位置
    return row["pid"], row["noise_seed"], ids


def collate(batch):
    pids = torch.tensor([b[0] for b in batch])
    seeds = torch.tensor([b[1] for b in batch])
    imgs = torch.stack([render_pattern(int(p), int(s)) for p, s in zip(pids, seeds)])
    max_len = max(len(b[2]) for b in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b[2])] = torch.tensor(b[2])
    return imgs, ids


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_rows, test_rows = load_rows()
    tok = build_tokenizer(train_rows, test_rows)
    train_samples = [make_sample(tok, r, args.seq) for r in train_rows]
    test_samples = [make_sample(tok, r, args.seq) for r in test_rows]
    print("vocab:", len(tok.vocab), "train samples:", len(train_samples),
          "test samples:", len(test_samples))

    cfg = TinyConfig(vocab_size=len(tok.vocab), dim=args.dim,
                     n_layers=args.layers, n_heads=args.heads, max_seq=args.seq)
    model = MiniMindV(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def evaluate():
        model.eval()
        right = total = 0
        with torch.no_grad():
            for (pid, seed, _ids), row in zip(test_samples, test_rows):
                img = render_pattern(pid, seed)
                prompt_ids = tok.encode("问：" + row["q"] + "答：")
                gen = model.generate(img, prompt_ids, max_new=16,
                                     temperature=0.01, top_k=1)
                text = tok.decode(gen)
                after = text.split("答：", 1)[1] if "答：" in text else ""
                if after.startswith(row["a"]):
                    right += 1
                total += 1
        model.train()
        return right / max(total, 1)

    # 训练
    steps_per_epoch = max(1, len(train_samples) // args.batch)
    total_steps = args.epochs * steps_per_epoch
    for step in range(total_steps):
        batch = random.choices(train_samples, k=args.batch)
        imgs, ids = collate(batch)
        loss = model.loss(imgs, ids)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == total_steps - 1:
            acc = evaluate()
            # 生成一条测试样例看效果
            row = test_rows[0]
            img = render_pattern(row["pid"], row["noise_seed"])
            gen = model.generate(img, tok.encode("问：" + row["q"] + "答："),
                                 max_new=12, temperature=0.6, top_k=8)
            print("step %3d loss %.4f val_acc %.3f | %s"
                  % (step, loss.item(), acc,
                     tok.decode(gen)[-18:].replace("\n", "")))

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "cfg": {"vocab_size": cfg.vocab_size, "dim": cfg.dim,
                        "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                        "max_seq": cfg.max_seq}},
               out_path)
    tok.save(ROOT / "out" / "v_vocab.json")
    print("saved ->", out_path)


"""
进阶改造 Prompt
1. 把测试图案也加入训练集，验证“见过的图”是否 100% 背住（对比过拟合曲线）；
2. 图像加旋转/平移增强，测试鲁棒性；
3. 把 CAPTIONS 换成更细的描述（位置+形状+数量），扩大词表；
4. 视觉塔加一层双向注意力（参考 MiniMind-O 思路），看 val acc 是否提升；
5. 对比不同 patch 大小（1x1/2x2/4x4）对精度与参数量的影响。
"""

if __name__ == "__main__":
    main()
