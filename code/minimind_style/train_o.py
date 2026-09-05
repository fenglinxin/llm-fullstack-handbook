# -*- coding: utf-8 -*-
"""
MiniMind-O 教学版训练（CPU 2-4 分钟）

【环境依赖】Python 3.10+, PyTorch 2.x；无 torchaudio/torchvision 依赖。

【前置】先跑 data/make_omni_data.py 生成 data/omni.jsonl
（视觉样本的图案渲染复用 vision_model.render_pattern）。

【核心逻辑】
1. tokenizer 在“基础语料 + 音频问答 + 看图问答”文本上 fit；
2. 音频样本走 AudioTower（2 token），图像样本走 VisionTower（36 token）；
3. 两条路径共享同一个文本解码器，batch 内分模态各自算 loss 后合并反传；
4. 训练/测试按 noise_seed 划分（0-2 / 3-4），考验模态特征的泛化。

【关键参数】--epochs 60 --batch_a 8 --batch_v 4 --dim 64 --layers 2。

【避坑】
1. 音频特征必须在训练/推理用同一函数提取，否则频点对不上；
2. 别把 audio 与 vision 样本塞进同一个张量（token 数不同），
   教学版分开 forward 即可；
3. 两个模态的 loss 直接相加前可观察各自量级是否接近。

【输出解读】audio_acc（3 类随机 33%）与 vision_acc（8 类随机 12.5%）
都明显高于随机，说明同一个解码器真的同时吃到了两种模态。
"""

import argparse
import json
import math
import pathlib
import random

import torch

from model import TinyConfig
from tokenizer import CharTokenizer
from omni_model import (MiniMindO, TONE_NAMES, audio_features, render_tone)
from vision_model import render_pattern, CAPTIONS

ROOT = pathlib.Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_a", type=int, default=8)
    p.add_argument("--batch_v", type=int, default=4)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="out/o.pt")
    return p.parse_args()


def load_rows():
    rows = [json.loads(line) for line in open(ROOT / "data" / "omni.jsonl", encoding="utf-8")]
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


def text_ids(tok, row, cap):
    """prompt+answer 截断，cap 给模态 token 留位置（audio=2, vision=36）。"""
    prompt = "问：" + row["q"] + "答："
    return tok.encode(prompt + row["a"])[: cap]


def collate_a(rows, tok, max_len):
    feats, ids = [], []
    for r in rows:
        wave = render_tone(r["freq"], r["noise_seed"])
        feats.append(audio_features(wave))
        ids.append(text_ids(tok, r, max_len - 2))
    L = max(len(x) for x in ids)
    m = max_len
    pad = torch.zeros(len(ids), L, dtype=torch.long)
    for i, x in enumerate(ids):
        pad[i, : len(x)] = torch.tensor(x)
    return torch.stack(feats), pad


def collate_v(rows, tok, max_len):
    imgs, ids = [], []
    for r in rows:
        imgs.append(render_pattern(r["pid"], r["noise_seed"]))
        ids.append(text_ids(tok, r, max_len - 36))
    L = max(len(x) for x in ids)
    pad = torch.zeros(len(ids), L, dtype=torch.long)
    for i, x in enumerate(ids):
        pad[i, : len(x)] = torch.tensor(x)
    return torch.stack(imgs), pad


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_rows, test_rows = load_rows()
    train_a = [r for r in train_rows if r["modality"] == "audio"]
    train_v = [r for r in train_rows if r["modality"] == "vision"]
    test_a = [r for r in test_rows if r["modality"] == "audio"]
    test_v = [r for r in test_rows if r["modality"] == "vision"]
    tok = build_tokenizer(train_rows, test_rows)
    print("vocab:", len(tok.vocab), "| train audio", len(train_a),
          "vision", len(train_v), "| test audio", len(test_a),
          "vision", len(test_v))

    cfg = TinyConfig(vocab_size=len(tok.vocab), dim=args.dim,
                     n_layers=args.layers, n_heads=args.heads, max_seq=args.seq)
    model = MiniMindO(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def gen_answer_a(row):
        wave = render_tone(row["freq"], row["noise_seed"])
        feats = audio_features(wave).unsqueeze(0)
        gen = model.generate_audio(feats, tok.encode("问：" + row["q"] + "答："),
                                   max_new=8, temperature=0.01, top_k=1)
        text = tok.decode(gen)
        return text.split("答：", 1)[1] if "答：" in text else text

    def gen_answer_v(row):
        img = render_pattern(row["pid"], row["noise_seed"])
        gen = model.generate_vision(img, tok.encode("问：" + row["q"] + "答："),
                                    max_new=12, temperature=0.01, top_k=1)
        text = tok.decode(gen)
        return text.split("答：", 1)[1] if "答：" in text else text

    def evaluate():
        model.eval()
        ra = sum(1 for r in test_a if gen_answer_a(r).startswith(r["a"])) / len(test_a)
        rv = sum(1 for r in test_v if gen_answer_v(r).startswith(r["a"])) / len(test_v)
        model.train()
        return ra, rv

    steps = args.epochs * max(1, max(len(train_a) // args.batch_a,
                                     len(train_v) // args.batch_v))
    for step in range(steps):
        ba = random.choices(train_a, k=args.batch_a)
        bv = random.choices(train_v, k=args.batch_v)
        fa, ia = collate_a(ba, tok, args.seq)
        iv, iv_ = collate_v(bv, tok, args.seq)
        la = model.loss_audio(fa, ia)
        lv = model.loss_vision(iv, iv_)
        loss = la + lv
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 30 == 0 or step == steps - 1:
            aa, va = evaluate()
            row = test_a[0]
            print("step %3d loss_a %.3f loss_v %.3f | audio_acc %.3f vision_acc %.3f | %s"
                  % (step, la.item(), lv.item(), aa, va, gen_answer_a(row)[:10]))

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "cfg": {"vocab_size": cfg.vocab_size, "dim": cfg.dim,
                        "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                        "max_seq": cfg.max_seq}},
               out_path)
    tok.save(ROOT / "out" / "o_vocab.json")
    print("saved ->", out_path)


"""
进阶改造 Prompt
1. 用统一的模态 token 序列训练（加 [AUDIO]/[IMG] 标记 token），
   对比本版“分路径 forward”的精度差异；
2. 增加音频+图像联合样本，考验跨模态组合理解；
3. 音频特征升级为 log-mel 谱（小 FFT），图像换真实灰度图；
4. 用 MiniMind-V 的 checkpoint 初始化视觉塔（两阶段训练），看是否加速；
5. 记录每种模态的 loss 权重，思考不平衡时如何调（主线 21/22 章数据配比）。
"""

if __name__ == "__main__":
    main()
