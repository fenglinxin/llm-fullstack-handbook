# -*- coding: utf-8 -*-
"""
TinyGPT 生成 Demo：加载预训练权重后交互/命令行生成

【环境依赖】Python 3.10+, PyTorch 2.x

【运行】
python generate.py --prompt "人工智能" --max_new 40 --temperature 0.8

【说明】先运行 pretrain.py 生成 out/pretrain.pt 与 out/vocab.json。
"""

import argparse
import pathlib

import torch

from model import TinyConfig, TinyGPT
from tokenizer import CharTokenizer

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="人工智能")
    p.add_argument("--max_new", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--ckpt", type=str, default="out/pretrain.pt")
    args = p.parse_args()

    tokenizer = CharTokenizer.load(ROOT / "out" / "vocab.json")
    data = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=True)
    cfg = TinyConfig(**data["cfg"])
    model = TinyGPT(cfg)
    model.load_state_dict(data["model"])
    model.eval()

    ids = tokenizer.encode(args.prompt)
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(
        x,
        max_new=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("生成：", tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
