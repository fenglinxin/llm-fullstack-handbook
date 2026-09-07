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

"""
规范字段补充（全局强制代码落地规范）

【层级】L1（推理演示脚本：加载 checkpoint 做温度/top-k 采样）
【运行结果示例】（真实运行，CPU，加载 out/pretrain.pt）
$ python generate.py --prompt "人工智能"
生成： 人工智能大语言模型通过预测下一个词来学习语言规律。
预训练需要大量高质量
（输出与语料句子一致 = 预训练生效；temperature 越大越随机）
【高频报错 Top5】
1. FileNotFoundError out/pretrain.pt：先跑 pretrain.py；
2. 输出全是 UNK/乱码：vocab 与模型不匹配；修复：重跑 pretrain 生成新 vocab；
3. top_k 过滤后候选为空：temperature 过小+logits 平；修复：temperature>=0.3；
4. 中文在 Windows 控制台乱码：PYTHONIOENCODING=utf-8；
5. 生成重复循环：小模型过拟合语料，属预期；换大语料/加重复惩罚。
【工程改造方向】封装成 ChatBot 类接 HTTP/WebSocket 流式输出（主线 34 章）；
加 KV Cache/投机解码（主线 33/38 章）时只改 generate 内部循环。
"""

"""
规范字段补充 2（补齐缺失字段标记）

【核心逻辑】
- 加载 out/pretrain.pt 的 TinyGPT，用字符级 tokenizer 把 prompt 编码；
- 自回归循环：logits 取最后一位 -> temperature 缩放 -> top_k 截断 -> 采样拼接。
【关键参数】
- --prompt（默认“人工智能”）：起始文本；
- --max_new 40（默认；越大生成越长）：新生成 token 数；
- --temperature 0.8（默认；0.3-1.0 按随机度需求调）：采样温度；
- --top_k 20（默认；0=关闭）：候选截断数。
【避坑】
- 生成质量受 checkpoint 训练数据限制，小语料会过拟合复读；
- top_k 与 temperature 同时用，数值小的先过滤。
【输出解读】前缀 + 生成文本一起打印；文本与语料风格一致 = 生效。
"""
