# -*- coding: utf-8 -*-
"""
TinyGPT 微型 SFT（MiniMind 式最小指令微调）

【环境依赖】Python 3.10+, PyTorch 2.x；CPU 1-3 分钟。

【前置】先运行 data/make_corpus.py 与 pretrain.py 生成 out/pretrain.pt 与 out/vocab.json。

【核心逻辑】
1. 数据格式：问：{question}答：{answer}
2. 回答部分参与 loss（指令部分用 mask 屏蔽），体现 SFT 的 response-only loss 思想；
3. 在预训练模型上继续微调，观察“被问到特定问题时会接出答案片段”。

【关键参数】--epochs 30 --batch 4 --lr 2e-4 --max_seq 64

【避坑】
1. 必须先重跑 make_corpus.py 让词表包含“问/答/：”等字符，否则新字符是 UNK；
2. mask 算错会导致模型“背问题”，这里用字符偏移实现最小版；
3. 语料太小会快速过拟合，属教学预期。
"""

import argparse
import json
import pathlib
import random

import torch

from model import TinyConfig, TinyGPT
from tokenizer import CharTokenizer

ROOT = pathlib.Path(__file__).resolve().parent


def make_sample(tokenizer, q, a, max_seq):
    prefix = "问：" + q + "答："
    full = prefix + a
    ids = tokenizer.encode(full)[:max_seq]
    if len(ids) < 3:
        return None
    p_len = len(tokenizer.encode(prefix))
    inp = ids[:-1]
    tgt = ids[1:]
    mask = [1 if (i + 1) >= p_len else 0 for i in range(len(tgt))]
    return inp, tgt, mask


def collate(batch, max_seq, pad_id=0):
    inp = torch.full((len(batch), max_seq), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), max_seq), -100, dtype=torch.long)
    for i, (x, y, m) in enumerate(batch):
        x = x[:max_seq]
        y = y[:max_seq]
        m = m[:max_seq]
        inp[i, : len(x)] = torch.tensor(x)
        for j, (yy, mm) in enumerate(zip(y, m)):
            if mm:
                tgt[i, j] = yy
    return inp, tgt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_seq", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tokenizer = CharTokenizer.load(ROOT / "out" / "vocab.json")
    data = torch.load(ROOT / "out" / "pretrain.pt", map_location="cpu", weights_only=True)
    cfg = TinyConfig(**data["cfg"])
    model = TinyGPT(cfg)
    model.load_state_dict(data["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    rows = [json.loads(line) for line in open(ROOT / "data" / "sft.jsonl", encoding="utf-8")]
    samples = []
    for row in rows:
        s = make_sample(tokenizer, row["q"], row["a"], args.max_seq)
        if s:
            samples.append(s)
    print("sft samples:", len(samples))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    steps_per_epoch = max(1, len(samples) // args.batch)
    total = args.epochs * steps_per_epoch
    for step in range(total):
        batch = random.choices(samples, k=args.batch)
        x, y = collate(batch, args.max_seq)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == total - 1:
            prompt = "问：打印机连不上怎么办？答："
            ids = tokenizer.encode(prompt)
            x0 = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(x0, max_new=24, temperature=0.6, top_k=10)
            print("step %3d loss %.4f | %s" % (step, loss.item(), tokenizer.decode(out[0].tolist())[:44]))

    out_path = ROOT / "out" / "sft.pt"
    torch.save({"model": model.state_dict(), "cfg": data["cfg"]}, out_path)
    print("saved ->", out_path)


"""
进阶改造 Prompt
1. 把字符偏移 mask 换成“按 token 位置”的通用实现（主线 15 章 response-only loss）；
2. 增加 system/角色模板，验证模板一致性（主线 17 章）；
3. 增加 held-out 评测题，做 SFT 前后对比（主线 20 章）；
4. 尝试多轮对话数据（主线 21 章）；
5. 调 epochs/lr，观察“学会回答”与“过拟合背题”的分界。
"""

if __name__ == "__main__":
    main()

"""
规范字段补充（全局强制代码落地规范）

【层级】L2（工程训练：SFT 流程 + response-only mask 的完整实现）
【运行结果示例】（真实运行，CPU，60 epochs/6 样本）
step   0 loss 1.0870 | 问：打印机连不上怎么办？答：先检查电源和网络。
step  59 loss 0.1561 | 问：打印机连不上怎么办？答：先检查电源和网络，然后重启打印机，并提交工单。
saved -> .../out/sft.pt
（loss 1.09 -> 0.16 且能背出答案开头 = SFT 管线正确；语料过小长回复会串词属预期）
【高频报错 Top5】
1. 答案字符全 UNK：词表没覆盖；修复：重跑 make_corpus.py + pretrain.py；
2. loss 不降：mask 算错（把指令也算进去）；修复：核对字符偏移 p_len；
3. 生成复读“：”：用错预训练 ckpt（旧版标签 bug）；修复：重跑 pretrain.py --steps 400；
4. batch 内长度不一：本实现已 padding+ignore；自改数据时保持 ids 为 list of list；
5. 换系统提示词不生效：模板写死“问/答”；修复：模板参数化（进阶 Prompt 1）。
【工程改造方向】接真实指令数据（jsonl 同构）；加验证集与早停；接主线 17 章模板一致性。
"""

"""
规范字段补充 2（补齐缺失字段标记）

【输出解读】
- loss 从约 ln(vocab) 降到 0.2 以下 = 学会背答案；
- 生成样例中出现训练答案的开头 = response-only mask 生效；
- 若 loss 降但生成全是“：”复读，说明预训练 ckpt 是旧版（标签未移位）。
"""
