# -*- coding: utf-8 -*-
"""
第21章 多轮对话微调专项：多轮历史拼接 + response-only loss（一键脚本）

【层级】L2（专项训练：多轮数据构造 + 微调 + 双轮生成验证）
【环境依赖】Python 3.10+；PyTorch 2.x；复用 minimind_style（out/sft.pt 缺失自动兜底）
【核心逻辑】把多轮样本拼成单一文本：问：q1答：a1问：q2答：a2；loss 只算最后一轮
回答 a2（字符偏移 mask），让模型学会“结合历史上下文回答”。
【关键参数】--epochs 40（默认）；--lr 2e-4；--max_seq 96（默认；多轮文本更长）；
--ckpt out/sft.pt（继续微调基座）。
【避坑】多轮文本超过 max_seq 会被截尾，长历史要按窗口截断；mask 偏移要按
“最后一个答：”计算，别把历史回答也当目标；同义多轮数据要多样化。
【运行结果示例】
$ python multiturn_sft.py
epoch 10 loss 0.4111 | 答：提交工单元是让机，再重置网络
epoch 20 loss 0.0678 | 答：提交工单联系管理员。
epoch 30 loss 0.0250 | 答：提交工单联系管理员。
saved -> .../out/multiturn.pt
【高频报错 Top5】
1. 生成时只回最后一轮：模板分隔符要固定“问：/答：”；2. 历史丢失：max_seq 太小；
3. loss 高：只 mask 最后回答（本实现已按偏移）；4. 中文乱码：utf-8；
5. 与单轮效果无差异：问题要有“上下文依赖”（第二问引用第一轮）。
【输出解读】loss 下降 + 双轮 prompt 生成含第二问答案 = 学会用历史。
【工程改造方向】接真实多轮对话语料（ch16 标注规范）；加角色/系统模板（ch17）。
"""

import argparse
import pathlib
import random
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "minimind_style"
sys.path.insert(0, str(ROOT))

TURNS = [
    ("无线网络连不上怎么办？", "先确认连接公司无线网络，再重置网络。",
     "重置后还连不上？", "提交工单联系管理员。"),
    ("账号被锁了怎么办？", "等待半小时后重试。",
     "管理员在哪找？", "提交工单联系管理员。"),
]


def ensure_ckpt():
    if (ROOT / "out" / "sft.pt").exists():
        return
    print("[bootstrap] 缺少 sft.pt，自动跑 pretrain 150 步 + SFT 20 epochs...")
    subprocess.run([sys.executable, "pretrain.py", "--steps", "150"], cwd=str(ROOT), check=True)
    subprocess.run([sys.executable, "train_sft.py", "--epochs", "20"], cwd=str(ROOT), check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_seq", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    ensure_ckpt()

    from tokenizer import CharTokenizer
    from model import TinyConfig, TinyGPT
    import torch
    import train_sft as T

    tok = CharTokenizer.load(ROOT / "out" / "vocab.json")
    d = torch.load(str(ROOT / "out" / "sft.pt"), map_location="cpu", weights_only=True)
    cfg = TinyConfig(**d["cfg"])
    m = TinyGPT(cfg)
    m.load_state_dict(d["model"])
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    samples = []
    for q1, a1, q2, a2 in TURNS:
        prefix = "问：" + q1 + "答：" + a1 + "问：" + q2 + "答："
        s = T.make_sample(tok, q1, "", args.max_seq)  # 占位结构复用
        # 手写多轮样本：全文本 ids + 只对最后回答做 mask
        full_ids = tok.encode(prefix + a2)[: args.max_seq]
        p2 = len(tok.encode(prefix))
        inp = full_ids[:-1]
        tgt = full_ids[1:]
        mask = [1 if i + 1 >= p2 else 0 for i in range(len(tgt))]
        samples.append((inp, tgt, mask))

    for ep in range(args.epochs):
        batch = random.choices(samples, k=len(samples))
        x, y = T.collate(batch, args.max_seq)
        loss = ce(m(x).reshape(-1, cfg.vocab_size), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            prompt = "问：无线网络连不上怎么办？答：先确认连接公司无线网络，再重置网络。问：重置之后还是连不上呢？答："
            gen = m.generate(torch.tensor([tok.encode(prompt)]), max_new=14,
                             temperature=0.6, top_k=10)
            print("epoch %d loss %.4f | %s" % (ep + 1, loss.item(),
                                               tok.decode(gen[0].tolist())[-16:]))
    out = ROOT / "out" / "multiturn.pt"
    torch.save({"model": m.state_dict(), "cfg": d["cfg"]}, out)
    print("saved ->", out)


if __name__ == "__main__":
    main()