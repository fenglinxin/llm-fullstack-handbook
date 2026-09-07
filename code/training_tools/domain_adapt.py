# -*- coding: utf-8 -*-
"""
第26章 小样本与领域自适应微调（一键脚本）

【层级】L2（冷启动微调：极小领域数据（2-4 条）快速适配，观察过拟合边界）
【环境依赖】Python 3.10+；PyTorch 2.x；复用 minimind_style（sft.pt 缺失自动兜底）
【核心逻辑】从 sft.pt 出发，用 2 条领域 QA（第10章 domain KB 同款）微调 30 epochs；
生成领域 prompt 验证“是否学会了新表述”，同时打印原语料 prompt 判断遗忘程度。
【关键参数】--epochs 30（默认）；--lr 2e-4；--max_seq 64。
【避坑】领域样本越少越容易过拟合：对比新旧两类 prompt 的输出；
小样本微调后建议回到 ch19/20 做遗忘与评估闭环。
【运行结果示例】
$ python domain_adapt.py
epoch 30 loss 0.0739 | 答：重启打印机并联系管理员。
saved -> .../minimind_style/out/domain_adapt.pt
【高频报错 Top5】
1. sft.pt 缺失：自动兜底训练或先跑 train_sft.py；2. 新词 UNK：领域文本字符不在词表；
3. 过拟合（只会背领域句）：epochs 减半；4. 通用能力崩：lr 调 1e-4；5. 样本太少没效果：
领域数据要 >20 条（本脚本 2 条仅演示流程）。
【输出解读】领域 prompt 命中新表述 + 通用 prompt 仍正常 = 小样本适配成功。
【工程改造方向】接 ch10 领域数据流水线产物；few-shot 数量扫描（2/5/10/20）。
"""

import argparse
import pathlib
import random
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "minimind_style"
sys.path.insert(0, str(ROOT))

DOMAIN_QA = [
    ("打印机连不上怎么办？", "重启打印机并联系管理员。"),
    ("账号被锁怎么办？", "提交工单联系管理员。"),
]


def ensure():
    if not (ROOT / "out" / "sft.pt").exists():
        print("[bootstrap] sft.pt 缺失，自动训练兜底...")
        subprocess.run([sys.executable, "pretrain.py", "--steps", "150"], cwd=str(ROOT), check=True)
        subprocess.run([sys.executable, "train_sft.py", "--epochs", "20"], cwd=str(ROOT), check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    ensure()

    from tokenizer import CharTokenizer
    from model import TinyConfig, TinyGPT
    import train_sft as T
    import torch

    tok = CharTokenizer.load(ROOT / "out" / "vocab.json")
    d = torch.load(str(ROOT / "out" / "sft.pt"), map_location="cpu", weights_only=True)
    m = TinyGPT(TinyConfig(**d["cfg"]))
    m.load_state_dict(d["model"])
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    samples = []
    for q, a in DOMAIN_QA:
        s = T.make_sample(tok, q, a, 64)
        if s:
            samples.append(s)
    for ep in range(args.epochs):
        x, y = T.collate(random.choices(samples, k=len(samples)), 64)
        loss = ce(m(x).reshape(-1, d["cfg"]["vocab_size"]), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            gen = m.generate(torch.tensor([tok.encode("问：" + DOMAIN_QA[0][0] + "答：")]),
                             max_new=12, temperature=0.6, top_k=10)
            print("epoch %d loss %.4f | %s" % (ep + 1, loss.item(),
                                               tok.decode(gen[0].tolist())[-14:]))
    out = ROOT / "out" / "domain_adapt.pt"
    torch.save({"model": m.state_dict(), "cfg": d["cfg"]}, out)
    print("saved ->", out)


if __name__ == "__main__":
    main()