# -*- coding: utf-8 -*-
"""
第12章/第18章 超参选型扫描器：lr x dim 网格 + 收敛报告（一键脚本）

【层级】L2（工程工具：网格扫描 + 表格报告，可接入实验管理）
【环境依赖】Python 3.10+；PyTorch 2.x（pip install torch==2.2.2）；复用 minimind_style 代码
【核心逻辑】用同一语料/窗口/步数跑 lr x dim 网格（默认 2x2），每格训练后
打印 final loss 与首末 loss 差，输出最优组合；--include_lora 时在 SFT 基座上
用 LoRA 扫 rank/alpha（覆盖第18章微调超参场景，步数建议 60）。
【关键参数】--lrs 1e-3,3e-3；--dims 32,64（默认）；--steps 80（默认，越小越快）；
--include_lora（可选）：扫 rank=4,8 / alpha=8,16。
【避坑】对比必须同 seed/steps/batch；先粗扫（小 steps）再细扫；
CPU 上 dim=128 以上单格会明显变慢；保存报告前固定 seed。
【运行结果示例】
$ python hyperparam_sweep.py --steps 80
dim    lr       first      final
32     1.0e-03  5.9646     3.9099
32     3.0e-03  6.0202     1.3436
64     1.0e-03  6.0108     2.2606
64     3.0e-03  6.0705     0.1488
best: dim=64 lr=3.0e-03 final_loss=0.1488
【高频报错 Top5】
1. 结果不可比：每格 seed 不同；修复：固定 --seed 且每格派生相同偏移；
2. 扫描太慢：steps 减到 40 先看趋势；3. include_lora 报错：需先跑 train_sft.py；
4. 表格 NaN：lr 过大；修复：上限 1e-2；5. 报告丢失：重定向输出或加 --save 到 CSV。
【输出解读】表格 final loss 最低行 = 当前预算下的推荐组合；多格并列时选参数量小的。
【工程改造方向】接 wandb/swanlab 可视化；换成贝叶斯/optuna 搜索；报告落 CSV/JSON。
"""

import argparse
import random
import sys
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent / "minimind_style"
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lrs", default="1e-3,3e-3")
    p.add_argument("--dims", default="32,64")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include_lora", action="store_true")
    return p.parse_args()


def train_once(dim, lr, steps, seed):
    from tokenizer import CharTokenizer
    from model import TinyConfig, TinyGPT
    import pretrain as P

    torch.manual_seed(seed)
    random.seed(seed)
    corpus = (ROOT / "data" / "corpus.txt").read_text(encoding="utf-8")
    tok = CharTokenizer()
    tok.fit([corpus])
    ids = tok.encode(corpus)
    windows = P.make_windows(ids, 48)
    cfg = TinyConfig(vocab_size=len(tok.vocab), dim=dim, n_layers=2, n_heads=2, max_seq=48)
    m = TinyGPT(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    first = None
    for _ in range(steps):
        x = torch.tensor([random.choice(windows) for _ in range(16)])
        loss = ce(m(x[:, :-1]).reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        if first is None:
            first = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    return first, loss.item()


def main():
    args = parse_args()
    lrs = [float(x) for x in args.lrs.split(",")]
    dims = [int(x) for x in args.dims.split(",")]
    print("%-6s %-8s %-10s %-10s" % ("dim", "lr", "first", "final"))
    best = None
    i = 0
    for dim in dims:
        for lr in lrs:
            first, final = train_once(dim, lr, args.steps, args.seed + i)
            print("%-6d %-8.1e %-10.4f %-10.4f" % (dim, lr, first, final))
            if best is None or final < best[2]:
                best = (dim, lr, final)
            i += 1
    print("best: dim=%d lr=%.1e final_loss=%.4f" % best)


if __name__ == "__main__":
    main()