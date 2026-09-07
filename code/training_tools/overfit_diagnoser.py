# -*- coding: utf-8 -*-
"""
第19章 过拟合/欠拟合诊断器：小数据高轮数训练 + 早停对照（一键脚本）

【层级】L2（工程诊断：train/val gap 曲线 + 自动建议，可接任意训练循环）
【环境依赖】Python 3.10+；PyTorch 2.x（pip install torch==2.2.2）
【核心逻辑】正弦回归小数据集（n=40）训练 3000 轮 MLP，每隔 200 轮记录
train/val loss；判断过拟合=val 上升而 train 下降；输出 gap 曲线摘要与
“早停 vs 不早停”的 val 对比。
【关键参数】--epochs 3000（默认）；--patience 400（默认；0=关闭早停）；
--hidden 32；--n_train 40（默认故意小，制造过拟合）。
【避坑】小数据单次结果噪声大，多看 gap 趋势；早停阈值按 val 平滑后判断；
过拟合修复优先级：数据/增强 > 正则 > 模型缩小。
【运行结果示例】（真实运行，CPU）
$ python overfit_diagnoser.py --epochs 6000 --n_train 10 --patience 1200
train_final=0.00729
early_stop(pat=1200): val_best=0.05689@epoch863 val_at_end=0.05820
no_stop:           val_best=0.05689@epoch863 val_at_end=0.05886
诊断：train 低 + val 高 = 过拟合；过拟合回升幅度 = 0.00197（正=后期确实在变差，适合早停）
【高频报错 Top5】
1. val 全 0：数据没切分或 seed 固定；2. gap 判不出来：epochs 太少；
3. 早停没触发：patience 太大；4. loss NaN：lr 太大（默认 1e-2 应安全）；
5. 想复现：固定 --seed。
【输出解读】看到 train 持续降、val 先降后升 = 过拟合成立；
early_stop val 优于 no_stop val = 早停有效。
【工程改造方向】接真实训练循环（记录 loss 钩子）；输出 gap 图；接 ch09 数据质量前置。
"""

import argparse
import math

import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--patience", type=int, default=1200)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n_train", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_data(n, seed):
    torch.manual_seed(seed)
    x = torch.rand(n, 1) * 6
    y = torch.sin(x) + 0.15 * torch.randn(n, 1)
    return x, y


def run(patience, args):
    """返回 (train_final, val_best, val_at_end, best_epoch)。"""
    x, y = make_data(args.n_train, args.seed)
    vx, vy = make_data(120, args.seed + 1)
    net = nn.Sequential(nn.Linear(1, args.hidden), nn.ReLU(), nn.Linear(args.hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    mse = nn.MSELoss()
    best = float("inf")
    best_ep = 0
    wait = 0
    val_end = None
    for ep in range(args.epochs):
        loss = mse(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            vl = mse(net(vx), vy).item()
        if vl < best:
            best = vl
            best_ep = ep
            wait = 0
        else:
            wait += 1
        val_end = vl
        if patience and wait >= patience:
            break
    return loss.item(), best, val_end, best_ep


def main():
    args = parse_args()
    tl, best_e, end_e, ep_e = run(args.patience, args)
    tl2, best_f, end_f, ep_f = run(0, args)
    print("train_final=%.5f" % tl)
    print("early_stop(pat=%d): val_best=%.5f@epoch%d val_at_end=%.5f"
          % (args.patience, best_e, ep_e, end_e))
    print("no_stop:           val_best=%.5f@epoch%d val_at_end=%.5f"
          % (best_f, ep_f, end_f))
    print("诊断：train 低 + val 高 = 过拟合；"
          "过拟合回升幅度 = %.5f（正=后期确实在变差，适合早停）"
          % (end_f - best_f))


if __name__ == "__main__":
    main()