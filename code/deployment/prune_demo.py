# -*- coding: utf-8 -*-
"""
第32章 模型剪枝与蒸馏压缩：幅度剪枝 + 微调恢复（CPU 一键脚本）

【层级】L3（压缩优化：magnitude pruning + 剪后微调，可换任意 nn.Linear 网络）
【环境依赖】Python 3.10+；PyTorch 2.x
【核心逻辑】正弦 MLP 训练 -> 按幅度剪掉 50% 权重 -> 冻结 mask 微调 200 轮；
报告 val_mse：原模型 vs 剪后(未微调) vs 剪后(微调)，体现“剪+恢复”流程。
【关键参数】--ratio 0.5（默认）；--ft_epochs 200（默认）；--seed 0。
【避坑】幅度剪枝假设小权重不重要（近似）；结构剪枝（通道/头）才有真实算力收益；
剪后必须微调，否则精度损失大。
【运行结果示例】
$ python prune_demo.py
val_mse: orig=0.00063 pruned=0.24682 pruned_ft=0.01754 | 恢复率=93.1%
【高频报错 Top5】
1. mask 没冻结导致剪掉的权重复活：mask 在 forward 里应用且不更新；
2. 精度崩：ratio 太高（>0.8）或没微调；3. 结构剪枝才能省显存：本脚本是权重剪枝；
4. 微调 lr 太大破坏剩余权重：lr 1e-3；5. 只报密度不报精度。
【输出解读】ft 后 val_mse 接近原模型 = 剪枝流程可用。
【工程改造方向】接结构化剪枝（行/通道）+ 蒸馏组合压缩（配合 distill_demo）。
"""

import argparse

import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--ft_epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def data(seed):
    torch.manual_seed(seed)
    x = torch.rand(200, 1) * 4
    return x, torch.sin(x) + 0.02 * torch.randn(200, 1)


def main():
    args = parse_args()
    x, y = data(args.seed)
    vx, vy = data(args.seed + 1)
    net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    mse = nn.MSELoss()
    for _ in range(600):
        loss = mse(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        orig = mse(net(vx), vy).item()
    # 剪枝
    masks = {}
    with torch.no_grad():
        for name, p in net.named_parameters():
            if p.dim() >= 2:
                th = torch.quantile(p.abs().flatten(), args.ratio)
                masks[name] = p.abs() >= th
                p.mul_(masks[name].float())
    with torch.no_grad():
        pruned = mse(net(vx), vy).item()
    # 冻结 mask 微调
    opt2 = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(args.ft_epochs):
        loss = mse(net(x), y)
        opt2.zero_grad(); loss.backward(); opt2.step()
        with torch.no_grad():
            for name, p in net.named_parameters():
                if name in masks:
                    p.mul_(masks[name].float())
    with torch.no_grad():
        ft = mse(net(vx), vy).item()
    print("val_mse: orig=%.5f pruned=%.5f pruned_ft=%.5f | 恢复率=%.1f%%"
          % (orig, pruned, ft, 100 * max(0, 1 - (ft - orig) / max(pruned - orig, 1e-9))))


if __name__ == "__main__":
    main()
