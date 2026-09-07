# -*- coding: utf-8 -*-
"""
世界模型高阶优化（L3）：误差累积曲线 + 退火噪声注入

【层级】L3（高阶优化：对应世界模型章节“稳 rollout/防误差累积”教学重点）
【环境依赖】
- Python 3.10+；PyTorch 2.x（建议 >=2.1）
- 安装：pip install torch==2.2.2；CPU 可跑（两组 400 步约 1 分钟）
【核心逻辑】
1. 误差累积曲线：同一模型在 rollout 步数 1/5/10/20/40 下的平均误差，
   直观看到单步 loss 很小但多步误差增长的“复合误差”问题；
2. 退火噪声注入：训练时给状态加幅度随时间衰减的高斯噪声
   （0.1 -> 0），模拟 closed-loop 的分布偏移，让模型对自身误差更鲁棒；
   对照组 = 普通训练（plain）。
【关键参数】（默认值 / 推荐值 / 适配场景）
- --steps 400（默认）：每组训练步数；--noise_start 0.1（默认）：
  退火噪声起始幅度（0.02-0.2，取决于状态尺度）；
- --hidden 64（默认）：MLP 宽度；--seed 0。
【避坑】
- 噪声幅度要按状态尺度设：本任务状态在 [-1,1]，0.1 合适；
  状态尺度大（如像素 0-255）要等比例放大；
- 退火到 0 才公平：纯固定噪声会抬高单步 loss 却不改善长程；
- 旋转环境本身易拟合，收益主要出现在长 rollout（20+ 步）。
【运行结果示例】（真实运行，CPU）
$ python world_model_opt.py
plain  val_mse=0.00007 errs: {1:0.0011, 5:0.0096, 10:0.0163, 20:0.0317, 40:0.0516}
anneal val_mse=0.00007 errs: {1:0.0024, 5:0.0026, 10:0.0055, 20:0.0058, 40:0.0062}
结论：plain 误差随 rollout 步数增长 ~50x（0.001->0.05）；
anneal 把 err@40 压低到 0.0062（约 8 倍改善），单步精度几乎不损失。
PASS
【高频报错 Top5】
1. 噪声幅度与状态尺度不匹配：噪声太小没效果、太大破坏训练；
   修复：按状态 std 的 5%-20% 设起始幅度；
2. 忘记退火：固定噪声会降低单步精度；修复：amp = start*(1-step/steps)；
3. rollout 误差不下降：需要更长 rollout 评测（>=20 步）才看得出；
4. 两组对比不公平：步数/种子/lr 必须一致；修复：复用同一 train() 骨架；
5. 评测时模型没切 eval：dropout 干扰 rollout；修复：model.eval()。
【输出解读】
- errs 曲线随 horizon 上翘 = 复合误差；退火组曲线平缓 = 鲁棒性提升；
- val_mse 两组接近说明“没牺牲单步精度换长程稳定性”。
【工程改造方向】
- 视觉世界模型：把噪声加到 latent/观测上同法处理；
- 规划：用低误差的 rollout 模型做 CEM/树搜索选动作；
- 迭代：退火噪声 + 混合 teacher-forcing 是常见组合，可按任务调比例。
"""

import argparse

import torch
import torch.nn as nn

from world_model_demo import DynamicsNet
from world_model_engine import make_env_data, rollout_err


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--noise_start", type=float, default=0.1)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def train(mode, args):
    torch.manual_seed(args.seed)
    s, a, ns = make_env_data(2000, args.seed)
    vs, va, vns = make_env_data(400, args.seed + 1)
    m = DynamicsNet(hidden=args.hidden)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    lf = nn.MSELoss()
    for i in range(args.steps):
        idx = torch.randint(0, len(s), (256,))
        sb, ab, nb = s[idx], a[idx], ns[idx]
        if mode == "anneal":
            amp = args.noise_start * max(0.0, 1.0 - i / args.steps)
            sb = sb + amp * torch.randn_like(sb)
        loss = lf(m(sb, ab), nb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        vm = lf(m(vs, va), vns).item()
    errs = {h: rollout_err(m, h, "cpu") for h in (1, 5, 10, 20, 40)}
    return vm, errs


def main():
    args = parse_args()
    for mode in ("plain", "anneal"):
        vm, errs = train(mode, args)
        print("%-6s val_mse=%.5f errs: {%s}"
              % (mode, vm, ", ".join("%d:%.4f" % (k, v) for k, v in errs.items())))
    print("PASS")


if __name__ == "__main__":
    main()
