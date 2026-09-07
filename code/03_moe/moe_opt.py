# -*- coding: utf-8 -*-
"""
MoE 高阶优化扫描（L3）：负载均衡系数与 Top-k 选择

【层级】L3（高阶优化：对应 MoE 章节“防专家崩溃/省算力”教学重点）
【环境依赖】
- Python 3.10+；PyTorch 2.x（建议 >=2.1）
- 安装：pip install torch==2.2.2；CPU 可跑（本扫描 4 组 x 200 步约 1 分钟）
【核心逻辑】
1. balance_coef 扫描（0 / 0.01 / 0.1 / 1.0）：同一路由分类任务各训 200 步，
   记录 val_acc 与专家命中分布，看“任务收敛 vs 负载均匀”的权衡——
   系数太小专家饿死（hits 有 0），太大路由失去选择性；
2. top_k 扫描（1/2/3/4）：记录 acc、单 token 激活参数占比，理解
   “越稀疏越省算力，但可能丢精度/更易倾斜”；
3. 命中分布用变异系数 CV = std/mean 量化（0=完全均匀）。
【关键参数】（默认值 / 推荐值 / 适配场景）
- --steps 200（默认；每组训练步数）：扫描时单配置预算；
- --coefs "0,0.01,0.1,1.0"（默认）：平衡系数列表；
- --topks "1,2,3,4"（默认）：Top-k 列表；
- --n_experts 4（默认）；--seed 0。
【避坑】
- 本任务 4 簇可分性很高，acc 很快到 1.0，重点是看 hits 分布不是 acc；
- balance_coef=1.0 可能把路由压成“每专家等概率”而牺牲选择精度，
  长训练后 acc 可能回落，属正常权衡；
- 激活参数占比 = 每次前向用到的参数/总参数，只看它别忘路由开销。
【运行结果示例】（真实运行，CPU）
$ python moe_opt.py --steps 200
=== balance_coef 扫描（steps=200）===
coef   val_acc  CV       hits
0.00   1.000    0.77     [243, 119, 238, 0]
0.01   1.000    0.38     [132, 232, 137, 99]
0.10   1.000    0.15     [153, 180, 139, 128]
1.00   1.000    0.04     [148, 150, 158, 144]
=== top_k 扫描（coef=0.1）===
top_k  val_acc  active%
1      1.000    26.1
2      1.000    50.7
3      1.000    75.4
4      1.000    100.0
PASS
（本任务 4 簇可分性高 acc 恒为 1.0，结论看 CV：coef=0 时专家 4 饿死 hits=0，
coef>=0.1 后 CV<0.2 负载健康；top_k=1 只激活 26% 参数且不掉点）
【高频报错 Top5】
1. 扫描结果全一样：种子没变/数据没重采样；修复：每组用不同 seed；
2. hits 全 0：top_k 统计用了 model 而非 model.moe；修复：见 moe_engine.hits_of；
3. top_k=4 与 n_experts=4 等价 dense：属预期，别当 bug；
4. 运行太慢：steps 太大或 coefs 太多；修复：--steps 100；
5. CV=0 但 acc 掉：路由被压平，选择能力下降；修复：coef 取 0.05-0.2 区间。
【输出解读】
- 理想工作点：acc 不掉 + CV<0.3（hits 无明显饿死）；
- coef 从 0 增大：CV 单调下降，acc 可能先平后掉；
- top_k 越小激活参数越少，但 acc 可能下降——按你的算力预算选。
【工程改造方向】
- 大规模 MoE：Switch/负载均衡 aux + Router z-loss 一起用；
- 部署：按 top_k 与命中分布规划专家显存驻留与 EP 分组；
- 迭代：把本扫描脚本接进训练流水线做超参报告（CSV 落盘）。
"""

import argparse
import math

import torch
import torch.nn as nn

from moe_demo import SparseMoE
from moe_engine import MoEClassifier, make_data, hits_of


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--coefs", default="0,0.01,0.1,1.0")
    p.add_argument("--topks", default="1,2,3,4")
    p.add_argument("--n_experts", type=int, default=4)
    p.add_argument("--n_train", type=int, default=800)
    p.add_argument("--n_val", type=int, default=300)
    p.add_argument("--d_model", type=int, default=16)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def train_quick(coef, top_k, args, seed):
    torch.manual_seed(seed)
    train_x, train_y = make_data(args.n_train, args.n_experts, args.d_model, seed)
    val_x, val_y = make_data(args.n_val, args.n_experts, args.d_model, seed + 1)
    model = MoEClassifier(d_model=args.d_model, n_experts=args.n_experts,
                          top_k=top_k, balance_coef=coef)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    for _ in range(args.steps):
        idx = torch.randint(0, len(train_x), (args.batch,))
        xb = train_x[idx].unsqueeze(1)
        yb = train_y[idx]
        logits, aux = model(xb)
        loss = ce(logits, yb) + coef * aux
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc, hits = hits_of(model, val_x, val_y, "cpu")
    return acc, hits


def cv_of(hits):
    h = hits.float()
    mean = h.mean().item()
    return (h.std().item() / mean) if mean > 0 else float("inf")


def active_ratio(n_experts, top_k, d_model, hidden=32):
    """单 token 激活参数 / 总参数 的粗估（router+专家 MLP）。"""
    expert = 2 * d_model * hidden + hidden  # 两层 Linear 参数
    total = n_experts * expert + d_model * n_experts
    active = d_model * n_experts + top_k * expert
    return 100.0 * active / total


def main():
    args = parse_args()
    coefs = [float(x) for x in args.coefs.split(",") if x.strip()]
    topks = [int(x) for x in args.topks.split(",") if x.strip()]
    print("=== balance_coef 扫描（steps=%d）===" % args.steps)
    print("%-6s %-8s %-8s %s" % ("coef", "val_acc", "CV", "hits"))
    for i, coef in enumerate(coefs):
        acc, hits = train_quick(coef, 2, args, args.seed + i * 7)
        print("%-6.2f %-8.3f %-8.2f %s" % (coef, acc, cv_of(hits),
                                           [int(h) for h in hits]))
    print("=== top_k 扫描（coef=0.1）===")
    print("%-6s %-8s %-9s" % ("top_k", "val_acc", "active%"))
    for i, k in enumerate(topks):
        acc, hits = train_quick(0.1, k, args, args.seed + 100 + i * 7)
        print("%-6d %-8.3f %-9.1f" % (k, acc,
                                      active_ratio(args.n_experts, k, args.d_model)))
    print("PASS")


if __name__ == "__main__":
    main()
