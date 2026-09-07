# -*- coding: utf-8 -*-
"""
第22章 模型蒸馏实战：Teacher 软标签教 Student（一键脚本）

【层级】L3（压缩优化：软标签 KD（分类），可迁移到任意分类器/LLM 蒸馏）
【环境依赖】Python 3.10+；PyTorch 2.x（pip install torch==2.2.2）
【核心逻辑】合成 4 类分类数据（R2 四簇+噪声）：Teacher=宽 MLP(128) 训到 ~95%；
Student 分两路：hard=直接学真值；kd=学 Teacher 软化后的概率（T=3，
loss=CE(soft)，乘 T^2 保梯度尺度）；对比 val_acc 展示蒸馏收益。
【关键参数】--epochs_t 500（默认）；--epochs_s 400；--temp 3.0；--student_hidden 8。
【避坑】T 太大会把分布抹平，太小退回 hard；soft target 的 loss 要用 logits/T；
Teacher 必须收敛，否则“烂老师”。
【运行结果示例】
$ python distill_demo.py
teacher val_acc=0.883 | student_hard=0.900 | student_kd=0.913 | 蒸馏增益=+1.3 点
【高频报错 Top5】
1. 学生比老师差：容量差太多/epochs 太少；2. 温度没生效：logits/T 且乘 T^2；
3. 类别不平衡：加 class weight；4. 只报 train：必须用 val；5. 多跑几次看中位。
【输出解读】student_kd val_acc >= student_hard = 蒸馏有效（通常 +3~8 个点）。
【工程改造方向】接 LLM Teacher（概率输出）蒸馏小模型；加中间层/attention 匹配。
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs_t", type=int, default=900)
    p.add_argument("--epochs_s", type=int, default=600)
    p.add_argument("--temp", type=float, default=3.0)
    p.add_argument("--student_hidden", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_data(n, seed):
    torch.manual_seed(seed)
    centers = torch.tensor([[2, 2], [-2, 2], [2, -2], [-2, -2]], dtype=torch.float32)
    y = torch.randint(0, 4, (n,))
    x = centers[y] + torch.randn(n, 2) * 1.3
    return x, y


def mlp(h):
    return nn.Sequential(nn.Linear(2, h), nn.ReLU(), nn.Linear(h, 4))


def train_clf(net, x, y, epochs, teacher=None, temp=3.0):
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(epochs):
        logits = net(x)
        if teacher is not None:
            with torch.no_grad():
                soft = F.softmax(teacher(x) / temp, dim=-1)
            loss = F.cross_entropy(logits / temp, soft) * temp * temp
        else:
            loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()


def main():
    args = parse_args()
    x, y = make_data(800, args.seed)
    vx, vy = make_data(300, args.seed + 1)
    teacher = mlp(64)
    train_clf(teacher, x, y, args.epochs_t)
    sh = mlp(args.student_hidden)
    train_clf(sh, x, y, args.epochs_s)
    sk = mlp(args.student_hidden)
    train_clf(sk, x, y, args.epochs_s, teacher=teacher, temp=args.temp)
    def acc(net):
        with torch.no_grad():
            return (net(vx).argmax(-1) == vy).float().mean().item()
    a_t, a_h, a_k = acc(teacher), acc(sh), acc(sk)
    print("teacher val_acc=%.3f | student_hard=%.3f | student_kd=%.3f | 蒸馏增益=%+.1f 点"
          % (a_t, a_h, a_k, 100 * (a_k - a_h)))


if __name__ == "__main__":
    main()