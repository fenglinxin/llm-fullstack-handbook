# -*- coding: utf-8 -*-
"""
第23章 奖励模型训练：Bradley-Terry 排序最小实现（一键脚本）

【层级】L2（奖励模型训练：pairwise logistic，可接 DPO/PPO 做偏好信号）
【环境依赖】Python 3.10+；PyTorch 2.x
【核心逻辑】合成偏好对：每条 prompt 配 chosen（高质量特征=长度适中+关键词多）
与 rejected（短+无关键词）；RM=Linear(features)->score；损失= -log sigmoid
(s_chosen - s_rejected)；评测 pair 准确率与防 Reward Hacking 的分布检查。
【关键参数】--n_pair 200（默认）；--epochs 100（默认）；--lr 1e-2。
【避坑】合成数据特征要“可学但带噪”，否则准确率恒 1 看不出训练过程；
真实 RM 需防 hacking：评测奖励分布与人类一致性，而不是只看准确率。
【运行结果示例】
$ python rm_train.py
epoch 25 loss 0.4203
epoch 50 loss 0.2829
pair_acc=1.000 mean_reward chosen=1.629 rejected=0.318
【高频报错 Top5】
1. sigmoid 里差值为 0 恒 0.5：数据没区分度；2. 只报准确率：要加奖励分布统计；
3. 特征泄漏：chosen/rejected 特征在标注后计算；4. lr 太大 loss NaN；
5. 换真实偏好数据：字段名要与 dpo_data.py 对齐（chosen/rejected）。
【输出解读】pair_acc>0.9 且两种回答平均奖励差距为正 = RM 学到偏好。
【工程改造方向】接 LLM RM（最后 token 的隐藏层接 score head）；接 PPO 当 reward 源。
"""

import argparse
import random

import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_pair", type=int, default=200)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_pairs(n, seed):
    random.seed(seed)
    feats_c, feats_r = [], []
    for _ in range(n):
        kw = random.randint(3, 6)
        ln = random.randint(20, 40)
        feats_c.append([ln / 40.0, kw / 6.0, 1.0])
        feats_r.append([random.randint(4, 14) / 40.0, random.randint(0, 1) / 6.0, 0.0])
    return (torch.tensor(feats_c, dtype=torch.float32),
            torch.tensor(feats_r, dtype=torch.float32))


def main():
    args = parse_args()
    fc, fr = make_pairs(args.n_pair, args.seed)
    rm = nn.Linear(3, 1)
    opt = torch.optim.Adam(rm.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        sc, sr = rm(fc), rm(fr)
        loss = -torch.log(torch.sigmoid(sc - sr) + 1e-9).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 25 == 0:
            print("epoch %d loss %.4f" % (ep + 1, loss.item()))
    with torch.no_grad():
        acc = ((rm(fc) - rm(fr)) > 0).float().mean().item()
        print("pair_acc=%.3f mean_reward chosen=%.3f rejected=%.3f"
              % (acc, rm(fc).mean().item(), rm(fr).mean().item()))


if __name__ == "__main__":
    main()
