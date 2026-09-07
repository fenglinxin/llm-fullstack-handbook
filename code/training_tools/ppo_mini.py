# -*- coding: utf-8 -*-
"""
第24章 RLHF/PPO 工程落地（极简版）：Actor-Critic + 优势学习（一键脚本）

【层级】L3（强化学习最小实现：策略梯度 + critic 基线，可扩展为完整 PPO）
【环境依赖】Python 3.10+；PyTorch 2.x
【核心逻辑】二臂赌博机环境：动作 0 奖励 0.2、动作 1 奖励 1.0（带噪声）；
策略=Softmax(Linear(1,2))，critic=Linear(1,1) 估状态值；
每 50 幕打印平均奖励，观察策略从随机 0.5 提升到 ~0.95。
【关键参数】--episodes 2000（默认）；--lr 1e-2；--gamma 0.9（默认）。
【避坑】advantage 用 r - V(s) 才能降方差；critic 与 policy 共用状态要分网络；
完整 PPO 还有 ratio clip/KL/多 epoch，本文件演示核心梯度路径。
【运行结果示例】
$ python ppo_mini.py
episode 250 avg_reward_last250=0.981
episode 500 avg_reward_last250=1.000
episode 1500 avg_reward_last250=1.000
【高频报错 Top5】
1. 奖励不升：advantage 没 detach 或 lr 太小；2. critic 发散：MSE 权重太大；
3. 恒选一个动作：温度/探索没了；4. 想直接上大模型：参考 minimind_style/train_grpo.py；
5. 随机种子不同结果波动：固定 --seed。
【输出解读】滑动平均奖励从 0.5 附近升到 0.9+ = 策略学到好动作。
【工程改造方向】加 ratio clip（PPO 正式版）；接 RM 输出当奖励；接语言策略模型。
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    policy = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 2))
    critic = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()),
                           lr=args.lr)
    state = torch.zeros(1, 1)
    buf = []
    for ep in range(args.episodes):
        logits = policy(state)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        r = 1.0 if a.item() == 1 else 0.2
        v = critic(state)
        adv = (r - v).detach()
        loss_p = -dist.log_prob(a) * adv
        loss_c = F.mse_loss(v, torch.tensor([[r]]))
        (loss_p + 0.5 * loss_c).backward()
        opt.step()
        buf.append(r)
        if (ep + 1) % 250 == 0:
            print("episode %d avg_reward_last250=%.3f" % (ep + 1,
                  sum(buf[-250:]) / len(buf[-250:])))


if __name__ == "__main__":
    main()
