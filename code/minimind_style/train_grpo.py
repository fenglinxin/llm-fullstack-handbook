# -*- coding: utf-8 -*-
"""
TinyGPT 微型 GRPO（P3：组相对策略优化，规则奖励 + 无 Critic）

【定位】展示 RLHF 中的 GRPO 核心思想：同一问题采样 K 个回答，
用规则奖励打分，组内归一化得到 advantage，再做策略梯度更新。
工业版（MiniMind/TRL）还会加 ratio clip 与对参考模型的 KL 惩罚，
本文件只保留“组相对优势”这一 GRPO 区别于 PPO 的关键。

【环境依赖】Python 3.10+, PyTorch 2.x；CPU 1-3 分钟。

【前置】先跑 pretrain.py / train_sft.py 生成 out/sft.pt、out/vocab.json。

【核心逻辑】
1. 对一个 question 采样 K 个回答（temperature 采样）；
2. 规则奖励 = 生成回答与标准答案的最长公共前缀占比（0~1）；
3. advantage = (reward - mean) / (std + eps)，组内归一化；
4. loss = -mean(advantage * 回答部分 logp 之和)，反向更新策略。

【关键参数】--steps 80 --group_size 5 --lr 1e-4 --temp 0.9。

【避坑】
1. 组内奖励全相同（std=0）时 advantage 无意义，本实现直接跳过该步；
2. 采样必须在当前策略上做，且 logp 只统计“回答部分”，别把 prompt 算进去；
3. 玩具模型已过拟合标准答案，采样奖励普遍偏高，改进幅度有限属正常；
4. 真实 GRPO 需要 ratio clip 防更新过猛、KL 惩罚防偏离参考模型。

【输出解读】mean_reward 代表组内平均规则得分；
mean_adv>0 说明策略在向高分回答倾斜；loss 上升/下降都不是唯一判据，
最终看固定评测集上的规则奖励是否提升。
"""

import argparse
import json
import pathlib
import random

import torch
import torch.nn.functional as F

from model import TinyConfig, TinyGPT
from tokenizer import CharTokenizer

ROOT = pathlib.Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="out/sft.pt")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--group_size", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--temp", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=30)
    p.add_argument("--max_new", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="out/grpo.pt")
    return p.parse_args()


def rule_reward(answer, generated):
    """最长公共前缀占比：生成与标准答案从第一个字逐字比对。"""
    gen = generated
    if "答：" in gen:
        gen = gen.split("答：", 1)[1]
    n = 0
    for a, b in zip(answer, gen):
        if a != b:
            break
        n += 1
    return n / max(len(answer), 1)


def resp_logp_sum(model, seq_ids, prompt_len):
    """回答部分（seq[prompt_len:]）的 logp 总和，可反传。"""
    x = torch.tensor([seq_ids[:-1]])
    logp = F.log_softmax(model(x), dim=-1)
    labels = torch.tensor(seq_ids[1:])
    gathered = logp[0, torch.arange(labels.shape[0]), labels]
    mask = (torch.arange(labels.shape[0]) + 1 >= prompt_len).float()
    return (gathered * mask).sum()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tokenizer = CharTokenizer.load(ROOT / "out" / "vocab.json")
    data = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=True)
    cfg = TinyConfig(**data["cfg"])
    policy = TinyGPT(cfg)
    policy.load_state_dict(data["model"])
    old = TinyGPT(cfg)  # 记录采样时的旧策略（本极简版仅用于对照说明）
    old.load_state_dict(data["model"])
    for p in old.parameters():
        p.requires_grad_(False)
    old.eval()

    rows = [json.loads(line) for line in open(ROOT / "data" / "sft.jsonl", encoding="utf-8")]
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    for step in range(args.steps):
        row = random.choice(rows)
        prompt = "问：" + row["q"] + "答："
        p_ids = tokenizer.encode(prompt)

        # 1) 当前策略采样 K 个回答
        with torch.no_grad():
            samples = []
            for _ in range(args.group_size):
                out = policy.generate(
                    torch.tensor([p_ids]), max_new=args.max_new,
                    temperature=args.temp, top_k=args.top_k)
                samples.append(out[0].tolist())

        # 2) 规则奖励 + 组内归一化 advantage
        rewards = torch.tensor(
            [rule_reward(row["a"], tokenizer.decode(s)) for s in samples])
        std, mean = rewards.std(), rewards.mean()
        if std.item() < 1e-6:  # 组内无差异，无学习信号
            print("step %3d reward %.3f (std=0 skip)" % (step, mean.item()))
            continue
        adv = (rewards - mean) / (std + 1e-6)

        # 3) 策略梯度：-advantage * 回答 logp（generate 内部会切 eval，这里切回 train）
        policy.train()
        losses = []
        for sample, a in zip(samples, adv):
            if a.item() == 0:
                continue
            logp = resp_logp_sum(policy, sample, len(p_ids))
            losses.append(-a * logp)
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 10 == 0 or step == args.steps - 1:
            print("step %3d loss %.4f reward %.3f adv %.3f | %s"
                  % (step, loss.item(), mean.item(), adv.abs().mean().item(),
                     tokenizer.decode(samples[rewards.argmax().item()])[:44]))

    out_path = ROOT / args.out
    torch.save({"model": policy.state_dict(), "cfg": data["cfg"]}, out_path)
    print("saved ->", out_path)


"""
进阶改造 Prompt
1. 加入 old policy 的 ratio = exp(logp_new - logp_old) 与 clip(0.8,1.2)，
   变成完整的 PPO-clip 风格 GRPO；
2. 加入对参考模型（SFT 版）的 KL 惩罚项：-beta * kl(policy || ref)；
3. 把规则奖励换成语义相似度/裁判模型，或 hh-rlhf 真实偏好标签；
4. 增加评测集：固定 20 条 unseen 问题，每轮统计 mean rule reward；
5. 对照 PPO（需要 Critic 网络）与 GRPO（组基线）的方差与显存差异。
"""

if __name__ == "__main__":
    main()
