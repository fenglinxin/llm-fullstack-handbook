# -*- coding: utf-8 -*-
"""
TinyGPT 微型 DPO（P2：偏好对齐最小实现，纯 PyTorch 手写 loss）

【环境依赖】Python 3.10+, PyTorch 2.x；CPU 1-2 分钟。

【前置】先跑 pretrain.py / train_sft.py 生成 out/sft.pt、out/vocab.json，
再跑 data/dpo_data.py 生成 data/dpo.jsonl（chosen=好回答，rejected=坏回答）。

【核心逻辑】
1. 对每条 (q, chosen, rejected) 分别拼成完整序列；
2. 用 response-only mask 算 chosen/rejected 回答部分的平均对数概率；
3. DPO loss = -log sigmoid( beta * (policy_chosen - ref_chosen
   - policy_rejected + ref_rejected) )；
4. 参考模型 ref 全程冻结，策略模型 policy 从 sft.pt 继续训练。

【关键参数】--beta 0.5 --epochs 40 --batch 2 --lr 1e-5。

【避坑】
1. 参考模型必须冻结并包 no_grad，否则“自己和自己比”会让梯度失效；
2. 玩具实现用“回答部分平均 logp”（除以 token 数），避免长度偏差；
   真实项目（DeepSpeed/HF 实现）一般对整段 logp 求和，用 beta 控制尺度；
3. chosen/rejected 不要出现 vocab 外字符，否则是 UNK、没有学习信号。

【输出解读】margin = chosen_logp - rejected_logp；
训练后 margin 应上升或保持为正，即模型更愿意输出被偏好的回答。
实测：lr=1e-5 时 margin 约 1.45 -> 2.4 且生成保持正常；
把 lr 调大（如 5e-4）margin 会冲到 8+，但生成崩坏——这就是
小样本 DPO 的 KL 漂移/过优化现象，教学上值得故意对比。
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
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_seq", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="out/dpo.pt")
    return p.parse_args()


def build_pair(tokenizer, row, max_seq):
    """返回 (prefix_ids, chosen_ids, rejected_ids)，各自截断到 max_seq。"""
    prefix = "问：" + row["q"] + "答："
    p_ids = tokenizer.encode(prefix)
    c_ids = (p_ids + tokenizer.encode(row["chosen"]))[:max_seq]
    r_ids = (p_ids + tokenizer.encode(row["rejected"]))[:max_seq]
    return p_ids, c_ids, r_ids


def seq_logp(model, prefix_ids, resp_ids):
    """回答部分平均对数概率（tensor，可反传）。

    logits[i] 预测 resp_ids[i+1]，只统计 i+1 >= len(prefix_ids) 的位置；
    其余位置用 mask 归零，不进入分母。
    """
    x = torch.tensor([resp_ids[:-1]])
    logp = F.log_softmax(model(x), dim=-1)  # [1, seq-1, vocab]
    labels = torch.tensor(resp_ids[1:])
    gathered = logp[0, torch.arange(labels.shape[0]), labels]
    mask = (torch.arange(labels.shape[0]) + 1 >= len(prefix_ids)).float()
    return (gathered * mask).sum() / max(mask.sum(), 1)


def dpo_loss(policy, ref, batch, beta):
    losses, margins = [], []
    for p_ids, c_ids, r_ids in batch:
        pc = seq_logp(policy, p_ids, c_ids)   # 策略给 chosen 的 logp
        pr = seq_logp(policy, p_ids, r_ids)   # 策略给 rejected 的 logp
        with torch.no_grad():
            rc = seq_logp(ref, p_ids, c_ids)  # 参考给 chosen 的 logp（冻结）
            rr = seq_logp(ref, p_ids, r_ids)
        margin = (pc - rc) - (pr - rr)        # 相对参考的偏好优势
        losses.append(-F.logsigmoid(beta * margin))
        margins.append(pc - pr)
    return torch.stack(losses).mean(), torch.stack(margins).mean()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tokenizer = CharTokenizer.load(ROOT / "out" / "vocab.json")
    data = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=True)
    cfg = TinyConfig(**data["cfg"])
    policy = TinyGPT(cfg)
    policy.load_state_dict(data["model"])
    ref = TinyGPT(cfg)  # 参考模型：与策略同一起点，但全程冻结
    ref.load_state_dict(data["model"])
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    rows = [json.loads(line) for line in open(ROOT / "data" / "dpo.jsonl", encoding="utf-8")]
    pairs = [build_pair(tokenizer, r, args.max_seq) for r in rows]
    print("dpo pairs:", len(pairs))

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    steps_per_epoch = max(1, len(pairs) // args.batch)
    total_steps = args.epochs * steps_per_epoch
    for step in range(total_steps):
        batch = random.choices(pairs, k=args.batch)
        loss, margin = dpo_loss(policy, ref, batch, args.beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == total_steps - 1:
            prompt = "问：打印机连不上怎么办？答："
            out = policy.generate(torch.tensor([tokenizer.encode(prompt)]),
                                  max_new=24, temperature=0.6, top_k=10)
            print("step %3d loss %.4f margin %.4f | %s" % (step, loss.item(),
                  margin.item(), tokenizer.decode(out[0].tolist())[:36]))

    out_path = ROOT / args.out
    torch.save({"model": policy.state_dict(), "cfg": data["cfg"]}, out_path)
    print("saved ->", out_path)


"""
进阶改造 Prompt
1. 把平均 logp 换成整段 logp 求和（更接近 DPO 论文，长度控制交给 beta）；
2. 一个问题配多个 rejected 样本，训练更稳（主线 24 章偏好数据构造）；
3. 与 SFT 模型对比生成分布，观察“偏好”是否真的改变输出（小数据下差异有限）；
4. 增加 KL 惩罚项或 mix 参考模型输出，防止策略漂移（主线 25 章 RLHF）；
5. 换真实偏好数据集前，先确认 tokenizer 能覆盖数据字符。
"""

if __name__ == "__main__":
    main()

"""
规范字段补充（全局强制代码落地规范）

【层级】L2（对齐训练：手写 DPO loss 的最小完整实现）
【运行结果示例】（真实运行，CPU，默认参数 lr=1e-5/beta=0.5）
sft 模型 mean margin=1.45 -> dpo 模型 mean margin=2.38（6 条偏好对评测）
step 119 loss 0.6199 margin 0.8886 | 问：打印机连不上怎么办？答：先检查电源和网络，然后重启打印机，并提交工单。
saved -> .../out/dpo.pt
（margin 温和上升且贪心生成保持完整 = 对齐生效；lr=5e-4 会把 margin 冲到 8+ 但生成崩坏，
这是小样本 KL 漂移的教学案例）
【高频报错 Top5】
1. ref 也在反传：ref 必须 requires_grad_(False) + no_grad（本文件已做）；
2. margin 恒 0：policy 与 ref 同起点且未训练；跑几轮后看趋势；
3. chosen/rejected 有 UNK：无学习信号；修复：字符需在词表内（dpo_data 已约束）；
4. 生成质量骤降：lr 过大过优化；修复：lr<=1e-5、epochs<=40；
5. 长度偏差：平均 logp 与求和 logp 结论不同；修复：明确口径再对比。
【工程改造方向】加 ratio clip 与 KL 惩罚变完整 GRPO（train_grpo.py）；换真实偏好数据。
"""
