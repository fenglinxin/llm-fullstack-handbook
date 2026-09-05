# -*- coding: utf-8 -*-
"""
微型 MoE 预训练（P3：Dense FFN -> MoE FFN 对照实验，CPU 可跑）

【环境依赖】Python 3.10+, PyTorch 2.x；CPU 2-4 分钟。

【前置】先跑 data/make_corpus.py（corpus.txt 已入库，通常无需重跑）。

【核心逻辑】
1. 用 build_moe_model 把 TinyGPT 的 FFN 全部替换成 MoEFFN；
2. 与 pretrain.py 相同的滑窗 next-token 训练（标签左移一位）；
3. total loss = 交叉熵 + aux_weight * 负载均衡辅助 loss；
4. 保存 out/moe.pt，生成样例与 Dense 版对照。

【关键参数】--n_experts 4 --top_k 2 --aux_weight 0.01 --steps 400。

【避坑】
1. MoE 参数变多但激活稀疏，小模型在 CPU 上不会更快，这是正常现象；
2. aux loss 别忘了乘小系数，否则主任务被带偏；
3. 若 router 概率长期集中（辅助 loss 不降），说明专家没有分化。

【输出解读】
- loss 从 ln(vocab)≈5.9 下降即训练正常；
- 对比 Dense：同步数下 MoE 拟合略快或相近，参数量明显更大；
- 生成样例能复现语料片段说明 MoE 路由没有破坏学习。
"""

import argparse
import pathlib
import random

import torch
import torch.nn.functional as F

from lora import count_params
from moe import build_moe_model
from model import TinyConfig, TinyGPT
from tokenizer import CharTokenizer
import pretrain  # 复用 make_windows / sample_text

ROOT = pathlib.Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_experts", type=int, default=4)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--aux_weight", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="out/moe.pt")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    corpus = (ROOT / "data" / "corpus.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer()
    tokenizer.fit([corpus])
    ids = tokenizer.encode(corpus)
    windows = pretrain.make_windows(ids, args.seq)
    print("vocab:", len(tokenizer.vocab), "windows:", len(windows))

    cfg = TinyConfig(vocab_size=len(tokenizer.vocab), dim=args.dim,
                     n_layers=args.layers, n_heads=args.heads, max_seq=args.seq)
    dense = TinyGPT(cfg)
    moe = build_moe_model(cfg, n_experts=args.n_experts, top_k=args.top_k)
    d_total, d_tr = count_params(dense)
    m_total, m_tr = count_params(moe)
    print("params dense=%d moe=%d (%.1fx)" % (d_total, m_total, m_total / d_total))

    opt = torch.optim.AdamW(moe.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss()

    for step in range(args.steps):
        batch_ids = [random.choice(windows) for _ in range(args.batch)]
        x = torch.tensor(batch_ids, dtype=torch.long)
        logits = moe(x[:, :-1])
        loss = ce(logits.view(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        aux = sum(block.ffn.aux_loss for block in moe.blocks)
        total_loss = loss + args.aux_weight * aux
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if step % 20 == 0 or step == args.steps - 1:
            text = pretrain.sample_text(moe, tokenizer, "cpu")
            print("step %4d ce %.4f aux %.4f | %s"
                  % (step, loss.item(), aux.item(), text[:40]))

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": moe.state_dict(),
                "cfg": {"vocab_size": cfg.vocab_size, "dim": cfg.dim,
                        "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                        "max_seq": cfg.max_seq,
                        "n_experts": args.n_experts, "top_k": args.top_k}},
               out_path)
    print("saved ->", out_path)


"""
进阶改造 Prompt
1. 训练时打印每个专家的 hit 占比曲线（前 20 步 vs 最后 20 步）；
2. 把 n_experts/top_k 扫描 2/4/8 x 1/2，画“参数量 vs loss”表；
3. 对比同总参数量下 Dense 大模型 vs MoE，哪个更划算；
4. 加 router z-loss（router logits 的方差惩罚），提高训练稳定；
5. 推理端思考：专家权重如何分布到多卡（EP），KV Cache 如何切。
"""

if __name__ == "__main__":
    main()
