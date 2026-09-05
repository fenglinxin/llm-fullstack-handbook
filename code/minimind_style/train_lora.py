# -*- coding: utf-8 -*-
"""
TinyGPT 微型 LoRA 训练（P2：用 LoRA 重跑 SFT，观察“冻结基座+训 A/B”）

【环境依赖】Python 3.10+, PyTorch 2.x；CPU 1-2 分钟。

【前置】先跑 pretrain.py 与 train_sft.py 生成 out/sft.pt、out/vocab.json。

【核心逻辑】
1. 加载 sft.pt 基座模型 -> apply_lora（冻结基座，只训 A/B）；
2. 数据/模板/response-only mask 与 train_sft.py 完全一致；
3. 训练完成后保存：lora 适配器（out/lora.pt）+ 合并后的完整模型（out/lora_merged.pt）。

【关键参数】--r 8 --alpha 16 --epochs 60 --lr 1e-3。

【避坑】
1. 基座必须全冻结，只解冻 lora_A/lora_B；
2. 合并前后生成应一致（误差仅在浮点精度）；
3. 玩具数据下 LoRA 与全量 SFT 质量接近，看“可训练参数占比”理解价值。

【输出解读】loss 从约 1.0 降到 0.08 左右说明 LoRA 学到回答；
大模型场景可训练参数占比通常 < 1%，本微型工程约 10%（骨干占比高），
价值在于体会“冻结基座、只训 A/B”的完整流程。
"""

import argparse
import json
import pathlib
import random

import torch

from lora import apply_lora, count_params, merge_lora
from model import TinyConfig, TinyGPT
from tokenizer import CharTokenizer
import train_sft  # 复用 make_sample / collate，保证与 P1 数据一致

ROOT = pathlib.Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="out/sft.pt")
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_seq", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adapter_out", type=str, default="out/lora.pt")
    p.add_argument("--merged_out", type=str, default="out/lora_merged.pt")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tokenizer = CharTokenizer.load(ROOT / "out" / "vocab.json")
    data = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=True)
    cfg = TinyConfig(**data["cfg"])
    model = TinyGPT(cfg)
    model.load_state_dict(data["model"])
    apply_lora(model, r=args.r, alpha=args.alpha)
    total, trainable = count_params(model)
    print("params total=%d trainable=%d (%.3f%%)" % (total, trainable, 100.0 * trainable / total))

    rows = [json.loads(line) for line in open(ROOT / "data" / "sft.jsonl", encoding="utf-8")]
    samples = []
    for row in rows:
        s = train_sft.make_sample(tokenizer, row["q"], row["a"], args.max_seq)
        if s:
            samples.append(s)
    print("sft samples:", len(samples))

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    steps_per_epoch = max(1, len(samples) // args.batch)
    total_steps = args.epochs * steps_per_epoch
    for step in range(total_steps):
        batch = random.choices(samples, k=args.batch)
        x, y = train_sft.collate(batch, args.max_seq)
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == total_steps - 1:
            prompt = "问：打印机连不上怎么办？答："
            out = model.generate(torch.tensor([tokenizer.encode(prompt)]),
                                 max_new=24, temperature=0.6, top_k=10)
            print("step %3d loss %.4f | %s" % (step, loss.item(),
                                               tokenizer.decode(out[0].tolist())[:40]))

    # 保存 LoRA 适配器（只存 A/B，不存基座——换基座即可迁移）
    adapter = {k: v.detach().clone()
               for k, v in model.named_parameters() if "lora_" in k}
    adapter_path = ROOT / args.adapter_out
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": adapter, "cfg": data["cfg"],
                "r": args.r, "alpha": args.alpha}, adapter_path)
    print("adapter saved ->", adapter_path)

    # 合并回普通模型并保存
    merge_lora(model)
    merged_path = ROOT / args.merged_out
    torch.save({"model": model.state_dict(), "cfg": data["cfg"]}, merged_path)
    print("merged model saved ->", merged_path)



"""
进阶改造 Prompt
1. 比较 LoRA vs 全量 SFT 的 loss/生成/可训练参数（主线 26 章）；
2. 只对注意力层打 LoRA vs 只对 FFN 打 LoRA；
3. 尝试把 LoRA 适配器换到另一个基座（pretrain.pt）上看效果；
4. 加 rank 扫描：r=1/4/8/16/64 的 loss 曲线；
5. 用 merge 后的模型做推理，验证速度与内存变化。
"""

if __name__ == "__main__":
    main()
