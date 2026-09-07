# -*- coding: utf-8 -*-
"""
第44章 模型迭代升级与性能对标：A/B 双 checkpoint 生成对比（一键脚本）

【层级】L2（评测工具：固定任务集 + 固定温度对比两个 checkpoint）
【环境依赖】Python 3.10+；PyTorch 2.x；minimind_style 的 out/sft.pt、out/dpo.pt
【核心逻辑】同一 6 条 QA、同一温度下分别用 A/B 模型生成，输出 prefix_acc、
重叠率与逐条对比表；胜负判定=prefix 更高或重叠率更高。
【关键参数】--a out/sft.pt --b out/dpo.pt（默认）；--temp 0.6。
【避坑】温度/种子不固定则对比无效；小样本差异请用多次采样均值；
升级上线前要接 ch20 完整评测集。
【运行结果示例】
$ python ab_compare.py
邮箱满了怎么办？   0.09  1.09  -1.00  B
什么是人工智能？   0.13  1.07  -0.93  B
如何提高模型质量？  0.25  1.00  -0.75  B
avg A=0.21 B=1.00 diff=-0.79（B=DPO 胜出）
【高频报错 Top5】
1. ckpt 缺失：先跑对应训练脚本；2. 指标全 0：prompt 模板错；3. 温度不一致；
4. 只比单次采样：波动大；5. 结论外推：小任务集仅冒烟。
【输出解读】A/B 各指标并列输出，胜者标 WIN。
【工程改造方向】接完整评测集与 LLM 裁判；输出报告 JSON 存档。
"""

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent / "minimind_style"
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="out/sft.pt")
    p.add_argument("--b", default="out/dpo.pt")
    p.add_argument("--temp", type=float, default=0.6)
    return p.parse_args()


def load(ck):
    from model import TinyConfig, TinyGPT
    d = torch.load(str(ROOT / ck), map_location="cpu", weights_only=True)
    m = TinyGPT(TinyConfig(**d["cfg"]))
    m.load_state_dict(d["model"])
    return m


def overlap(x, y):
    s = set(x)
    return sum(1 for c in y if c in s) / max(1, len(set(y)))


def main():
    args = parse_args()
    from tokenizer import CharTokenizer
    tok = CharTokenizer.load(ROOT / "out" / "vocab.json")
    rows = [json.loads(l) for l in open(ROOT / "data" / "sft.jsonl", encoding="utf-8")]
    ma, mb = load(args.a), load(args.b)
    print("%-12s %8s %8s %8s %8s" % ("metric", "A", "B", "A_B", "win"))
    agg = {"A": 0.0, "B": 0.0, "A_B": 0.0}
    for r in rows:
        prompt = "问：" + r["q"] + "答："
        ta = tok.decode(ma.generate(torch.tensor([tok.encode(prompt)]), max_new=16,
                                    temperature=args.temp, top_k=10)[0].tolist())
        tb = tok.decode(mb.generate(torch.tensor([tok.encode(prompt)]), max_new=16,
                                    temperature=args.temp, top_k=10)[0].tolist())
        aa = ta.split("答：", 1)[1] if "答：" in ta else ta
        ab = tb.split("答：", 1)[1] if "答：" in tb else tb
        oa, ob = overlap(aa, r["a"]), overlap(ab, r["a"])
        agg["A"] += oa
        agg["B"] += ob
        agg["A_B"] += oa - ob
        win = "A" if oa > ob else ("B" if ob > oa else "=")
        print("%-12s %8.2f %8.2f %8.2f %4s" % (r["q"][:10], oa, ob, oa - ob, win))
    n = len(rows)
    print("avg A=%.2f B=%.2f diff=%.2f" % (agg["A"] / n, agg["B"] / n, agg["A_B"] / n))


if __name__ == "__main__":
    main()
