# -*- coding: utf-8 -*-
"""
第20章 微调效果评估体系：任务集 + 指标 + 轻量裁判（一键脚本）

【层级】L2（评估工具：prefix-match/重叠率/关键词裁判；可替换成更强裁判）
【环境依赖】Python 3.10+；PyTorch 2.x；复用 minimind_style（out/sft.pt 缺失自动兜底训练）
【核心逻辑】对 6 条 QA 任务集逐条生成回答，计算：前缀精确率、字符重叠率、
关键词命中分；输出逐条表与汇总。评估集固定 seed 保证可复现对比。
【关键参数】--ckpt out/sft.pt（默认）；--temp 0.6/--top_k 10（默认）；
--bootstrap：ckpt 缺失时自动跑 pretrain 150 步 + SFT 20 epochs（默认开）。
【避坑】生成温度影响指标，对比模型时必须同一温度；前缀匹配偏严、重叠率偏松，
两个一起看；裁判提示词要固定，否则不可比。
【运行结果示例】
$ python eval_harness.py
无线网络连不上怎么办？    1      0.89   先确认连接公司无线网络，再重置网络。
账号被锁怎么办？         1      0.75   等待半小时后重试或联系管理员。
邮箱满了怎么办？         1      0.65   清理大附件并删除旧邮件。
prefix_acc=0.50 mean_overlap=0.74 keyword_hit=6/6
【高频报错 Top5】
1. ckpt 不存在：自动 bootstrap 或先跑 train_sft.py；2. 指标全 0：prompt 模板没拼“答：”；
3. 中文乱码：PYTHONIOENCODING=utf-8；4. 换模型对比要固定 --temp/--top_k；
5. 裁判关键词不覆盖：把关键词表换成自己的领域词。
【输出解读】prefix_acc>0.5 或 overlap>0.6 = 基础对话能力达标；训练前后对比看提升。
【工程改造方向】接主线任务集（数学/代码/长文）与 LLM 裁判（同题多次打分取中位）。
"""

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "minimind_style"
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/sft.pt")
    p.add_argument("--temp", type=float, default=0.6)
    p.add_argument("--top_k", type=int, default=10)
    return p.parse_args()


def ensure_ckpt(ckpt):
    if (ROOT / ckpt).exists():
        return
    print("[bootstrap] 缺少 %s，自动 pretrain 150 步 + SFT 20 epochs..." % ckpt)
    subprocess.run([sys.executable, "data/make_corpus.py"], cwd=str(ROOT), check=True)
    subprocess.run([sys.executable, "pretrain.py", "--steps", "150"], cwd=str(ROOT), check=True)
    subprocess.run([sys.executable, "train_sft.py", "--epochs", "20"], cwd=str(ROOT), check=True)


def overlap(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def main():
    args = parse_args()
    ensure_ckpt(args.ckpt)
    from tokenizer import CharTokenizer
    from model import TinyConfig, TinyGPT
    import torch

    tok = CharTokenizer.load(ROOT / "out" / "vocab.json")
    d = torch.load(str(ROOT / args.ckpt), map_location="cpu", weights_only=True)
    m = TinyGPT(TinyConfig(**d["cfg"]))
    m.load_state_dict(d["model"])
    m.eval()
    rows = [json.loads(l) for l in open(ROOT / "data" / "sft.jsonl", encoding="utf-8")]
    keywords = {"打印机": "电源", "无线网络": "网络", "账号": "管理员",
                "邮箱": "邮件", "人工智能": "智能", "模型质量": "数据"}
    hits = 0
    pref_ok = []
    ovs = []
    print("%-22s %-6s %-6s %s" % ("question", "pref", "overlap", "answer"))
    for r in rows:
        prompt = "问：" + r["q"] + "答："
        gen = m.generate(torch.tensor([tok.encode(prompt)]), max_new=20,
                         temperature=args.temp, top_k=args.top_k)
        text = tok.decode(gen[0].tolist())
        after = text.split("答：", 1)[1] if "答：" in text else ""
        pref = int(after.startswith(r["a"]))
        ov = overlap(after, r["a"])
        kw = any(k in after for k in keywords.get(r["q"][:3], [r["a"][:2]]))
        hits += kw
        pref_ok.append(pref)
        ovs.append(ov)
        print("%-22s %-6d %-6.2f %s" % (r["q"][:11], pref, ov, after[:20]))
    print("prefix_acc=%.2f mean_overlap=%.2f keyword_hit=%d/6"
          % (sum(pref_ok) / len(pref_ok), sum(ovs) / len(ovs), hits))


if __name__ == "__main__":
    main()