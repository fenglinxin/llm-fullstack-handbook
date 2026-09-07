# -*- coding: utf-8 -*-
"""
第07章 数据去重算法：精确去重 + MinHash 近似去重（工程脚本）

【层级】L2（工程算法：精确哈希 + MinHash/LSH 近重复检测，可直接嵌入管道）
【环境依赖】Python 3.10+，仅标准库
【核心逻辑】
- 精确去重：整段 sha1 哈希集合；
- 近似去重：3-gram shingle 集合 -> k=32 个 MinHash 签名（随机种子哈希）-> LSH
  （8 波段 x 4 行）把相似文档分桶 -> 桶内算 Jaccard >= 0.8 判定近重复。
【关键参数】--sample：生成演示数据（含完全重复/轻度改写/正常文档，默认关）；
--threshold 0.8（默认；相似度阈值）；--bands 8、--rows 4（LSH 参数）。
【避坑】MinHash 是概率方法会漏报/误报；阈值与带宽按数据分布调；
中文按字切 3-gram，英文建议按词；短文档 shingle 少时直接用 Jaccard。
【运行结果示例】
$ python dedup_minhash.py --sample
docs: 4 exact duplicates: 1
near-dup pairs: 1 [(0, 1)]
sample Jaccard(0,1)=1.000 Jaccard(0,2)=0.639
【高频报错 Top5】
1. 空文档集：先生成样例或检查输入；2. 全判重复：threshold 太低或 shingle 太短；
3. 漏检改写：shingle 长度/band 参数不合适；4. 内存大：文档多时先精确去重再 MinHash；
5. 中文编码：统一 utf-8。
【输出解读】exact_dups / near_dup_pairs / 样例对相似度打印即成功。
【工程改造方向】接 spark/多进程分片；签名落盘做增量去重；band 参数自动调优。
"""

import argparse
import hashlib
import itertools
import random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true")
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--bands", type=int, default=8)
    p.add_argument("--rows", type=int, default=4)
    return p.parse_args()


def shingles(text, k=3):
    text = text.replace(" ", "")
    return set(text[i:i + k] for i in range(len(text) - k + 1))


def minhash(sig_set, n_hash=32, seed=7):
    rnd = random.Random(seed)
    coeffs = [rnd.randrange(1, 1 << 31) for _ in range(n_hash)]
    sig = []
    for c in coeffs:
        sig.append(min(hash(sh) ^ c for sh in sig_set))
    return sig


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def gen_sample():
    base = "注意力机制让模型可以关注上下文中的关键信息并建立长距离依赖关系"
    docs = [base]
    docs.append(base)  # 完全重复
    docs.append(base[:18] + "并建立长程依赖关系" + base[24:])  # 轻度改写
    docs.append("预训练需要大量高质量文本数据与清洗流程支持工程落地")
    return docs


def main():
    args = parse_args()
    docs = gen_sample() if args.sample else ["示例文档"]
    sigs = [minhash(shingles(d)) for d in docs]
    exact = len(docs) - len({hashlib.sha1(d.encode()).hexdigest() for d in docs})
    buckets = {}
    for i, sig in enumerate(sigs):
        for b in range(args.bands):
            key = (b, tuple(sig[b * args.rows:(b + 1) * args.rows]))
            buckets.setdefault(key, []).append(i)
    near = []
    seen = set()
    for cands in buckets.values():
        for a, b in itertools.combinations(sorted(set(cands)), 2):
            if (a, b) in seen:
                continue
            seen.add((a, b))
            if jaccard(shingles(docs[a]), shingles(docs[b])) >= args.threshold:
                near.append((a, b))
    print("docs:", len(docs), "exact duplicates:", exact)
    print("near-dup pairs:", len(near), [(a, b) for a, b in near])
    print("sample Jaccard(0,1)=%.3f Jaccard(0,2)=%.3f"
          % (jaccard(shingles(docs[0]), shingles(docs[1])),
             jaccard(shingles(docs[0]), shingles(docs[2])) if len(docs) > 2 else 0))


if __name__ == "__main__":
    main()
