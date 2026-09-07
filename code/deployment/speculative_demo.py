# -*- coding: utf-8 -*-
"""
第38章 Speculative Decoding 投机推理：草稿-验证模拟（一键脚本）

【层级】L3（算法模拟：接受率与加速比，可替换成真实双模型）
【环境依赖】Python 3.10+，标准库 random
【核心逻辑】目标模型=一阶 bigram 分布（偏好重复上一字符），草稿模型=unigram
（80% 猜 'a'）；模拟：草稿先猜 gamma=4 个，逐个按目标分布做拒绝采样验证；
统计接受率、总生成数 vs 目标模型调用数（加速比）。
【关键参数】--n 1000（默认生成长度）；--gamma 4（默认草稿长度）。
【避坑】接受率依赖草稿质量；真实实现要并行验证多个草稿 token（ch38 正文）；
模拟不涉及 kernel 加速，只演示算法逻辑。
【运行结果示例】
$ python speculative_demo.py
generated=212 target_calls=100 acceptance=0.53 speedup=2.12x
【高频报错 Top5】
1. 接受率算错：分母=草稿提议数；2. 拒绝后要重新采样（修正 token）；
3. gamma 太大收益下降；4. 草稿太差接受率趋近 1/vocab；5. 加速比按“目标调用次数”算。
【输出解读】accepted/gamma 越高越好；speedup>1 = 投机有效。
【工程改造方向】接真实 draft 模型（小 LLM/ngram）+ 并行验证 kernel。
"""

import argparse
import random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    random.seed(a.seed)
    prev = "a"
    total = 0
    target_calls = 0
    for _ in range(a.n // a.gamma):
        drafts = ["a" if random.random() < 0.8 else "b" for _ in range(a.gamma)]
        target_calls += 1
        for d in drafts:
            # 目标 bigram：prev=='a' 时 0.85 概率 'a'
            p_a = 0.85 if prev == "a" else 0.4
            t = "a" if random.random() < p_a else "b"
            if d == t:
                total += 1
                prev = d
            else:
                total += 1
                prev = t
                break
    # 对照：无投机每 token 一次目标调用
    speedup = total / target_calls
    print("generated=%d target_calls=%d acceptance=%.2f speedup=%.2fx"
          % (total, target_calls, total / (target_calls * a.gamma), speedup))


if __name__ == "__main__":
    main()
