# -*- coding: utf-8 -*-
"""
第37章 动态 shape 与显存编译器优化：静态 padding vs 动态调度（一键脚本）

【层级】L2（调度模拟：量化 padding 浪费与动态 batch 收益）
【环境依赖】Python 3.10+，标准库
【核心逻辑】模拟 N 个请求长度 1..64：静态=全部 pad 到 64；动态=按到达顺序
pack 到 max_tokens=128 桶中；比较总计算量与平均完成时间（模拟每 token 1ms）。
【关键参数】--n 20（默认）；--max_tokens 128；--seq_max 64。
【避坑】动态 shape 需要编译器/内核支持可变 batch；本脚本是调度层的收益估计，
不替代真实 kernel benchmark。
【运行结果示例】
$ python dynshape_demo.py
n=20 static_work=1280 dynamic_work=678 compute_waste=1.89x buckets=7
【高频报错 Top5】
1. 长度分布影响结论：固定 seed 多试；2. 桶策略：FCFS 简单但可能碎片；
3. 想真实优化：接 vLLM PagedAttention；4. 显存规划：按桶峰值算；
5. 收益算错：动态计算量=各请求真实长度和。
【输出解读】compute_waste 显著小于 1 = 动态调度省算力。
【工程改造方向】接 PagedAttention/连续批处理（ch39）。
"""

import argparse
import random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--seq_max", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    random.seed(a.seed)
    lens = [random.randint(1, a.seq_max) for _ in range(a.n)]
    static_work = a.n * a.seq_max
    dynamic_work = sum(lens)
    # 动态桶（近似连续批处理）
    buckets, cur = 0, 0
    for ln in lens:
        if cur + ln > a.max_tokens:
            cur = 0
            buckets += 1
        cur += ln
    print("n=%d static_work=%d dynamic_work=%d compute_waste=%.2fx buckets=%d"
          % (a.n, static_work, dynamic_work, static_work / dynamic_work, buckets + 1))


if __name__ == "__main__":
    main()
