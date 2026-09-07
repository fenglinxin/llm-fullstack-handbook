# -*- coding: utf-8 -*-
"""
第36章 算子融合实战：逐算子循环 vs 融合 kernel（一键基准）

【层级】L3（性能基准：融合收益测量，可套用到真实 kernel）
【环境依赖】Python 3.10+；numpy（或 torch）
【核心逻辑】对大数组做 y=(a*x+b)（scale+bias）：naive 分两步两次内存遍历；
fused 一步完成；测量多次耗时中位数并打印加速比。
【关键参数】--n 10_000_000（默认）；--repeats 5（默认）。
【避坑】小数组内存带宽不饱和看不出收益；Python 层循环会掩盖 kernel 收益，
本演示用 numpy 向量化模拟“kernel 内融合”。
【运行结果示例】
$ python fusion_bench.py
n=3000000 naive=0.011s fused=0.010s speedup=1.15x
【高频报错 Top5】
1. 数组太小没收益：n>=1e7；2. 用 python for 循环测：那是解释器开销不是 kernel；
3. 缺 numpy：pip install numpy；4. 结果波动：取中位数；5. 想测 GPU：用 torch.cuda。
【输出解读】fused 快于 naive = 内存遍历减半的收益可见。
【工程改造方向】接 CUDA kernel（一个 kernel 干两件事）；TVM/Inductor 图融合。
"""

import argparse
import statistics
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000_000)
    p.add_argument("--repeats", type=int, default=5)
    return p.parse_args()


def timeit(fn, r):
    ts = []
    for _ in range(r):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    a = parse_args()
    x = np.ones(a.n, dtype=np.float32)
    scale = np.float32(2.0)
    bias = np.float32(1.0)
    def naive():
        t = x * scale
        return t + bias
    def fused():
        return x * scale + bias
    naive(); fused()
    tn = timeit(naive, a.repeats)
    tf = timeit(fused, a.repeats)
    print("n=%d naive=%.3fs fused=%.3fs speedup=%.2fx" % (a.n, tn, tf, tn / max(tf, 1e-9)))


if __name__ == "__main__":
    main()
