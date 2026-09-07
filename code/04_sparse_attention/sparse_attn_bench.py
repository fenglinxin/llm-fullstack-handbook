# -*- coding: utf-8 -*-
"""
稀疏注意力高阶基准（L3）：全量 vs 窗口 vs SDPA 的复杂度与正确性

【层级】L3（高阶优化：对应稀疏注意力章节“提速/省显存/长序列”教学重点）
【环境依赖】
- Python 3.10+；PyTorch 2.x（建议 >=2.1，SDPA 需要 2.0+）
- 安装：pip install torch==2.2.2；CPU 可跑基准
【核心逻辑】
1. 三种实现：
   - full：全量 scores（O(n^2) 内存+算力）；
   - window：chunked 滑窗实现（每个 token 只算窗口内 scores，O(n*w)）；
   - sdpa：PyTorch 原生融合内核（无 mask，只作后端速度参照）；
2. 正确性断言：window>=seq 时与 full 全等（allclose）；window 小时
   输出必须与 full 不同（窗口限制确实生效）；
3. 复杂度趋势：理论上 full 是 O(n^2)、window 是 O(n*w)；CPU 小规模下
   向量化 full 常数小，趋势要到长 seq/GPU 才明显（见输出解读）。
【关键参数】（默认值 / 推荐值 / 适配场景）
- --seqs 128,256,512,1024（默认）：序列长度列表；
- --window 32（默认；长文档 64-256）：窗口宽度；
- --dim 16（默认）、--heads 1（默认；python 循环版只测单头）：模型宽度；
- --repeats 3（默认）：中位数计时次数。
【避坑】
- python 循环版 window 实现没有 kernel，常数开销大，CPU 小规模下
  绝对耗时可能反超向量化 full——别用本文件绝对值下结论；
- window 必须覆盖因果前向窗口（本实现只看左侧 w 个历史位置）；
- 窗口注意力会让长程依赖失效，质量测试请配合任务（见 L2 引擎）。
【运行结果示例】（真实运行，CPU，window=32，中位数）
$ python sparse_attn_bench.py --window 32
correctness: window>=seq allclose=True | window<s differs=True
seq  full(ms)   window(ms)  sdpa(ms)  full_ratio  window_ratio
128  0.6        19.5        0.1       -           -
256  1.1        35.7        0.2       1.7         1.8
512  2.1        67.4        0.5       1.9         1.9
1024 4.9        147.0       2.7       2.3         2.2
PASS
（诚实结论：python 循环窗口实现常数开销大，CPU 小规模下绝对耗时反而不如
向量化 full；本文件的正确性断言与“实现形态差异”才是教学重点，
O(n^2) vs O(n*w) 的收益要到长 seq + kernel/GPU 才显现）
【高频报错 Top5】
1. allclose 失败：window 边界/因果索引错；修复：对照 col 区间公式；
2. seq=1024 full 很慢/内存涨：mask 版 O(n^2)；修复：用 window/sdpa；
3. window 计时没增长：chunk 实现写成了全量；修复：检查是否只算窗口列；
4. sdpa 需要 q/k/v [b,h,s,d]：形状错报；修复：transpose(1,2) 统一格式；
5. 结果波动：CPU 噪声；修复：repeats>=3 取中位数。
【输出解读】
- 正确性两项都 True = 实现可信；
- 理论复杂度：full O(n^2)、window O(n*w)、sdpa 由内核优化；本机小规模
  向量化 full 常数小，python 版 window 常数大，绝对值不能直接比；
- 工程验证请到 GPU + 长 seq（>=4096）跑同脚本，并记录显存峰值。
【工程改造方向】
- 工程落地用真实 kernel（FlashAttention 块级窗口/flash-linear-attention）；
- 长文档：滑窗 + 全局 token 或哈希路由补长程；
- 基准扩展到 GPU：显存峰值用 torch.cuda.max_memory_allocated 记录。
"""

import argparse
import math
import statistics
import time

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seqs", default="128,256,512,1024")
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    return p.parse_args()


def timeit(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def full_attn(q, k, v):
    """全量因果注意力（mask 法）。"""
    b, h, s, d = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    causal = torch.tril(torch.ones(s, s, device=q.device, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    w = F.softmax(scores, dim=-1)
    return torch.matmul(w, v)


def window_attn(q, k, v, window):
    """滑窗因果注意力：每个 token 只看左侧 window 个历史位置。"""
    b, h, s, d = q.shape
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(d)
    for i in range(s):  # 逐 token（教学实现；工程用 kernel）
        lo = max(0, i - window + 1)
        q_i = q[:, :, i:i + 1]                     # [b,h,1,d]
        k_i = k[:, :, lo:i + 1]                    # [b,h,<=w,d]
        v_i = v[:, :, lo:i + 1]
        scores = torch.matmul(q_i, k_i.transpose(-2, -1)) * scale
        wts = F.softmax(scores, dim=-1)
        out[:, :, i:i + 1] = torch.matmul(wts, v_i)
    return out


def main():
    args = parse_args()
    seqs = [int(x) for x in args.seqs.split(",") if x.strip()]
    torch.manual_seed(0)

    # 正确性（小规模）
    q = torch.randn(1, args.heads, 64, args.dim)
    k = torch.randn(1, args.heads, 64, args.dim)
    v = torch.randn(1, args.heads, 64, args.dim)
    ok_full = bool(torch.allclose(full_attn(q, k, v),
                                  window_attn(q, k, v, 64), atol=1e-5))
    ok_diff = not bool(torch.allclose(full_attn(q, k, v),
                                      window_attn(q, k, v, 8), atol=1e-4))
    print("correctness: window>=seq allclose=%s | window<s differs=%s"
          % (ok_full, ok_diff))

    print("%-4s %-10s %-11s %-9s %-11s %-12s" % ("seq", "full(ms)", "window(ms)",
                                                  "sdpa(ms)", "full_ratio",
                                                  "window_ratio"))
    prev_f = prev_w = None
    for seq in seqs:
        q = torch.randn(1, args.heads, seq, args.dim)
        k = torch.randn(1, args.heads, seq, args.dim)
        v = torch.randn(1, args.heads, seq, args.dim)
        tf = timeit(lambda: full_attn(q, k, v), args.repeats)
        tw = timeit(lambda: window_attn(q, k, v, args.window), args.repeats)
        ts = timeit(lambda: F.scaled_dot_product_attention(q, k, v), args.repeats)
        rf = "%.1f" % (tf / prev_f) if prev_f else "-"
        rw = "%.1f" % (tw / prev_w) if prev_w else "-"
        print("%-4d %-10.1f %-11.1f %-9.1f %-11s %-12s" % (seq, tf, tw, ts, rf, rw))
        prev_f, prev_w = tf, tw
    print("PASS" if (ok_full and ok_diff) else "FAIL")


if __name__ == "__main__":
    main()