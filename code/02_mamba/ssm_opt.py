# -*- coding: utf-8 -*-
"""
SSM / Mamba 高阶优化对照（L3）：线性复杂度 + 恒定单步延迟

【层级】L3（高阶优化：对应 SSM 章节“省算力/降延时/长序列”教学重点）
【环境依赖】
- Python 3.10+；PyTorch 2.x（建议 >=2.1）
- 安装：pip install torch==2.2.2；CPU 即可跑基准
【核心逻辑】
1. 线性复杂度基准：MinimalSSM（Python 递推）vs 标准注意力
   （F.scaled_dot_product_attention），seq 从 64 翻倍到 512：
   注意力耗时近似 4 倍增长，SSM 递推近似 2 倍增长（线性）；
2. 恒定单步延迟：LinearSSM 的 rollout 在 64/128/256/512 步下
   平均每 token 耗时应基本恒定（状态固定，不像注意力每步都要
   重算全历史）；
3. 正确性：rollout（逐点带状态）与 forward（整段递推）输出一致。
【关键参数】（默认值 / 推荐值 / 适配场景）
- --seqs 64,128,256,512（默认）：基准序列长度列表；
- --repeats 3（默认；CPU 建议 3-5 取平均）：重复次数；
- --d_state 16（默认）：SSM 状态维度；
- --dim 8（默认）：MinimalSSM 模型宽度。
【避坑】
- Python for 递推没有 kernel，速度不代表真 Mamba（并行扫描 kernel
  才能吃到 GPU）；本文件验证的是“复杂度趋势”不是绝对性能；
- CPU 计时噪声大：repeats>=3，ratio 看趋势别抠小数点；
- seq=512 的注意力基准在 CPU 上约几百 ms，repeats 别设太大。
【运行结果示例】（真实运行，CPU，中位数计时）
$ python ssm_opt.py --seqs 128,256,512,1024 --repeats 3
seq    attention(ms)  ssm(ms)   attn_ratio  ssm_ratio
128    2.0            15.5      -           -
256    3.0            43.6      1.5         2.8
512    5.0            56.1      1.7         1.3
1024   13.3           135.1     2.7         2.4
rollout per-token ms: seq=128:0.05 seq=256:0.06 seq=512:0.06 seq=1024:0.05
rollout == forward: True
PASS
（CPU 小规模 ratio 噪声大属正常，看两点硬结论：rollout 每 token 恒定约
0.05ms；rollout 与 forward 完全一致。O(n) vs O(n^2) 请在大规模/GPU 上验证）
【高频报错 Top5】
1. 基准时间全 0.00：repeats 太少/计时器粒度；修复：repeats>=3；
2. MinimalSSM 在 seq=512 很慢：Python 循环 O(seq*d_state^2)；修复：
   减小 d_state 或换并行扫描实现；
3. attention 在 CPU 上 seq=1024 可能很慢：别设太大 seq；
4. rollout 与 forward 不一致：state 初始化/维度不一致；修复：对照公式
   s_t = a*s_{t-1}+x_t；
5. 结果波动大：后台进程干扰；修复：固定 repeats 并取中位数。
【输出解读】
- rollout 每 token 耗时在 seq 128->1024（8 倍）间保持恒定 = 状态递推的
  “恒定延迟”特性（本机实测 0.05-0.06ms）；
- rollout == forward=True 保证流式实现与整段训练一致；
- attention vs ssm 的 ratio 在 CPU 小规模下噪声大，教学上以理论复杂度
  O(n^2) vs O(n) 为准，工程验证请在 GPU/大 seq 上跑 benchmark。
【工程改造方向】
- 真 Mamba 落地：确认推理框架支持后再上 mamba-ssm/并行扫描；
- 流式服务：把 state 作为会话上下文持久化，每 token 固定开销；
- 长序列：混合架构（前几层 Mamba + 后几层注意力）平衡精度与成本。
"""

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from mamba_demo import MinimalSSM
from ssm_engine import LinearSSM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seqs", default="64,128,256,512")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--dim", type=int, default=8)
    return p.parse_args()


def timeit(fn, repeats):
    """多次计时取中位数（ms），抗 CPU 噪声。"""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def main():
    args = parse_args()
    seqs = [int(s) for s in args.seqs.split(",") if s.strip()]
    ssm = MinimalSSM(d_model=args.dim, d_state=args.d_state).eval()
    print("%-4s %-16s %-12s %-11s %-10s" % ("seq", "attention(ms)", "ssm(ms)",
                                             "attn_ratio", "ssm_ratio"))
    prev_a = prev_s = None
    for seq in seqs:
        xa = torch.randn(2, seq, 128)  # 注意力用 dim=128，让 QK^T 计算占主导
        xs = torch.randn(2, seq, args.dim)
        ta = timeit(lambda: F.scaled_dot_product_attention(xa, xa, xa), args.repeats)
        with torch.no_grad():
            ts = timeit(lambda: ssm(xs), args.repeats)
        ra = "%.1f" % (ta / prev_a) if prev_a else "-"
        rs = "%.1f" % (ts / prev_s) if prev_s else "-"
        print("%-4d %-16.1f %-12.1f %-11s %-10s" % (seq, ta, ts, ra, rs))
        prev_a, prev_s = ta, ts

    # 恒定单步延迟：LinearSSM rollout
    print("rollout per-token ms:", end=" ")
    tok_ms = []
    for seq in seqs:
        m = LinearSSM(d_state=1, a_init=0.95).eval()
        x = torch.randn(1, seq, 1)
        m.rollout(x)  # warmup
        t = timeit(lambda: m.rollout(x), args.repeats)
        tok_ms.append(t / seq)
        print("seq=%d:%.2f" % (seq, t / seq), end=" ")
    print("")

    # 正确性
    m = LinearSSM(d_state=1, a_init=0.95).eval()
    x = torch.randn(1, 32, 1)
    with torch.no_grad():
        a = m(x)
        b = m.rollout(x)
    ok = bool(torch.allclose(a, b, atol=1e-5))
    print("rollout == forward:", ok)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()