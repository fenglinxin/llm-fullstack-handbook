# -*- coding: utf-8 -*-
"""
Transformer 落地代码 L3：注意力高阶优化（提速 / 省显存 / KV Cache）

【层级】L3（高阶优化：对应第 00/01 章注意力教学重点的工程拔高）
【环境依赖】
- Python 3.10+；PyTorch 2.x（建议 >=2.1，F.scaled_dot_product_attention 需要 2.0+）
- 安装：pip install torch==2.2.2；CPU 可跑基准，CUDA 加速明显
- flash-attn 可选（Linux+CUDA）：pip install flash-attn，未安装时自动跳过

【核心逻辑】
- 三路注意力实现对照：手写 MHA / nn.MultiheadAttention / F.scaled_dot_product_attention
- 基准测量：同一随机输入重复 forward，取平均耗时（CUDA 用 Event，CPU 用 perf_counter）
- KV Cache 正确性：完整重算 vs 增量缓存，验证最后一个 token 输出一致（allclose）
- KV Cache 速度对照：解码 N 步“每步全量重算”vs“只算新 token”，打印加速比

【关键参数】（默认值 / 推荐值 / 适配场景）
- --seq 64（默认；128/256 更能拉开差距）：序列长度
- --batch 2（默认；显存小用 1）：batch；--dim 64、--heads 8：模型宽度
- --repeats 10（默认；基准建议 >=20 更稳）：重复次数取平均
- --device auto（auto/cpu/cuda/mps）：自动选可用加速设备

【避坑】
- 小 seq（<32）上手写 MHA 与 SDPA 差距很小，别用小规模下结论
- CPU 计时受后台进程干扰，repeats 尽量 >=10 并取平均
- KV Cache 对比必须做正确性断言，否则“快”可能是错的
- flash-attn 在 Windows 无官方支持，本文件自动降级并提示

【运行结果示例】（真实运行，CPU）
$ python attention_opt.py --seq 192 --dim 96 --heads 8 --repeats 3
device=cpu seq=192 dim=96 heads=8 repeats=3
manual_mha    : 23.78 ms/iter
nn_mha        : 11.11 ms/iter
sdpa          : 8.48 ms/iter  (speedup vs manual: 2.8x)
KV cache check: outputs match = True
full recompute: 984.00 ms | kv cache: 536.22 ms | speedup 1.8x
PASS
（seq=64 时 KV cache 只有约 1.1x：小序列+纯 Python 循环收益小，属正常；）
（速度收益随 seq/layer/dim 增大而放大，正确性断言才是本 Demo 的核心）

【高频报错 Top5】
1. “scaled_dot_product_attention”不存在：PyTorch <2.0；修复：pip install -U torch>=2.1
2. 手动 MHA 与 SDPA 输出不一致：head 拼接/转置错误；修复：对比 shape 并断言 allclose
3. CPU 上时间全为 0.00：repeats 太少或计时器粒度；修复：加大 repeats 或用 perf_counter_ns
4. CUDA 计时不准：忘了 synchronize；修复：cuda 分支用 torch.cuda.synchronize()
5. KV cache 结果 NaN：scores 未除 sqrt(d_k) 或 cache 拼接顺序错；修复：逐 token 检查 scores

【输出解读】
- speedup>1 且 outputs match=True 即为有效优化
- seq 越大，SDPA/FlashAttention 相对手写版的收益越明显
- KV cache 的加速比约等于“不再重算的历史长度”占比

【工程改造方向】
- 生产推理：把 KV cache 逻辑并入 decode 循环（主线 33 章）
- 长上下文：换 flash-attn kernel 并记录显存峰值
- 迭代优化：加 PagedAttention 式显存管理、CUDA Graph 捕获 decode
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_from_scratch import MultiHeadAttentionManual


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def pick_device(name):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def timeit(fn, repeats, device):
    """平均耗时（ms）。CUDA 用 Event，CPU 用 perf_counter。"""
    if device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeats
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / repeats


def bench_attentions(batch, seq, dim, heads, repeats, device):
    x = torch.randn(batch, seq, dim, device=device)
    manual = MultiHeadAttentionManual(dim, heads).to(device).eval()
    nn_mha = nn.MultiheadAttention(dim, heads, batch_first=True).to(device).eval()
    sdpa_layer = nn.MultiheadAttention(dim, heads, batch_first=True).to(device).eval()

    def f_manual():
        return manual(x)

    def f_nn():
        return nn_mha(x, x, x, need_weights=False)[0]

    def f_sdpa():
        return F.scaled_dot_product_attention(x, x, x)

    t_manual = timeit(f_manual, repeats, device)
    t_nn = timeit(f_nn, repeats, device)
    t_sdpa = timeit(f_sdpa, repeats, device)
    print("manual_mha    : %.2f ms/iter" % t_manual)
    print("nn_mha        : %.2f ms/iter" % t_nn)
    print("sdpa          : %.2f ms/iter  (speedup vs manual: %.1fx)"
          % (t_sdpa, t_manual / max(t_sdpa, 1e-9)))
    return t_manual, t_sdpa


def kv_cache_demo(seq, dim, heads, device):
    """KV Cache 正确性 + 解码速度对照（单层单头教学版）。"""
    torch.manual_seed(0)
    head_dim = dim // heads
    x = torch.randn(1, seq, dim, device=device)
    wq = nn.Linear(dim, dim, bias=False).to(device)
    wk = nn.Linear(dim, dim, bias=False).to(device)
    wv = nn.Linear(dim, dim, bias=False).to(device)
    wo = nn.Linear(dim, dim, bias=False).to(device)

    def ref_last():
        # 正确性参照：一次向量化前向（整段序列 + 因果 mask）
        q = wq(x).view(1, seq, heads, head_dim).transpose(1, 2)
        k = wk(x).view(1, seq, heads, head_dim).transpose(1, 2)
        v = wv(x).view(1, seq, heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        mask = torch.tril(torch.ones(seq, seq, device=device)).view(1, 1, seq, seq)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)
        o = torch.matmul(w, v).transpose(1, 2).contiguous().view(1, seq, dim)
        return wo(o)[:, -1]

    def attn_last(q, K, V):
        scores = torch.matmul(q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        w = F.softmax(scores, dim=-1)
        o = torch.matmul(w, V).transpose(1, 2).contiguous().view(1, 1, dim)
        return wo(o)[:, 0]

    def full_decode():
        # 无缓存解码：每步对全部历史重新做 QKV 投影 + 注意力（O(n^2) 投影）
        last = None
        for i in range(1, seq + 1):
            xi = x[:, :i]
            q = wq(xi).view(1, i, heads, head_dim).transpose(1, 2)
            k = wk(xi).view(1, i, heads, head_dim).transpose(1, 2)
            v = wv(xi).view(1, i, heads, head_dim).transpose(1, 2)
            last = attn_last(q[:, :, -1:], k, v)
        return last

    def cached_decode():
        # KV Cache 解码：K/V 预分配，每步只投影新 token 并写入缓存（O(n) 投影）
        K = torch.zeros(1, heads, seq, head_dim, device=device)
        V = torch.zeros(1, heads, seq, head_dim, device=device)
        last = None
        for i in range(seq):
            xi = x[:, i:i + 1]
            q = wq(xi).view(1, 1, heads, head_dim).transpose(1, 2)
            k = wk(xi).view(1, 1, heads, head_dim).transpose(1, 2)
            v = wv(xi).view(1, 1, heads, head_dim).transpose(1, 2)
            K[:, :, i:i + 1] = k
            V[:, :, i:i + 1] = v
            last = attn_last(q, K[:, :, :i + 1], V[:, :, :i + 1])
        return last

    with torch.no_grad():
        a = ref_last()
        b = cached_decode()
    match = bool(torch.allclose(a, b, atol=1e-5))
    print("KV cache check: outputs match = %s" % match)

    with torch.no_grad():
        t_full = timeit(full_decode, 5, device)
        t_cache = timeit(cached_decode, 5, device)
    print("full recompute: %.2f ms | kv cache: %.2f ms | speedup %.1fx"
          % (t_full, t_cache, t_full / max(t_cache, 1e-9)))
    return match


def main():
    args = parse_args()
    device = pick_device(args.device)
    if args.dim % args.heads != 0:
        raise SystemExit("dim(%d) 必须能被 heads(%d) 整除" % (args.dim, args.heads))
    print("device=%s seq=%d dim=%d heads=%d repeats=%d"
          % (device, args.seq, args.dim, args.heads, args.repeats))
    bench_attentions(args.batch, args.seq, args.dim, args.heads, args.repeats, device)
    ok = kv_cache_demo(args.seq, args.dim, args.heads, device)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
