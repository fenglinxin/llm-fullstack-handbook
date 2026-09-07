# -*- coding: utf-8 -*-
"""
FlashAttention 落地 Demo：调用 flash-attn 库 + 自动回退到 PyTorch SDPA

【层级】L1（极简 Demo：新手跑通，CPU 可跑自动降级路径）

【环境依赖】
- Python 3.10+；PyTorch 2.x（建议 >=2.1，SDPA 需要 2.0+）
- 安装：pip install torch==2.2.2（GPU 版按官网 CUDA 版本装）
- 可选：flash-attn（仅 Linux + CUDA；pip install flash-attn，版本约束以官方文档为准）
- 无 flash-attn 时自动降级到 F.scaled_dot_product_attention（本机即此路径）

【核心逻辑】
- FlashAttention 是 IO 感知的精确注意力：分块计算 + 在线 softmax；
- 使用方式与普通注意力一致：输入 q/k/v，输出 o；
- 本 Demo 演示调用与形状要求，并自动降级。

【关键参数】
- batch=2, seq=64, n_heads=4, head_dim=16（head_dim 需能被 8 整除更稳）
- dtype=float16 时 flash 最快；CPU/无卡环境请用 float32

【避坑】
1. flash_attn_func 要求输入在 CUDA 且为半精度/bf16；
2. 序列长度与 head_dim 需满足 kernel 约束，版本不同要求不同；
3. 降级路径输出应与 flash 数值接近（误差很小）。

【运行结果示例】（真实运行，CPU 无 flash-attn）
$ python flash_attn_demo.py
backend: torch_sdpa (fallback, reason=No module named flash_attn)
output: (2, 64, 4, 16) 耗时 60.8 ms
（backend 打印降级原因但 output 形状正确 = 自动降级成功；Linux+CUDA 装 flash-attn 后会打印 flash_attn）

【输出解读】
- 输出形状 [batch, seq, n_heads, head_dim]；
- 打印使用的后端与耗时；
- 降级路径的数值与 flash 几乎一致（误差来自实现细节，可自行断言）。

【高频报错 Top5】
1. “No module named flash_attn”：未安装；修复：Linux+CUDA 装 flash-attn，或接受自动降级；
2. “Expected q,k,v to be on CUDA”：flash kernel 不支持 CPU；修复：数据 .cuda() 或走 SDPA 降级；
3. dtype 不支持：flash 需 fp16/bf16；修复：q/k/v 转 .half()/.bfloat16()；
4. 形状约束报错：head_dim/seq 不满足 kernel 要求；修复：查该版本 flash-attn 约束，调 head_dim 为 8 的倍数；
5. 装了 flash 仍报错：与 torch 版本不匹配；修复：按官方文档对齐 torch/cuda/flash 版本。

【工程改造方向】
- 在主线 36 章融合注意力中替换自研 attention；
- 对长文本/长上下文训练与推理收益最大；
- 配合 KV 量化（33 章）进一步省显存。
"""

import time

import torch
import torch.nn.functional as F


def run_flash(q, k, v):
    """优先 flash_attn，失败则用 PyTorch SDPA 回退。"""
    try:
        from flash_attn import flash_attn_func

        o = flash_attn_func(q, k, v)
        return o, "flash_attn"
    except Exception as exc:
        # 回退：PyTorch 原生 SDPA（等价、支持更多环境）
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        return o, "torch_sdpa (fallback, reason=%s)" % str(exc)[:60]


def main():
    torch.manual_seed(0)
    batch, seq, heads, head_dim = 2, 64, 4, 16
    q = torch.randn(batch, seq, heads, head_dim)
    k = torch.randn(batch, seq, heads, head_dim)
    v = torch.randn(batch, seq, heads, head_dim)

    t0 = time.time()
    o, backend = run_flash(q, k, v)
    print("backend:", backend)
    print("output:", tuple(o.shape), "耗时 %.1f ms" % ((time.time() - t0) * 1000))


"""
进阶改造 Prompt
1. 在不同 seq 长度（512/2048/8192）下对比 SDPA 与 flash 的显存/速度；
2. 把它接入第 36 章的融合注意力与 torch.compile 流程；
3. 验证长文本任务（第 33 章 needle 评测）前后质量不变；
4. 尝试 causal/局部 mask 版本并测量收益；
5. 在无 flash kernel 的 GPU 上评估 SDPA 是否已足够。
"""

if __name__ == "__main__":
    main()