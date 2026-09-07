# -*- coding: utf-8 -*-
"""
第31章 模型量化实操：FP32 vs INT8 动态量化（CPU 可跑一键脚本）

【层级】L3（压缩优化：per-row 动态量化实现 + 误差/体积对比）
【环境依赖】Python 3.10+；PyTorch 2.x（pip install torch==2.2.2，CPU 即可）
【核心逻辑】随机 256x256 权重做 per-row 动态量化：scale=max(abs)/127，
q=round(w/scale)，反量化 w_hat=q*scale；打印相对误差与内存占比；
并用 torch.quantization.quantize_dynamic 对比库实现（可选路径）。
【关键参数】--dim 256（默认）；--mode manual|torch（默认 manual 展示原理）。
【避坑】动态量化对激活无校准；静态量化需要校准集；INT8 速度收益在 GPU/推理
框架中体现，纯 torch CPU 可能更慢。
【运行结果示例】
$ python quantize_demo.py
mode=manual mse=0.000012 size_ratio=25% (fp32->int8)
【高频报错 Top5】
1. 误差过大：权重分布宽，先按行 scale；2. torch.quantize 报错：先转 CPU float；
3. 收益不明显：小矩阵；4. NaN：含 inf 权重先裁剪；5. 精度敏感层别量化。
【输出解读】mse<1e-2 且 size_ratio≈25% = 动态量化工作正常。
【工程改造方向】静态量化+校准集；接 vLLM/TRT 的 INT4/FP8 后端。
"""

import argparse

import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--mode", default="manual", choices=["manual", "torch"])
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    w = torch.randn(args.dim, args.dim) * 0.5
    if args.mode == "manual":
        scale = w.abs().amax(dim=1, keepdim=True) / 127.0
        q = torch.round(w / scale).clamp(-127, 127)
        w_hat = q * scale
    else:
        import copy
        m = nn.Linear(args.dim, args.dim)
        with torch.no_grad():
            m.weight.copy_(w)
        mq = torch.quantization.quantize_dynamic(copy.deepcopy(m), {nn.Linear}, dtype=torch.qint8)
        w_hat = torch.dequantize(mq.weight())
    mse = ((w - w_hat) ** 2).mean().item()
    ratio = (w_hat.numel() * 1) / (w_hat.numel() * 4)
    print("mode=%s mse=%.6f size_ratio=%.0f%% (fp32->int8)" % (args.mode, mse, 100 * ratio))


if __name__ == "__main__":
    main()
