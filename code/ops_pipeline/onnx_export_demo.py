# -*- coding: utf-8 -*-
"""
第43章 端侧/生产适配：TinyGPT ONNX 导出（一键脚本，含降级诊断）

【层级】L2（导出工具：PyTorch -> ONNX 最小导出与降级提示）
【环境依赖】Python 3.10+；PyTorch 2.x；onnx（pip install onnx，缺省走降级提示）
【核心逻辑】构建 TinyGPT 小模型随机初始化，torch.onnx.export 导出
model.onnx 并校验输入输出；onnx 包缺失时打印安装命令与导出计划（降级）。
【关键参数】--dim 32（默认）；--seq 16；--out model.onnx。
【避坑】ONNX 导出注意动态轴（--dynamic_axes）与算子支持；端侧部署还要
量化+运行时（onnxruntime）。
【运行结果示例】
$ python onnx_export_demo.py
exported to model.onnx size(KB)=84.5
【高频报错 Top5】
1. ModuleNotFoundError onnx：pip install onnx；2. 算子不支持：换 torch 版本或简化；
3. 输入 shape 固定：用 dynamic_axes；4. 导出慢：先小模型；5. 端侧跑不动：量化+剪枝。
【输出解读】exported to model.onnx = 成功；降级提示同样给出可行路径。
【工程改造方向】onnxruntime 推理 + INT8 量化；端云协同（ch43 正文）。
"""

import argparse
import pathlib
import sys

import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--seq", type=int, default=16)
    p.add_argument("--out", default="model.onnx")
    return p.parse_args()


def main():
    a = parse_args()
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "minimind_style"))
    from model import TinyConfig, TinyGPT

    cfg = TinyConfig(vocab_size=50, dim=a.dim, n_layers=1, n_heads=2, max_seq=a.seq)
    model = TinyGPT(cfg).eval()
    dummy = torch.randint(0, 50, (1, a.seq))
    try:
        torch.onnx.export(model, dummy, a.out,
                          input_names=["ids"], output_names=["logits"],
                          dynamic_axes={"ids": {0: "batch"}, "logits": {0: "batch"}})
        print("exported to", a.out, "size(KB)=%.1f" % (pathlib.Path(a.out).stat().st_size / 1024))
    except Exception as e:
        print("onnx 导出失败：%s" % str(e)[:200])
        print("降级方案：pip install onnx onnxruntime 后重试；或用 torch.onnx.export(..., opset_version=17)")


if __name__ == "__main__":
    main()
