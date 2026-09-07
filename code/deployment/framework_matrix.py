# -*- coding: utf-8 -*-
"""
第27章 部署框架横向对比：环境检测 + 选型矩阵（一键脚本）

【层级】L1（选型工具：本机可用性检测，输出推荐矩阵）
【环境依赖】Python 3.10+，标准库；检测目标框架：vllm/sglang/tensorrt_llm/llama_cpp
【核心逻辑】逐一 import 探测各框架并输出：版本/可用性/适用场景建议表；
--all 时尝试打印每个框架的典型启动命令骨架（dry-run 不真正拉起服务）。
【关键参数】--all（默认关）：是否展示各框架命令骨架。
【避坑】框架安装依赖 CUDA/torch 版本矩阵，检测失败不代表不能用；
生产选型还要看：吞吐、前缀复用、结构化输出、量化支持。
【运行结果示例】
$ python framework_matrix.py
vllm           -     高吞吐/生产首选
sglang         -     前缀复用/结构化输出强
tensorrt_llm   -     极致延迟/需要编译期
llama_cpp      -     端侧/CPU 可跑
【高频报错 Top5】
1. 误报不可用：import 路径不同（如 sglang 子模块）；2. 版本不兼容：与 torch 对齐；
3. 检测慢：框架 import 慢，可加超时；4. Windows 上多数框架不支持（正常）；
5. dry-run 命令需按官方文档核对参数名。
【输出解读】有 ✅ 的框架可直接用 launch 脚本；全 ✗ 时先装依赖。
【工程改造方向】接 CI 每天检测环境矩阵；按任务输出吞吐/时延对比表。
"""

import argparse
import importlib
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    return p.parse_args()


FRAMEWORKS = {
    "vllm": ("vllm", "高吞吐/生产首选"),
    "sglang": ("sglang", "前缀复用/结构化输出强"),
    "tensorrt_llm": ("tensorrt_llm", "极致延迟/需要编译期"),
    "llama_cpp": ("llama_cpp", "端侧/CPU 可跑"),
}


def main():
    args = parse_args()
    print("%-14s %-5s %s" % ("framework", "avail", "适用"))
    for name, (mod, note) in FRAMEWORKS.items():
        try:
            importlib.import_module(mod)
            avail = "OK"
        except Exception:
            avail = "-"
        print("%-14s %-5s %s" % (name, avail, note))
    if args.all:
        print("commands:")
        print("  vllm:   vllm serve ./model --max-model-len 8192 --gpu-memory-utilization 0.9")
        print("  sglang: python -m sglang.launch_server --model-path ./model --port 30000")
        print("  trtllm: trtllm-build --checkpoint_dir ckpt --output_dir engine")
        print("  llama:  llama-server -m model.gguf --port 8080")


if __name__ == "__main__":
    main()
