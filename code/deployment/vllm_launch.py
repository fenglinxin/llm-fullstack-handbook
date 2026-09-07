# -*- coding: utf-8 -*-
"""
第28章 vLLM 部署实战：启动参数校验 + dry-run（一键脚本）

【层级】L2（部署工具：参数校验后打印可执行命令；--run 时才真正启动）
【环境依赖】Python 3.10+；vLLM（pip install vllm，版本以官方文档为准，需 Linux+CUDA）
【核心逻辑】收集 model/gpu-memory/tensor-parallel/max-len 等参数并校验取值范围，
默认 dry-run 输出命令与建议；--run 时执行（生产请配合 systemd/k8s）。
【关键参数】--model（必填）；--gpu_mem 0.9；--tp 1；--max_len 8192；--port 8000。
【避坑】tp 必须能被 GPU 数整除；gpu_mem 在 0.1-0.95；max_len 受 RoPE 限制；
Windows 无官方支持，dry-run 仍可输出命令。
【运行结果示例】
$ python vllm_launch.py --model ./qwen2.5-1.5b
DRY-RUN command: vllm serve ./qwen2.5-1.5b --gpu-memory-utilization 0.9 --tensor-parallel-size 1 --max-model-len 8192 --port 8000
提示: --run 参数会真正执行；建议先用 curl /health 验证
【高频报错 Top5】
1. 参数越界：脚本会校验并退出；2. 显存不足：gpu_mem 调小；3. tp 大于 GPU 数：
报错提示；4. 模型路径不存在：启动前检查；5. 端口占用：--port 换一个。
【输出解读】打印 “DRY-RUN command” 即参数通过；加 --run 才真正启动。
【工程改造方向】加健康检查轮询、自动扩缩容、metrics 采集。
"""

import argparse
import shlex
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--gpu_mem", type=float, default=0.9)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max_len", type=int, default=8192)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--run", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if not (0.1 <= a.gpu_mem <= 0.95):
        sys.exit("gpu_mem 需在 0.1-0.95")
    if a.tp < 1:
        sys.exit("tp>=1")
    cmd = ["vllm", "serve", a.model, "--gpu-memory-utilization", str(a.gpu_mem),
           "--tensor-parallel-size", str(a.tp), "--max-model-len", str(a.max_len),
           "--port", str(a.port)]
    print("DRY-RUN command:", shlex.join(cmd))
    print("提示: --run 参数会真正执行；建议先用 curl /health 验证")
    if a.run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
