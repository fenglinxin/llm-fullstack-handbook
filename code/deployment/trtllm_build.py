# -*- coding: utf-8 -*-
"""
第30章 TensorRT-LLM 固化与加速：构建流程 dry-run（一键脚本）

【层级】L2（部署工具：checkpoint 转 engine 的命令编排与校验）
【环境依赖】Linux+CUDA；TensorRT-LLM（安装以官方容器为准，版本敏感）
【核心逻辑】按步骤输出：convert_checkpoint -> trtllm-build -> 校验产物；
默认 dry-run 打印命令；--run 时依序执行并在 engine 目录做文件存在性校验。
【关键参数】--model_dir（必填）；--ckpt_dir out_ckpt；--engine_dir out_engine；
--dtype float16（默认）；--tp 1。
【避坑】不同模型架构要用对应 convert 脚本（GPT/LLaMA/Qwen 参数不同）；
engine 与 torch 版本强绑定，换环境需重建；构建期长，先小 max_len 验证。
【运行结果示例】
$ python trtllm_build.py --model_dir ./qwen2.5-1.5b
step1: python -m tensorrt_llm.commands.convert_checkpoint --model_dir ./qwen2.5-1.5b --output_dir out_ckpt --dtype float16 --tp_size 1
step2: trtllm-build --checkpoint_dir out_ckpt --output_dir out_engine --gemm_plugin auto
step3: 校验 out_engine 内存在 engine 文件
【高频报错 Top5】
1. convert 脚本名随版本变化：dry-run 输出后对照官方；2. 显存不足：分步构建；
3. dtype 不支持：该架构不支持 fp8 等；4. engine 目录空：构建失败看日志；
5. 版本不匹配：必须用官方容器镜像。
【输出解读】dry-run 打印三步命令 = 流程可用；--run 最后打印 engine files 列表。
【工程改造方向】接 CI 每晚重建 engine；产物校验（md5）+ 性能回归。
"""

import argparse
import pathlib
import shlex
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--ckpt_dir", default="out_ckpt")
    p.add_argument("--engine_dir", default="out_engine")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--run", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    ck = [sys.executable, "-m", "tensorrt_llm.commands.convert_checkpoint",
          "--model_dir", a.model_dir, "--output_dir", a.ckpt_dir,
          "--dtype", a.dtype, "--tp_size", str(a.tp)]
    bd = ["trtllm-build", "--checkpoint_dir", a.ckpt_dir, "--output_dir",
          a.engine_dir, "--gemm_plugin", "auto"]
    print("step1:", shlex.join(ck))
    print("step2:", shlex.join(bd))
    print("step3: 校验", a.engine_dir, "内存在 engine 文件")
    if a.run:
        subprocess.run(ck, check=True)
        subprocess.run(bd, check=True)
        files = list(pathlib.Path(a.engine_dir).glob("*.engine"))
        print("engine files:", [str(f) for f in files])


if __name__ == "__main__":
    main()
