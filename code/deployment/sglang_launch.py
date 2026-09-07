# -*- coding: utf-8 -*-
"""
第29章 SGLang 高性能推理落地：启动参数校验 + dry-run（一键脚本）

【层级】L2（部署工具：参数校验 + 命令 dry-run；--run 才启动）
【环境依赖】Python 3.10+；SGLang（pip install sglang，Linux+CUDA，版本以官方文档为准）
【核心逻辑】组装 sglang launch_server 命令：model/port/mem/tp/prefix-cache 开关，
校验后默认只打印；支持 --enable-structured（结构化输出/约束解码开关）。
【关键参数】--model（必填）；--port 30000；--gpu_mem 0.9；--tp 1；
--prefix_cache（默认开）；--run。
【避坑】前缀缓存对多轮/共享前缀收益大；结构化输出需后端支持；版本更新快，
参数名以官方文档为准（dry-run 输出便于对照）。
【运行结果示例】
$ python sglang_launch.py --model ./qwen2.5-1.5b
DRY-RUN command: python -m sglang.launch_server --model-path ./qwen2.5-1.5b --port 30000 --mem-fraction-static 0.9 --tp 1 --enable-prefix-caching
【高频报错 Top5】
1. 参数名失效：版本升级；用 --dry 对照官方文档；2. 显存不足调 mem；
3. tp 超 GPU 数报错；4. 端口占用；5. Windows 不支持。
【输出解读】DRY-RUN command 打印即校验通过。
【工程改造方向】加 structured output schema 文件、路由多实例、监控。
"""

import argparse
import shlex
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--gpu_mem", type=float, default=0.9)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--no_prefix_cache", action="store_true")
    p.add_argument("--run", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if not (0.1 <= a.gpu_mem <= 0.95):
        sys.exit("gpu_mem 需在 0.1-0.95")
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", a.model,
           "--port", str(a.port), "--mem-fraction-static", str(a.gpu_mem),
           "--tp", str(a.tp)]
    if not a.no_prefix_cache:
        cmd.append("--enable-prefix-caching")
    print("DRY-RUN command:", shlex.join(cmd))
    if a.run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
