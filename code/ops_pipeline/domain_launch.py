# -*- coding: utf-8 -*-
"""
第42章 领域大模型定制化落地：领域数据→续训→自适应 一键启动

【层级】L2（领域启动器：串起 data_engineering/domain_build 与 training_tools）
【环境依赖】Python 3.10+；PyTorch 2.x
【核心逻辑】按序执行：domain_build(生成领域语料+QA) -> continue_pretrain(增量)
-> domain_adapt(小样本微调)；每步可 --skip。
【关键参数】--domain demo（默认）；--steps 100（续训步数）；--epochs 20（自适应）。
【避坑】领域数据规模决定效果：先用 ch10 流水线扩充数据再跑本脚本；
每步失败会中断，日志看 stage。
【运行结果示例】
$ python domain_launch.py
== stage: domain_build.py ==
== stage: continue_pretrain.py ==
saved -> .../out/pretrain_v2.pt
== stage: domain_adapt.py ==
saved -> .../out/domain_adapt.pt
DOMAIN_LAUNCH OK
【高频报错 Top5】
1. 模块路径：cwd 需在 code 仓库根；2. 词表外字符：先扩数据；3. 显存：步数调小；
4. 效果差：数据量不足（见 ch10/ch26）；5. 版本不匹配：统一 torch 版本。
【输出解读】stage1-3 SUCCESS 且最后打印领域回答样例 = 定制闭环。
【工程改造方向】接评测（ch20）与 AB（ch44）形成迭代闭环。
"""

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="demo")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--epochs", type=int, default=20)
    return p.parse_args()


def run(cwd, cmd):
    print("== stage:", cmd[0], "==", flush=True)
    subprocess.run([sys.executable] + cmd, cwd=str(cwd), check=True)


def main():
    a = parse_args()
    run(ROOT / "data_engineering",
        ["domain_build.py", "--domain", a.domain, "--out_dir", "_tmp/domain_out"])
    run(ROOT / "training_tools", ["continue_pretrain.py", "--steps", str(a.steps)])
    run(ROOT / "training_tools", ["domain_adapt.py", "--epochs", str(a.epochs)])
    print("DOMAIN_LAUNCH OK")


if __name__ == "__main__":
    main()
