# -*- coding: utf-8 -*-
"""
第41章 全链路串联：数据→预训练→SFT→生成评估 一键编排

【层级】L2（编排脚本：子进程串联 minimind_style 各阶段）
【环境依赖】Python 3.10+；PyTorch 2.x；需 minimind_style 目录
【核心逻辑】按顺序执行：make_corpus -> pretrain(小步数) -> train_sft(小 epochs)
-> generate 冒烟；各步失败即中断并打印阶段日志；成功打印全链路摘要。
【关键参数】--pretrain_steps 120（默认）；--sft_epochs 15（默认）。
【避坑】全程 CPU 约 1-3 分钟；真实链路要加数据版本/checkpoint 归档；
失败定位看打印的 stage=xx 行。
【运行结果示例】
$ python end_to_end.py
== stage=1_data ==
== stage=1_data SUCCESS ==
== stage=2_pretrain SUCCESS ==
== stage=3_sft SUCCESS ==
== stage=4_smoke SUCCESS ==
END_TO_END OK
【高频报错 Top5】
1. 子进程找不到模块：cwd 必须切到 minimind_style；2. 步数太小效果差：调大；
3. 磁盘满：out/ 清理；4. 某阶段失败：看 stage 打印与 train.log；5. GPU 机器显存：
加 --pretrain_steps 大参数。
【输出解读】看到 4 个 stage 均 SUCCESS 即链路闭环。
【工程改造方向】接 DVC/MLflow 记录版本与指标；失败自动重试。
"""

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "minimind_style"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain_steps", type=int, default=120)
    p.add_argument("--sft_epochs", type=int, default=15)
    return p.parse_args()


def run(stage, cmd):
    print("== stage=%s ==" % stage, flush=True)
    subprocess.run([sys.executable] + cmd, cwd=str(ROOT), check=True)
    print("== stage=%s SUCCESS ==" % stage, flush=True)


def main():
    a = parse_args()
    run("1_data", ["data/make_corpus.py"])
    run("2_pretrain", ["pretrain.py", "--steps", str(a.pretrain_steps)])
    run("3_sft", ["train_sft.py", "--epochs", str(a.sft_epochs)])
    run("4_smoke", ["generate.py", "--prompt", "人工智能"])
    print("END_TO_END OK")


if __name__ == "__main__":
    main()
