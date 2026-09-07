# -*- coding: utf-8 -*-
"""
第39章 批量调度与并行深度适配：连续批处理模拟（一键脚本）

【层级】L2（调度模拟：对比“等齐再批”与“先到先服务连续批”的吞吐）
【环境依赖】Python 3.10+，标准库
【核心逻辑】请求按泊松间隔到达、长度不一；调度器每 tick 把空闲槽位塞入
待处理请求（连续批处理），对比“静态等齐”（每 tick 固定批）的完成时间。
【关键参数】--n 30（默认）；--max_batch 4（默认）；--tick 1（每 tick=1 个 decode 步）。
【避坑】真实系统还要管抢占/优先级/PD 分离；模拟输出为相对收益参考。
【运行结果示例】
$ python continuous_batching_demo.py
avg_completion continuous=17.8 static=19.8 | 提速=9.7%（--n 12）
【高频报错 Top5】
1. 饥饿：长请求占满槽位；加最长等待时间；2. 到达率太低看不出差距；3. 收益算错：
比平均完成时间；4. 想上生产：接 vLLM scheduler；5. 随机波动：固定 seed。
【输出解读】continuous 平均完成时间更短 = 连续批处理收益。
【工程改造方向】接真实模型推理循环；PD 分离/预填充抢占。
"""

import argparse
import random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--max_batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def simulate(a, continuous):
    random.seed(a.seed)
    reqs = []
    t = 0
    for i in range(a.n):
        reqs.append((t, random.randint(2, 12)))
        t += random.randint(1, 3)
    done = []
    active = []
    idx = 0
    clock = 0
    while idx < len(reqs) or active:
        # 新请求到达
        while idx < len(reqs) and reqs[idx][0] <= clock:
            active.append([reqs[idx][1], 0])  # remain, wait
            idx += 1
        if continuous:
            # 只取前 max_batch 个执行
            run = active[:a.max_batch]
        else:
            # 静态等齐：够批才执行；若没有新请求会来（尾部），允许不满批排空
            enough = len(active) >= a.max_batch
            no_more = idx >= len(reqs)
            run = active[: a.max_batch] if (enough or no_more) else []
        if not run:
            clock += 1
            continue
        for r in active[:]:  # 每 tick 全部活动请求也推进（模拟并行度内）
            r[1] += 1 if r in run else 0
        for r in list(active):
            if r[1] >= r[0]:
                done.append(clock)
                active.remove(r)
        clock += 1
    return sum(done) / len(done)


def main():
    a = parse_args()
    ct = simulate(a, True)
    st = simulate(a, False)
    print("avg_completion continuous=%.1f static=%.1f | 提速=%.1f%%"
          % (ct, st, 100 * (st - ct) / max(st, 1e-9)))


if __name__ == "__main__":
    main()