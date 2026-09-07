# -*- coding: utf-8 -*-
"""
第34章 流式推理封装与高并发服务化：async 流式 + 并发客户端（一键脚本）

【层级】L2（服务化骨架：可替换成真实模型/HTTP 服务）
【环境依赖】Python 3.10+，标准库 asyncio
【核心逻辑】模拟模型逐 token 产出（sleep 1ms/token）；stream() 为 async
generator；两个客户端并发消费，统计总时长与每 token 平均延迟。
【关键参数】--tokens 50（默认）：单请求长度；--clients 2（默认）：并发数。
【避坑】真实服务用 FastAPI/StreamingResponse + 队列限流；模拟 sleep 用于演示
并发模型，不要在生产照搬。
【运行结果示例】
$ python stream_server_demo.py
clients=2 each=50 tokens | serial=1.539s parallel=0.778s concurrency_speedup=1.98x
tokens received: [50, 50]
【高频报错 Top5】
1. 事件循环嵌套：asyncio.run 里别再 loop.run_until_complete；2. 生成器没 await：
stream 需 async for；3. 并发没生效：客户端要 gather；4. 模拟太慢：tokens 调小；
5. 真实模型接入：把 sleep 换成模型 generate step。
【输出解读】total_time 接近 单请求时长（并行） = 并发流式生效。
【工程改造方向】FastAPI 封装 + 背压/队列 + SSE；负载均衡（ch43）。
"""

import argparse
import asyncio
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=50)
    p.add_argument("--clients", type=int, default=2)
    return p.parse_args()


async def stream(tokens):
    for i in range(tokens):
        await asyncio.sleep(0.001)
        yield "t%d " % i


async def client(cid, tokens):
    got = 0
    async for tok in stream(tokens):
        got += 1
    return got


async def main():
    args = parse_args()
    # 串行基线
    t0 = time.perf_counter()
    for i in range(args.clients):
        await client(i, args.tokens)
    serial = time.perf_counter() - t0
    # 并行
    t0 = time.perf_counter()
    results = await asyncio.gather(*[client(i, args.tokens) for i in range(args.clients)])
    parallel = time.perf_counter() - t0
    print("clients=%d each=%d tokens | serial=%.3fs parallel=%.3fs concurrency_speedup=%.2fx"
          % (args.clients, args.tokens, serial, parallel, serial / max(parallel, 1e-9)))
    print("tokens received:", results)


if __name__ == "__main__":
    asyncio.run(main())