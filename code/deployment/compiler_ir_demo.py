# -*- coding: utf-8 -*-
"""
第35章 推理编译器核心原理：极简 IR + 拓扑排序 + 代码生成（一键脚本）

【层级】L2（教学编译器骨架：图 -> IR -> 调度 -> 代码字符串）
【环境依赖】Python 3.10+，标准库
【核心逻辑】定义 IR 节点（op/inputs/attrs），构建 matmul+add 计算图；
按依赖做拓扑排序生成调度顺序，再生成类 C 伪代码字符串并打印。
【关键参数】--graph matmul_add（默认）：内置示例图。
【避坑】真实编译器有 IR 规范化/算子选择/内存规划，本文件只演示骨架；
代码生成产物需再交给后端（如 CUDA/CPU kernel）。
【运行结果示例】
$ python compiler_ir_demo.py
topo order: [x, W, b, mm, add]
--- generated pseudo code ---
float* x = input();
tensor W = load_param();
tensor b = load_param();
tensor mm = matmul(x, W);
tensor add = elem_add(mm, b);
output(y);
【高频报错 Top5】
1. 循环依赖：图必须 DAG；2. 拓扑顺序错：用入度队列算法；3. 代码生成缩进错：
用模板字符串；4. 想真编译：接 torch.compile/TVM；5. 内存规划缺失：真实会炸显存。
【输出解读】打印 topo order 与生成代码 = IR 管线可跑。
【工程改造方向】加常量折叠/算子融合（ch36）；动态 shape 规划（ch37）。
"""

import argparse
import collections


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="matmul_add")
    return p.parse_args()


class Node:
    def __init__(self, name, op, ins=(), attrs=""):
        self.name, self.op, self.ins, self.attrs = name, op, ins, attrs
        self.outs = []


def build():
    # y = (W @ x) + b
    x = Node("x", "input")
    W = Node("W", "param")
    b = Node("b", "param")
    mm = Node("mm", "matmul", (x, W))
    add = Node("add", "add", (mm, b))
    for n in (x, W, b, mm, add):
        for i in n.ins:
            i.outs.append(n.name)
    return [x, W, b, mm, add]


def topo(nodes):
    indeg = {n.name: len(n.ins) for n in nodes}
    adj = collections.defaultdict(list)
    for n in nodes:
        for i in n.ins:
            adj[i.name].append(n)
    q = collections.deque([n for n in nodes if indeg[n.name] == 0])
    order = []
    while q:
        n = q.popleft()
        order.append(n)
        for m in adj[n.name]:
            indeg[m.name] -= 1
            if indeg[m.name] == 0:
                q.append(m)
    return order


def codegen(order):
    lines = []
    for n in order:
        if n.op == "input":
            lines.append("float* %s = input(%s);" % (n.name, n.attrs))
        elif n.op == "param":
            lines.append("tensor %s = load_param(%s);" % (n.name, n.attrs))
        elif n.op == "matmul":
            lines.append("tensor %s = matmul(%s, %s);" % (n.name, n.ins[0].name, n.ins[1].name))
        elif n.op == "add":
            lines.append("tensor %s = elem_add(%s, %s);" % (n.name, n.ins[0].name, n.ins[1].name))
    lines.append("output(y);")
    return lines


def main():
    parse_args()
    nodes = build()
    order = topo(nodes)
    print("topo order:", [n.name for n in order])
    print("--- generated pseudo code ---")
    print("\n".join(codegen(order)))


if __name__ == "__main__":
    main()