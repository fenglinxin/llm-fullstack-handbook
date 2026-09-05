# -*- coding: utf-8 -*-
"""
Mamba / 结构化状态空间模型 落地 Demo

【环境依赖】
- Python 3.10+, PyTorch 2.x（基础版 CPU 可跑）
- 可选：mamba-ssm（Linux + CUDA 环境，安装与版本以官方文档为准）

【两条路径】
- 路径 A：用开源库 mamba-ssm 快速加载 Mamba 做推理；
- 路径 B：用 PyTorch 写一个最小 SSM 直觉版，任何环境可跑，
         帮助理解固定状态 + 逐时间步递推 + 线性复杂度。

【核心逻辑】
- SSM 每步：state = A*state + B*x；y = C*state（简化为可学习的离散递推）；
- Mamba 的选择机制让 A/B/C 随输入变化，本 Demo 用固定矩阵演示骨架。

【避坑】
1. mamba-ssm 依赖 CUDA 与特定 torch 版本，Windows 通常不支持，别硬装；
2. 真正 Mamba 的并行扫描 kernel 无法用简单 for 循环替代性能；
3. 无 GPU 时请走路径 B，理解原理后再上真库。

【输出解读】
- 路径 B 输出形状与输入一致；
- state 维度过大/过小分别对应记太多/记不住。

【工程改造方向】
- 用真 Mamba 前先确认推理框架（vLLM/SGLang）支持该架构；
- 超长序列/流式场景优先评估；
- 混合架构（部分层 Mamba）是折中方向。
"""

import time

import torch
import torch.nn as nn


def try_mamba_lib():
    """路径 A：mamba-ssm 库接入（若环境支持）。"""
    try:
        from mamba_ssm import Mamba
        model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2)
        x = torch.randn(2, 32, 64)  # [batch, seq, d_model]
        y = model(x)
        print("mamba-ssm output:", tuple(y.shape))
    except Exception as exc:
        print("mamba-ssm 不可用（请 Linux+CUDA 环境安装）：", str(exc)[:120])


class MinimalSSM(nn.Module):
    """路径 B：最小状态空间递推直觉版。

    state[t] = A * state[t-1] + B * x[t]
    y[t]     = C * state[t]
    """

    def __init__(self, d_model=8, d_state=16):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.05)
        self.B = nn.Linear(d_model, d_state)
        self.C = nn.Linear(d_state, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch, seq, d = x.shape
        state = torch.zeros(batch, self.A.shape[0])
        outs = []
        for t_step in range(seq):
            state = torch.tanh(
                x[:, t_step] @ self.B.weight.T + self.B.bias + state @ self.A.T
            )
            outs.append(self.C(state))
        y = torch.stack(outs, dim=1)
        return self.norm(x + y)  # 残差 + 归一化，方便训练


def main():
    torch.manual_seed(0)
    try_mamba_lib()

    model = MinimalSSM(d_model=8, d_state=16)
    x = torch.randn(2, 20, 8)
    y = model(x)
    print("MinimalSSM output:", tuple(y.shape))

    # 线性复杂度直观验证：seq 翻倍，时间近似翻倍（而非 4 倍）
    for seq in (64, 128):
        xx = torch.randn(1, seq, 8)
        t0 = time.time()
        with torch.no_grad():
            model(xx)
        print("seq=%d forward %.1f ms" % (seq, (time.time() - t0) * 1000))


"""
进阶改造 Prompt
1. 让 A/B/C 随输入变化（selective scan 的最小版）并观察长程记忆差异；
2. 把逐时间步 for 换成并行扫描思路，理解 kernel 必要性；
3. 用正弦/长程依赖任务对比 MinimalSSM 与 LSTM/Transformer；
4. 在有 CUDA 的 Linux 机器上安装 mamba-ssm，跑真实 Mamba 推理；
5. 评估推理框架（vLLM/SGLang）对 Mamba 架构的支持再决定落地。
"""

if __name__ == "__main__":
    main()
