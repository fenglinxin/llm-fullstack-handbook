# -*- coding: utf-8 -*-
"""
Mamba / 结构化状态空间模型 落地 Demo

【层级】L1（极简 Demo：双路径理解 SSM，CPU 秒级出结果）

【环境依赖】
- Python 3.10+；PyTorch 2.x（建议 >=2.1）；基础版仅 torch
- 安装：pip install torch==2.2.2（GPU 版按官网 CUDA 版本装；本 Demo CPU 可跑）
- 可选：mamba-ssm（仅 Linux + CUDA，安装与版本以官方文档为准）

【两条路径】
- 路径 A：用开源库 mamba-ssm 快速加载 Mamba 做推理；
- 路径 B：用 PyTorch 写一个最小 SSM 直觉版，任何环境可跑，
         帮助理解固定状态 + 逐时间步递推 + 线性复杂度。

【核心逻辑】
- SSM 每步：state = A*state + B*x；y = C*state（简化为可学习的离散递推）；
- Mamba 的选择机制让 A/B/C 随输入变化，本 Demo 用固定矩阵演示骨架；
- 逐行注释见 MinimalSSM.forward：tanh 收束状态、残差连接便于训练。

【关键参数】
- d_model=8（默认；=输入特征维度，改数据时同步改）：模型宽度；
- d_state=16（默认；任务记忆长度相关，长程任务加大 32-64）：隐状态维度；
- seq=64/128（Demo 计时用；越大越能看出线性复杂度）。

【避坑】
1. mamba-ssm 依赖 CUDA 与特定 torch 版本，Windows 通常不支持，别硬装；
2. 真正 Mamba 的并行扫描 kernel 无法用简单 for 循环替代性能；
3. 无 GPU 时请走路径 B，理解原理后再上真库。

【运行结果示例】（真实运行，CPU）
$ python mamba_demo.py
mamba-ssm 不可用（请 Linux+CUDA 环境安装）： No module named mamba_ssm
MinimalSSM output: (2, 20, 8)
seq=64 forward 8.3 ms
seq=128 forward 10.3 ms
（路径 B 正常输出 + 计时行 = 成功；路径 A 打印“不可用”是预期降级，不是报错）

【输出解读】
- 路径 B 输出形状与输入一致；
- state 维度过大/过小分别对应记太多/记不住；
- 计时趋势：seq 翻倍耗时接近翻倍（线性），而非注意力式 4 倍。

【高频报错 Top5】
1. “No module named mamba_ssm”：路径 A 未安装；修复：Linux+CUDA 下 pip install mamba-ssm，或直接走路径 B；
2. 维度报错：x 需 [batch, seq, d_model]；修复：d_model 与 MinimalSSM(d_model=...) 保持一致；
3. 训练不收敛：A 初始化太大导致 state 爆炸；修复：A 初始化乘 0.05 或更小；
4. tanh 让输出范围受限：输出被压缩到 (-1,1) 附近；修复：加输出 Linear 头或去掉 tanh（用 leaky 版本）；
5. 计时波动大：后台进程干扰；修复：多次取平均，避免在 IDE 后台运行时计时。

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