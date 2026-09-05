# 【LLM全栈工程·第36章】算子融合与计算图编译实战：把零散 kernel 合成一个

> 本篇为「LLM全栈工程」连载第 36 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 为什么“算子越碎越慢”？融合实战与收益测量；② 从 RMSNorm 到 FlashAttention：LLM 融合全景；③ torch.compile、Triton 手写融合与框架内置融合怎么选。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「高阶推理编译器极致优化」第 2 篇：识别并落地算子融合 |
| 适用场景 | 个人学习（融合实验）；工程师性能调优；企业推理内核优化 |
| 核心知识点 | 融合收益来源；LLM 高频融合点；torch.compile/Triton/框架融合路径 |
| 技术选型 | 自动编译 vs 手写 Triton vs 框架内置（FlashAttention/SDPA） |
| 分步实操 | 融合基准实验 → Triton 手写融合 kernel → 数值校验 → 模型级接入与回归 |
| 参数详解 | BLOCK/num_warps/mode/autotune/attention 实现 |
| 踩坑排查 | 数值漂移、动态 shape、非 memory-bound 融合无效、attention 硬件不支持 |
| 进阶优化 | 融合注意力、persistent kernel、autotune、warp 特化 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 37 章：动态 shape 与显存编译器优化 |

---

## 01 开篇导语

上一章讲了编译器的心智模型。本章落到最实用的动作：**算子融合（Operator Fusion）**——把多个小 kernel 合并成一个。

为什么融合能快？两个原因：

1. **省启动**：10 个小 kernel = 10 次启动；1 个大 kernel = 1 次启动；
2. **省显存带宽**：中间结果不用写回显存再读出来，直接在寄存器/共享内存里接力。

LLM 里到处都是融合机会：RMSNorm+残差、QKV 投影、MLP 的 gate/up、注意力（FlashAttention 就是最著名的融合）、解码期的 GEMV 融合。

本章给你三层落地路径：**自动编译（torch.compile）→ 手写 Triton → 框架内置**，并用实验量化收益。

> 一句话记住本章：**融合的本质是“少启动、少搬数据”；先让编译器自动做，特殊热点再手写 kernel，最后别忘了质量回归。**

---

## 02 白话原理：融合为什么快

### 2.1 一个反例：三个小算子

~~~text
t1 = x * scale + bias     # kernel 1：读 x，写 t1
t2 = relu(t1)             # kernel 2：读 t1，写 t2
y  = t2 * weight          # kernel 3：读 t2，写 y
~~~

每一步都在“显存↔计算单元”之间搬一整份数据。三个 kernel 至少搬 3–4 轮；融合成一个 kernel 后，中间结果留在片上，只搬 1 轮输入与 1 轮输出。

### 2.2 什么时候融合收益最大

| 场景 | 收益 |
|---|---|
| 小张量/逐元素算子多 | 大（启动+带宽双省） |
| memory-bound 算子（Norm/激活） | 大（省带宽） |
| 大 GEMM 本身 | 小（已是计算瓶颈） |
| 动态 shape 频繁变化 | 可能被重编译抵消 |

> 记忆锚点：**融合对“小而碎”的算子最有效——LLM 的 decode 阶段正好是“小算子排队”的重灾区。**

## 03 LLM 高频融合点地图

| 融合点 | 原来 | 融合后 | 落地 |
|---|---|---|---|
| RMSNorm+残差 | 2–3 kernel | 1 | torch.compile/框架 |
| QKV 投影 | 3 个 GEMM | 1 个大 GEMM | 权重拼接/框架 |
| MLP gate/up | 2 GEMM+激活 | 融合 kernel | TRT-LLM/Triton |
| Attention | QK^T+mask+softmax+PV | FlashAttention 一类 | flash-attn/SDPA |
| Norm+量化 | 2–3 kernel | 1 | 量化部署（第 31 章） |
| Decode GEMV | 逐层小矩阵乘 | 批量化/融合 | 引擎内核优化 |

## 04 三条落地路径怎么选

| 路径 | 成本 | 收益 | 适合 |
|---|---|---|---|
| torch.compile | 低 | 中（自动融合常见组合） | 第一步必试 |
| 框架内置（SDPA/FlashAttention） | 极低 | 高（注意力大头） | 直接开 |
| 手写 Triton | 高 | 最高（针对热点） | 编译器覆盖不了时 |

**工程顺序**：先开 SDPA/FlashAttention（一行配置）→ torch.compile 自动融合 → profiling 找剩余热点 → 手写 Triton。

> 提醒：框架已内置融合（vLLM/TRT-LLM 内核）时，外层再 torch.compile 可能重复优化甚至冲突——先查框架文档。

## 05 分步实操：融合实验三层走（约 30 分钟）

### Step 1 融合基准实验（10 分钟）

保存 scripts/fusion_bench.py：

~~~python
# scripts/fusion_bench.py —— eager vs torch.compile vs Triton 融合对比
import time

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

def ref_op(x, scale, bias):
    return torch.relu(x * scale + bias)

def bench(fn, iters=50):
    for _ in range(5):
        fn()
    if x.is_cuda:
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn()
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.time() - start) / iters * 1000

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(1 << 20, device=device)
scale = 1.5
bias = -0.2

eager_ms = bench(lambda: ref_op(x, scale, bias))

compiled = torch.compile(ref_op)
compiled(x, scale, bias)  # 触发编译
compiled_ms = bench(lambda: compiled(x, scale, bias))

print("device:", device, "| eager: %.4f ms | compiled: %.4f ms | speedup %.2fx"
      % (eager_ms, compiled_ms, eager_ms / max(compiled_ms, 1e-6)))

if HAS_TRITON and x.is_cuda:
    @triton.jit
    def scale_bias_relu_kernel(x_ptr, out_ptr, scale, bias, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        v = tl.load(x_ptr + offs, mask=mask)
        y = tl.maximum(v * scale + bias, 0.0)
        tl.store(out_ptr + offs, y, mask=mask)

    def triton_op(x, scale, bias):
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        scale_bias_relu_kernel[grid](x, out, scale, bias, n, BLOCK=BLOCK)
        return out

    out_ref = ref_op(x, scale, bias)
    out_tri = triton_op(x, scale, bias)
    err = (out_ref - out_tri).abs().max().item()
    triton_ms = bench(lambda: triton_op(x, scale, bias))
    print("triton: %.4f ms | max err %.2e | vs eager %.2fx"
          % (triton_ms, err, eager_ms / max(triton_ms, 1e-6)))
else:
    print("triton 不可用或非 GPU：跳过手写 kernel 对比（语法与逻辑可先在 CPU 环境审阅）")
~~~

运行：

~~~bash
python scripts/fusion_bench.py
~~~

**看什么**：compiled 与手写 kernel 是否都快于 eager；max err 应在 1e-5 量级（融合改变浮点结合顺序，小误差正常）。

> 逐元素融合是“最小演示”，LLM 的真正大头在注意力与 decode GEMV——但收益逻辑完全相同：少启动、少搬数据。

### Step 2 模型级接入：先开注意力融合（10 分钟）

在加载 transformers 模型时直接选用融合注意力实现（支持情况以 transformers/硬件为准）：

~~~python
from transformers import AutoModelForCausalLM

# 方案 A：SDPA（PyTorch 原生融合注意力，通用）
model = AutoModelForCausalLM.from_pretrained(
    "models/helpdesk-sft-merged",
    attn_implementation="sdpa",
)

# 方案 B：FlashAttention（CUDA 专属，需 flash-attn 可用）
# model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="flash_attention_2")
~~~

再叠加 torch.compile：

~~~python
import torch
model = torch.compile(model, mode="reduce-overhead")
~~~

1. 用第 27 章 bench 对比“原始 vs sdpa vs sdpa+compile”的 TTFT/吞吐；
2. 跑 20–50 题质量回归（注意力实现不同会有数值微差）；
3. 若使用 vLLM/TRT-LLM 部署：它们已内置融合注意力，直接跳过本步。

### Step 3 找剩余热点（可选，10 分钟）

~~~python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    # 跑 5 次推理
    pass
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
~~~

**热点判断**：表格里“小而多”的 kernel（单次 <10us 但数量巨大）就是融合候选；优先处理 decode 路径的逐 token 小算子。

### Step 4 归档

~~~bash
cd llm-demo
git add scripts logs
git commit -m "fusion: bench + sdpa/compile integration + quality check"
git tag fusion-v1
~~~

> 原则：融合以 profiler 数据为准，不为融合而融合；每次融合改动都跑质量回归与整模型 bench。

## 06 参数详解：融合配置速查

| 参数 | 参考 | 说明 |
|---|---|---|
| attn_implementation | sdpa 优先，flash_attention_2 备选 | 看硬件与 transformers 版本 |
| torch.compile mode | reduce-overhead（GPU 显存足） | 叠加 CUDA Graph |
| BLOCK（Triton） | 256–1024 | 按张量大小调 |
| num_warps | 4–8 | Triton kernel 调优 |
| 编译预热 | 1–2 次 dummy | 避免首请求编译 |

---

## 07 高频踩坑排查

**坑 1：数值漂移**
症状：融合后个别回答不同。
解法：跑质量回归；误差大时关掉可疑融合/换实现。

**坑 2：动态 shape 重编译**
症状：每来一个新长度就编译一次。
解法：固定/分桶长度；或在推理框架层做（vLLM 自带）。

**坑 3：不是热点也硬融**
症状：融合了大 GEMM，收益≈0。
解法：先 profiler，只处理“小且多”的算子。

**坑 4：attention 实现硬件不支持**
症状：flash_attention_2 报错/回退。
解法：查 GPU 兼容性；用 SDPA 兜底。

**坑 5：手写 kernel 索引错**
症状：结果错或越界。
解法：先小尺寸数值对比（本章 max err 检查）；再放大。

**坑 6：与框架重复优化**
症状：vLLM 里再 torch.compile，性能不升反降。
解法：框架内置融合优先；自研编译只用于自有推理代码。

**坑 7：只看融合不看整模型**
症状：单 kernel 快 3 倍，端到端没变化。
解法：整模型 bench + profiler 验证占比。

---

## 08 进阶优化：融合的进化方向

**① FlashAttention 系**：分块在线 softmax、无中间大矩阵，长序列注意力显存与速度双优——LLM 最值钱的一次融合。

**② Persistent Kernel**：kernel 常驻、按块循环取任务，省掉反复 launch，适合 decode 小算子流。

**③ Autotune**：同一 kernel 多组 BLOCK/num_warps 自动选优（Triton autotune），一次编译多次受益。

**④ Warp 特化**：让不同 warp 分工（如一个算 GEMM、一个算注意力），H100 时代的新方向。

**⑤ 融合+量化协同**：Norm+量化、GEMV+反量化等组合 kernel，配合第 31 章量化权重进一步省带宽。

---

## 09 本章核心总结（TOP3）

**TOP1**：融合收益 = 省 kernel 启动 + 省中间张量搬运；对“小而碎”的 memory-bound 算子最有效，decode 阶段是重灾区。

**TOP2**：落地路径按成本递增：框架内置（SDPA/FlashAttention）→ torch.compile 自动融合 → profiler 找热点 → 手写 Triton；别与框架重复优化。

**TOP3**：每次融合都要“整模型 bench + 质量回归 + max err 校验”三件套；融合以 profiler 数据为准，不为融合而融合。

---

## 10 连载衔接

上一章（第 35 章）建立了编译器心智；本章把最实用的融合动作落了地——自动编译、手写 kernel 与模型级接入都有了可复现流程。

下一章解决编译器最头疼的问题：【第 37 章】动态 shape 与显存编译器优化。真实流量长度千变万化，怎么让编译不吃亏？

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #算子融合 #torch.compile #Triton #FlashAttention #编译优化 #工程实战**（Voice前沿 出品，欢迎收藏追更）

**系列目录（46 章，随连载持续更新）**

第 0 阶段·开篇引路
- 01 一条大模型生产线全程发生了什么
- 02 2 小时跑通最小闭环
- 03 预训练到底在练什么

第 1 阶段·预训练工程·数据核心层
- 04 数据管道从零构建
- 05 数据清洗规范与脏数据剔除
- 06 文本过滤实战
- 07 数据去重算法
- 08 敏感数据脱敏
- 09 数据质量打分体系
- 10 领域专属数据构建
- 11 增量预训练方案
- 12 预训练超参选型与硬件适配
- 13 断点续训、收敛判断与失败排查

第 2 阶段·微调与对齐体系
- 14 全维度微调技术拆解
- 15 SFT 监督微调实战
- 16 SFT 数据集构建与标注规范
- 17 微调模板设计
- 18 微调超参调优策略
- 19 过拟合、欠拟合与灾难遗忘
- 20 微调效果评估体系
- 21 多轮对话微调专项
- 22 模型蒸馏实战
- 23 奖励模型训练
- 24 RLHF/PPO 工程落地
- 25 RLAIF：AI 反馈自动对齐
- 26 小样本与领域自适应微调

第 3 阶段·基础部署与推理优化
- 27 部署框架横向对比
- 28 vLLM 部署实战与参数调优
- 29 SGLang 高性能推理落地
- 30 TensorRT-LLM 固化与加速
- 31 模型量化实操
- 32 模型剪枝与蒸馏压缩
- 33 KV Cache 优化与上下文窗口拓展
- 34 流式推理封装与高并发服务化

第 4 阶段·高阶推理编译器极致优化
- 35 推理编译器核心原理
- 36 算子融合与计算图编译实战（本篇）
- 37 动态 shape 与显存编译器优化
- 38 Speculative Decoding 投机推理进阶
- 39 批量调度与并行深度适配
- 40 内核重构与编译级量化

第 5 阶段·全栈工程联调与项目落地
- 41 全链路串联：数据到上线一体化流程
- 42 领域大模型定制化落地
- 43 高并发生产适配与端侧部署
- 44 模型迭代升级与性能对标评测
- 45 线上问题闭环排查
- 46 工程化最佳实践汇总（手册终章）

---

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 37 章 动态 shape 与显存编译器优化*