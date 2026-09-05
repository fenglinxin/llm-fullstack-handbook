# 【LLM全栈工程·第27章】部署框架横向对比：vLLM/SGLang/TensorRT-LLM/llama.cpp

> 本篇为「LLM全栈工程」连载第 27 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 模型训完怎么上线？四大推理框架一次比清；② vLLM、SGLang、TensorRT-LLM、llama.cpp 选型决策表；③ 别用错框架：在线对话/Agent/边缘/低时延场景怎么选。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「基础部署与推理优化」开篇：建立部署框架全景与选型方法 |
| 适用场景 | 个人学习（本地起服务）；小团队选型；企业推理架构评审 |
| 核心知识点 | 推理指标；四大框架原理与差异；场景化选型；基准测试方法 |
| 技术选型 | vLLM vs SGLang vs TensorRT-LLM vs llama.cpp（+MLX/TGI 等） |
| 分步实操 | 起 vLLM/llama.cpp 服务 → 并发压测脚本 → 填对比表 → 决策卡 |
| 参数详解 | 并发数、请求数、上下文长度、量化、批大小 |
| 踩坑排查 | 只测单请求、不 warmup、比不同量化、模型支持差异、版本错配 |
| 进阶优化 | 混合部署、框架路由、前缀缓存对比、LLMPerf 类标准 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 28 章：vLLM 部署实战与参数调优 |

---

## 01 开篇导语

模型训练好了、评测过了——接下来是“最后一公里”：把它变成稳定、快速、便宜的服务。推理框架就是干这个的。

市面上主流框架各有绝活：

- **vLLM**：PagedAttention + 连续批处理，生态最大，默认首选；
- **SGLang**：RadixAttention 前缀缓存，多轮/Agent/多轮共享前缀场景吞吐惊人；
- **TensorRT-LLM**：NVIDIA 深度优化引擎，低时延王者，但要“编译固化”；
- **llama.cpp**：单文件 GGUF，CPU/边缘/本地利器。

没有“最好的框架”，只有“最合适场景的框架”。本章给你完整的对比坐标和一套可复现的压测流程，避免“用 A 框架跑不过就换 B”的拍脑袋选型。

> 一句话记住本章：**选框架 = 先定场景（在线/Agent/边缘/极致时延）与硬件，再用同一模型、同一量化、同一压测脚本对比，最后用数据说话。**

---

## 02 白话原理：先看懂推理指标，再谈框架

### 2.1 四个关键指标

| 指标 | 含义 | 谁在乎 |
|---|---|---|
| TTFT | 首 token 时延 | 用户体验（开口快不快） |
| TPOT/ITL | 每输出 token 耗时 | 打字速度 |
| 吞吐 | tokens/s、req/s | 成本与容量 |
| 并发/排队 | 同时服务与等待 | 高并发场景 |

### 2.2 框架在优化什么

大模型推理的瓶颈主要在**显存带宽**（逐 token 生成）与**批处理效率**（prefill 阶段）。框架的差异本质是：

1. KV Cache 怎么管理（连续内存？分页？前缀复用？）；
2. 批处理调度（静态批 vs 连续批）；
3. kernel 与编译优化深度（通用 PyTorch vs 定制 CUDA engine）。

> 记忆锚点：**vLLM 赢在“调度”，SGLang 赢在“前缀复用”，TensorRT-LLM 赢在“内核固化”，llama.cpp 赢在“随处可跑”。**

## 03 四大框架逐个看

### 3.1 vLLM（通用默认）

- 核心：PagedAttention（KV Cache 分页管理）+ Continuous Batching（连续批处理）；
- 优点：模型支持最广、OpenAI 兼容 API、社区最大、文档全；
- 缺点：极致时延不如 TRT-LLM；前缀缓存能力弱于 SGLang（虽有 prefix caching，但共享前缀场景 SGLang 更强）；
- 适合：绝大多数在线服务的第一选择。

### 3.2 SGLang（前缀复用/Agent 场景）

- 核心：RadixAttention——把历史 KV 按前缀树缓存，多轮对话、多请求共享 system prompt、Agent 工具调用等场景命中率极高；
- 优点：高前缀复用下吞吐显著领先；原生支持结构化输出；
- 缺点：生态与模型支持面比 vLLM 略窄、迭代快需跟版本；
- 适合：长 system prompt、多轮、Agent、RAG 前缀共享类流量。

### 3.3 TensorRT-LLM（NVIDIA 极致优化）

- 核心：把模型编译成 NVIDIA 定制 engine，算子融合与 FP8/INT4 深度优化；
- 优点：同硬件下时延/吞吐常最优；生产级多卡支持；
- 缺点：需要构建 engine、模型/TRT 版本强绑定、迭代慢、仅 NVIDIA；
- 适合：固定模型、固定 shape 的极致性能场景（大厂在线推理）。

### 3.4 llama.cpp（边缘/本地/CPU）

- 核心：GGUF 量化格式 + 单文件运行，CPU/GPU 混合；
- 优点：部署极简、跨平台、内存占用低；
- 缺点：大并发吞吐与生态弱于前三者；
- 适合：个人本机、边缘盒子、CPU 兜底、快速 Demo。

### 3.5 补充：TGI / MLX / MLC-LLM

TGI（Hugging Face）适合 HF 生态托管；MLX 面向 Apple 芯片；MLC-LLM 面向移动/编译部署。按硬件选，不展开。

---

## 04 选型决策表

| 场景 | 首选 | 次选 | 理由 |
|---|---|---|---|
| 通用在线对话 | vLLM | SGLang | 生态+稳定性 |
| 长 system/多轮/Agent | SGLang | vLLM（开 prefix cache） | 前缀命中决定吞吐 |
| NVIDIA 极致低时延 | TensorRT-LLM | vLLM（量化后） | 引擎固化 |
| 边缘/CPU/本地 | llama.cpp | MLC-LLM | 轻量跨平台 |
| Apple 芯片 | MLX | llama.cpp | 原生优化 |
| HF 托管/快速实验 | TGI | vLLM | 生态顺手 |

**三条硬规则**：

1. 对比必须“同一模型+同一量化+同一硬件”；
2. 先跑业务真实流量分布（请求长度、并发、前缀复用率），再选框架；
3. 不要在生产同时维护三个框架——选一个主力 + 一个边缘备胎。

## 05 分步实操：同模型跨框架压测（约 1 小时）

### Step 1 框架决策助手（2 分钟）

保存 scripts/pick_framework.py：

~~~python
# scripts/pick_framework.py —— 按场景与硬件推荐框架
# 用法：python pick_framework.py <chat|agent|edge|latency> <nvidia|amd|cpu|apple>
import sys

scenario = sys.argv[1] if len(sys.argv) > 1 else "chat"
hw = sys.argv[2] if len(sys.argv) > 2 else "nvidia"

TABLE = {
    ("chat", "nvidia"): ("vllm", "生态最稳，通用在线首选"),
    ("agent", "nvidia"): ("sglang", "前缀缓存命中高，多轮/Agent 吞吐好"),
    ("latency", "nvidia"): ("tensorrt-llm", "引擎固化，同卡时延最优"),
    ("edge", "cpu"): ("llama.cpp", "GGUF 单文件，CPU 可跑"),
    ("edge", "apple"): ("mlx", "Apple 芯片原生优化"),
    ("chat", "amd"): ("vllm", "AMD 支持看版本，先用 vLLM 验证"),
}

rec = TABLE.get((scenario, hw), ("vllm", "默认用 vLLM 验证，再按压测数据调整"))
print("推荐:", rec[0])
print("理由:", rec[1])
~~~

试几个场景：

~~~bash
python scripts/pick_framework.py chat nvidia
python scripts/pick_framework.py agent nvidia
python scripts/pick_framework.py edge cpu
~~~

### Step 2 通用压测脚本（15 分钟）

保存 scripts/bench_frameworks.py：

~~~python
# scripts/bench_frameworks.py —— 并发压测：TTFT/总时延/近似吞吐
# 用法：python bench_frameworks.py <名称> <base_url> <并发> <总请求>
import concurrent.futures
import json
import pathlib
import statistics
import sys
import time

from openai import OpenAI

name = sys.argv[1]
base_url = sys.argv[2]
concurrency = int(sys.argv[3]) if len(sys.argv) > 3 else 8
total = int(sys.argv[4]) if len(sys.argv) > 4 else 32
root = pathlib.Path(__file__).resolve().parent.parent

PROMPT = "请按公司 IT 帮助台规范回答：打印机连不上，处理步骤是什么？请分点说明。"
MODEL = "helpdesk-sft"

def run_once(_):
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    start = time.time()
    first_ts = None
    chars = 0
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=200,
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        if first_ts is None:
            first_ts = time.time()
        delta = chunk.choices[0].delta.content
        if delta:
            chars += len(delta)
    end = time.time()
    return {"ttft": first_ts - start if first_ts else end - start,
            "total": end - start, "chars": chars}

# warmup
for _ in range(2):
    run_once(None)

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
    for res in pool.map(run_once, range(total)):
        results.append(res)

def pct(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]

report = {
    "name": name, "base_url": base_url, "concurrency": concurrency, "total": total,
    "success": len(results),
    "ttft_avg_ms": round(statistics.mean(r["ttft"] for r in results) * 1000, 1),
    "ttft_p50_ms": round(pct([r["ttft"] for r in results], 0.5) * 1000, 1),
    "ttft_p95_ms": round(pct([r["ttft"] for r in results], 0.95) * 1000, 1),
    "total_avg_ms": round(statistics.mean(r["total"] for r in results) * 1000, 1),
    "chars_per_sec": round(sum(r["chars"] for r in results) / max(1e-9, sum(r["total"] for r in results)), 1),
}
out = root / "logs" / ("bench_" + name + ".json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
~~~

> 说明：脚本输出“字符吞吐”作为近似（中文场景约 2 字符≈1 token，英文约 4 字符≈1 token）；正式评测建议用服务端 usage 统计精确 token 数（第 34 章给完整版）。

### Step 3 同模型多框架实测（30 分钟）

用第 15 章的 helpdesk-sft 合并模型做压测（保证“同模型同量化同硬件”）。

**vLLM（8000 端口）**：

~~~bash
vllm serve models/helpdesk-sft-merged \
  --served-model-name helpdesk-sft \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8000
python scripts/bench_frameworks.py vllm http://localhost:8000/v1 8 32
~~~

**llama.cpp（8001 端口，需先转 GGUF）**：

~~~bash
# 用 llama.cpp 官方转换脚本（路径以你的 clone 为准）
python llama.cpp/convert_hf_to_gguf.py models/helpdesk-sft-merged \
  --outfile models/helpdesk-sft-q8.gguf --outtype q8_0

llama.cpp/build/bin/llama-server -m models/helpdesk-sft-q8.gguf \
  -c 4096 --port 8001
python scripts/bench_frameworks.py llama-cpp http://localhost:8001/v1 8 32
~~~

**SGLang（可选，8002 端口）**：

~~~bash
python -m sglang.launch_server --model-path models/helpdesk-sft-merged \
  --port 8002 --mem-fraction-static 0.85
python scripts/bench_frameworks.py sglang http://localhost:8002/v1 8 32
~~~

**TensorRT-LLM**：需要 engine 构建（第 30 章），本章先不做横向对比，避免“没构建好就比”的无效结论。

### Step 4 填对比表并决策（10 分钟）

| 框架 | TTFT p50/p95 | 平均总时延 | 近似吞吐 | 部署复杂度 | 结论 |
|---|---|---|---|---|---|
| vLLM | ? | ? | ? | 低 | 默认候选 |
| llama.cpp | ? | ? | ? | 最低 | 边缘/本地候选 |
| SGLang（可选） | ? | ? | ? | 中 | Agent/前缀场景候选 |

**决策要点**：先看 TTFT 是否满足业务（如 <500ms），再看吞吐与成本；最后用第 34 章的完整监控确认稳定性。压测结论与业务流量不符时，以真实流量（前缀复用率、并发曲线、输出长度分布）为准重测。

### Step 5 归档

~~~bash
cd llm-demo
git add scripts logs
git commit -m "bench: vLLM vs llama.cpp vs SGLang (same model/quant)"
git tag deploy-framework-bench-v1
~~~

> 提醒：框架版本迭代快，benchmark 报告要记录框架版本与 GPU 型号，否则三个月后无法对比。

## 06 参数详解：压测配置速查

| 参数 | 参考取值 | 说明 |
|---|---|---|
| 并发数 | 1/8/32/64 阶梯 | 看拐点，别只看单值 |
| 总请求数 | 每档 30–100 | 太少不稳定，太多耗时 |
| 输入长度 | 业务真实分布（如 200–2000 token） | 别只用一句话 |
| 输出长度 | max_tokens 200–500 | 与业务对齐 |
| warmup | 2–5 请求 | 预热 CUDA/缓存 |
| 记录字段 | 框架版本/GPU/量化/上下文长度 | 可复现 |

---

## 07 高频踩坑排查

**坑 1：只测单请求**
症状：单请求时延漂亮，一上并发就崩。
解法：至少测 1/8/32 三档并发，画吞吐拐点。

**坑 2：不 warmup 直接测**
症状：第一次请求含 CUDA 初始化，TTFT 虚高。
解法：正式测量前 warmup 2–5 次。

**坑 3：不同量化互相比**
症状：vLLM 用 BF16、llama.cpp 用 Q4，比完说 llama.cpp 更快。
解法：同量化同精度才可比；量化对比在第 31 章单独做。

**坑 4：只比吞吐不比 TTFT**
症状：批处理吞吐高，但首字要 2 秒，聊天体验崩。
解法：TTFT/TPOT/吞吐三个指标一起看，按场景加权。

**坑 5：模型支持差异没查**
症状：某框架不支持你的模型架构/自定义结构，硬跑报错。
解法：先查官方支持列表，再决定框架。

**坑 6：TensorRT-LLM 没构建好就比**
症状：用默认 engine 或错误 shape 对比，结论失真。
解法：engine 用目标 shape/量化构建并验证后再测（第 30 章）。

**坑 7：压测流量与线上不符**
症状：压测全短请求，线上全是长文档，结论反转。
解法：录制线上真实请求分布回放（第 34/37 章）。

---

## 08 进阶优化：部署选型的进化方向

**① 混合部署**：边缘/低峰用 llama.cpp 小模型，高峰/复杂任务路由到 vLLM 大模型——成本与时延双优（第 43 章）。

**② 框架路由**：同一网关后面挂 vLLM 与 SGLang：前缀命中高的流量走 SGLang，其余走 vLLM；按指标动态路由。

**③ 标准压测**：用 LLMPerf 类工具+固定脚本+固定版本做季度基准，跟踪框架升级收益，避免每次手动比。

**④ 前缀复用率监控**：上线后统计“前缀缓存命中率”——命中率低时 SGLang 优势发挥不出来，选型结论要随流量演化。

**⑤ 量化与编译联动**：框架选择与量化深度绑定：TRT-LLM 配 FP8、llama.cpp 配 GGUF、vLLM 配 AWQ/GPTQ；先定量化再定框架，顺序不要反（第 31 章）。

---

## 09 本章核心总结（TOP3）

**TOP1**：四大框架各有所长：vLLM 调度稳、SGLang 前缀强、TensorRT-LLM 引擎快、llama.cpp 随处跑；默认 vLLM，特殊场景再换。

**TOP2**：选型必须“同模型同量化同硬件”做压测：TTFT/吞吐/并发拐点三张图，加上真实流量分布，而不是拍脑袋。

**TOP3**：生产只维护“一个主力+一个边缘备胎”；压测报告记录框架版本与 GPU；框架升级、量化变化都要重测。

---

## 10 连载衔接

上一章（第 26 章）完成了微调与对齐阶段；本章打开了部署阶段的大门——你知道了框架地图与选型方法。

下一章把默认选项吃透：【第 28 章】vLLM 部署实战与参数调优。从启动参数到并发调优，把 vLLM 用到生产级。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #vLLM #SGLang #TensorRT #llama.cpp #推理部署 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 27 部署框架横向对比（本篇）
- 28 vLLM 部署实战与参数调优
- 29 SGLang 高性能推理落地
- 30 TensorRT-LLM 固化与加速
- 31 模型量化实操
- 32 模型剪枝与蒸馏压缩
- 33 KV Cache 优化与上下文窗口拓展
- 34 流式推理封装与高并发服务化

第 4 阶段·高阶推理编译器极致优化
- 35 推理编译器核心原理
- 36 算子融合与计算图编译
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 28 章 vLLM 部署实战与参数调优*