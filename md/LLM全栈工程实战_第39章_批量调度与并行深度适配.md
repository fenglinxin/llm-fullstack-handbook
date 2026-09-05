# 【LLM全栈工程·第39章】批量调度与并行深度适配：continuous batching、PD 分离与 TP/PP/EP

> 本篇为「LLM全栈工程」连载第 39 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 单卡到多卡：推理并行策略 TP/PP/DP/EP 怎么选；② Prefill 和 Decode 为什么要分开？PD 分离架构；③ Continuous Batching 的调度艺术与压测方法。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「高阶推理编译器极致优化」第 5 篇：调度与集群并行，把单卡优化放大到多卡 |
| 适用场景 | 个人学习（batch 压测）；小团队多卡部署；企业大模型集群 |
| 核心知识点 | continuous batching；prefill/decode 不对称；PD 分离；TP/PP/DP/EP |
| 技术选型 | TP/PP/DP/EP/PD 组合决策 |
| 分步实操 | batch 阶梯压测 → 并行方案规划 → 多卡实测（TP vs DP）→ PD 评估与归档 |
| 参数详解 | max-num-seqs、chunked prefill、TP/PP/DP 规模、PD KV 传输 |
| 踩坑排查 | TP 跨普通网络、小模型过度并行、PP 小 batch 气泡、PD 传输开销 |
| 进阶优化 | MoE EP、disaggregated KV、chunked prefill 调优、路由器 PD |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 40 章：内核重构与编译级量化 |

---

## 01 开篇导语

单卡优化到顶后，性能增长来自两个方向：**同一时刻多塞请求（调度）**与**把模型/流量摊到多卡（并行）**。

调度侧的核心是 Continuous Batching：请求不是“整批进整批出”，而是谁完成谁走、新请求随时插入——GPU 永远不空等。并行侧则要回答：模型太大用 TP，流量太大用 DP，时延敏感用 PD，MoE 用 EP。

本章先讲调度，再讲并行，最后给决策脚本与压测流程。

> 一句话记住本章：**调度解决“单卡别闲着”，并行解决“多卡怎么分工”；prefill 吃算力、decode 吃带宽——把它们分开（PD）是生产级时延优化的关键一步。**

---

## 02 白话原理：先调度后并行

### 2.1 Continuous Batching

传统静态批：一批 32 个请求全部生成完才进下一批——最后几个慢请求拖住整批。连续批处理：每个 token 步后，完成的请求立刻退出、新请求立刻补入——平均排队时间大幅下降，吞吐提升。

### 2.2 Prefill 与 Decode 的不对称

| 阶段 | 特征 |
|---|---|
| Prefill（读题） | 计算密集：一次算完整段 prompt |
| Decode（答题） | 带宽密集：每步只产 1 token |

两种阶段挤在同一批里会互相拖累（长 prefill 让 decode 请求等很久）。**Chunked Prefill** 把大 prompt 切成块插入 decode 间隙；**PD 分离**则让 prefill 与 decode 各用一批实例。

### 2.3 PD 分离的代价

prefill 实例算完要把 KV 传给 decode 实例（网络传输），因此收益取决于：prefill/decode 时长比、KV 大小、网络带宽。长 prompt、高并发场景收益最大，短请求场景可能得不偿失。

> 记忆锚点：**Chunked Prefill 是“时间片穿插”，PD 分离是“流水线分工”；它们都在解决 prefill 与 decode 抢资源的问题。**

## 03 并行策略全家桶

| 策略 | 拆什么 | 通信 | 解决什么 |
|---|---|---|---|
| TP（张量并行） | 每层拆到多卡 | 每层 allreduce，极高 | 单卡放不下/单层太大 |
| PP（流水线并行） | 层分段放不同卡 | 段间传输，较低 | 多节点大模型 |
| DP（数据并行） | 整模型复制、切流量 | 几乎无 | 吞吐扩容 |
| EP（专家并行） | MoE 专家分布到卡 | all2all | MoE 模型 |
| PD（阶段分离） | prefill/decode 分工 | KV 传输 | 时延与吞吐 |

### 3.1 选型四问

1. **模型单卡放得下吗？** 放得下→先 DP/调度；放不下→TP（节点内）或 PP（跨节点）；
2. **卡间是什么网络？** NVLink 可 TP；普通以太网优先 DP/PP；
3. **流量是时延敏感还是吞吐优先？** 时延敏感→PD/更小 batch；吞吐→DP+连续批；
4. **是 MoE 吗？** 是→EP 是核心，配合 TP。

### 3.2 组合示例

| 场景 | 推荐组合 |
|---|---|
| 7B 单机 4×80GB | DP×4 或 TP×2+DP×2（看流量） |
| 70B 单机 8×80GB | TP×8（NVLink） |
| 70B 跨 4 节点 | TP×4 + PP×4（或 TP×8+PP×2） |
| MoE 大模型 | EP + TP 组合 |
| 高并发低时延 | PD 分离 + DP |

> 铁律：**TP 只用在 NVLink 级互联；小模型别过度并行（通信开销可能大于收益）；并行方案要用压测验证，别按感觉组合。**

## 05 分步实操：调度压测与并行规划（约 1 小时）

### Step 1 并行方案助手（5 分钟）

保存 scripts/parallel_plan.py：

~~~python
# scripts/parallel_plan.py —— 推理并行方案建议
# 用法：python parallel_plan.py <模型B> <卡数> <单卡GB> <nvlink|eth> <moe:y/n> <latency|throughput>
import sys

model_b = float(sys.argv[1])
n_gpu = int(sys.argv[2])
vram = int(sys.argv[3])
net = sys.argv[4] if len(sys.argv) > 4 else "nvlink"
is_moe = (sys.argv[5] if len(sys.argv) > 5 else "n").lower() == "y"
traffic = sys.argv[6] if len(sys.argv) > 6 else "throughput"

weights_gb = model_b * 2  # BF16 权重
total_vram = n_gpu * vram * 0.8
print("模型权重约 %.0fGB，可用显存约 %.0fGB" % (weights_gb, total_vram))

if weights_gb > total_vram:
    print("单机放不下：需要多节点 PP 或量化；TP 仅在节点内")
elif is_moe:
    print("推荐：EP 为核心，配合 TP；MoE 的 all2all 需要高速互联")
elif n_gpu == 1:
    print("单卡即可：先连续批处理与 PD 之外的单机优化")
elif net == "nvlink" and weights_gb > vram * 0.8:
    print("推荐：TP=%d（模型放不下单卡且节点内互联好）" % n_gpu)
elif traffic == "latency":
    print("推荐：PD 分离（如框架支持）+ 小 batch；TP 按模型大小")
else:
    print("推荐：DP=%d 扩容吞吐；模型放不下时节点内 TP、跨节点 PP" % n_gpu)
~~~

试算：

~~~bash
python scripts/parallel_plan.py 7 4 80 nvlink n throughput
python scripts/parallel_plan.py 70 8 80 nvlink n latency
python scripts/parallel_plan.py 70 32 80 eth n throughput
~~~

### Step 2 Batch 阶梯压测（15 分钟）

保存 scripts/batch_sweep.py：

~~~python
# scripts/batch_sweep.py —— 连续批处理阶梯压测
# 用法：python batch_sweep.py <名称> <base_url>
import concurrent.futures
import json
import pathlib
import statistics
import sys
import time

from openai import OpenAI

name = sys.argv[1]
base_url = sys.argv[2]
root = pathlib.Path(__file__).resolve().parent.parent
LEVELS = [1, 4, 16, 64]
PROMPT = "请分点说明打印机连不上时的处理步骤。"

def run_level(concurrency, total=24):
    def once(_):
        client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=120)
        t0 = time.time()
        resp = client.chat.completions.create(
            model="helpdesk", messages=[{"role": "user", "content": PROMPT}],
            max_tokens=150, temperature=0.2, stream=False)
        return time.time() - t0
    for _ in range(2):
        once(None)
    times = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for t in pool.map(once, range(total)):
            times.append(t)
    return {"concurrency": concurrency, "avg_ms": round(statistics.mean(times) * 1000, 1),
            "p95_ms": round(sorted(times)[int(len(times) * 0.95)] * 1000, 1),
            "req_per_min": round(60.0 / statistics.mean(times), 1)}

rows = [run_level(c) for c in LEVELS]
print("| 并发 | avg ms | p95 ms | req/min |")
print("|---|---|---|---|")
for r in rows:
    print("| %d | %.1f | %.1f | %.1f |" % (r["concurrency"], r["avg_ms"], r["p95_ms"], r["req_per_min"]))
with open(root / "logs" / ("batch_sweep_" + name + ".json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
~~~

运行并读拐点：

~~~bash
python scripts/batch_sweep.py single http://localhost:8000/v1
~~~

**判读**：req/min 增长变平=实例饱和；p95 暴涨=排队严重；拐点处的并发就是该实例的“甜点区”。

### Step 3 多卡实测：TP vs DP（20 分钟，需要 ≥2 卡）

**TP=2（模型拆到 2 卡）**：

~~~bash
vllm serve models/helpdesk-sft-merged --served-model-name helpdesk-tp2 --tensor-parallel-size 2 --port 8001
python scripts/batch_sweep.py tp2 http://localhost:8001/v1
~~~

**DP=2（两台单卡实例 + 网关轮询）**：起两个单卡 vLLM（8002/8003），Nginx 轮询到 8004，压测：

~~~bash
python scripts/batch_sweep.py dp2 http://localhost:8004/v1
~~~

对比表：

| 方案 | req/min 甜点 | p95 | 说明 |
|---|---|---|---|
| 单卡 | ? | ? | 基线 |
| TP=2 | ? | ? | 单请求更快？ |
| DP=2 | ? | ? | 吞吐更高？ |

**典型结论**：小模型（7B）单卡放得下时，DP 通常比 TP 吞吐更好（TP 通信开销）；模型单卡放不下时才必须 TP。**用你的数据验证，不要照搬。**

### Step 4 PD 分离评估（10 分钟）

PD 分离（prefill/decode 分工）依赖框架支持与网络条件：

1. 查所用框架官方“disaggregated serving/PD”文档与版本要求；
2. 支持则部署 prefill 池 + decode 池，用长 prompt+高并发压测对比 TTFT/TPOT；
3. 不支持则用 chunked prefill + 连续批处理近似（vLLM 的 max-num-batched-tokens）；
4. 记录架构与收益：

~~~bash
cd llm-demo
git add scripts logs
git commit -m "sched: parallel plan + batch sweep + TP/DP compare"
git tag sched-parallel-v1
~~~

> PD 不是银弹：KV 网络传输可能吃掉收益；先小规模 PoC 再全量。

## 06 参数详解：调度与并行速查

| 参数 | 参考 | 说明 |
|---|---|---|
| max-num-seqs | 甜点区×1.2~1.5 | 用 batch_sweep 找 |
| max-num-batched-tokens | 2048–8192 | chunked prefill 分块 |
| TP size | 模型放不下时 2–8 | 需 NVLink |
| DP size | 按吞吐需求 | 整模型复制 |
| PP size | 多节点大模型 | 气泡随 batch 增大减少 |
| EP size | MoE 专家数相关 | all2all 互联 |

---

## 07 高频踩坑排查

**坑 1：TP 用在普通网络**
症状：跨节点 TP 通信比计算还慢。
解法：TP 限 NVLink 节点内；跨节点用 PP/DP。

**坑 2：小模型过度并行**
症状：7B 拆 TP=8，吞吐不升反降。
解法：单卡放得下就 DP/调度优先；并行按压测定。

**坑 3：PP 小 batch 气泡**
症状：流水线空闲气泡吃掉收益。
解法：PP 配大 batch/多请求；小 batch 用 TP。

**坑 4：PD 传输开销被忽略**
症状：KV 网络传输比节省的算力还贵。
解法：先小规模 PoC；长 prompt 高并发才划算。

**坑 5：静态批没换连续批**
症状：整批进出，慢请求拖全队。
解法：确认框架开 continuous batching（vLLM 默认）。

**坑 6：MoE 不用 EP**
症状：MoE 模型纯 TP/DP，专家负载不均。
解法：EP 分布专家；all2all 需要高速互联。

**坑 7：只测平均不测拐点**
症状：并发加到 256 全在排队，均值还行。
解法：batch_sweep 看 req/min 拐点与 p95。

---

## 08 进阶优化：调度与并行进化方向

**① Disaggregated KV**：PD 分离后 KV 高效传输/缓存，长 prompt 场景 TTFT 与 TPOT 双优。

**② MoE EP 深化**：专家负载均衡路由、EP+TP 组合、跨节点 all2all 优化。

**③ Chunked Prefill 自调优**：按请求长度分布自动切 chunk，让 decode 间隙被完美利用。

**④ 路由级 PD**：网关把长 prompt 请求路由到 prefill 池、短请求直接 decode——混合架构。

**⑤ 集群调度与装箱**：多模型共享 GPU 池，按模型/流量动态装箱（第 43 章），利用率再上一个台阶。

---

## 09 本章核心总结（TOP3）

**TOP1**：调度解决“单卡别闲着”：continuous batching 是基础，chunked prefill 穿插长短请求，PD 分离是生产级时延优化的关键。

**TOP2**：并行选型四问：放不放得下（TP）、什么网络（NVLink 才 TP）、时延还是吞吐（PD/DP）、是不是 MoE（EP）；小模型别过度并行。

**TOP3**：用 batch_sweep 找甜点区、用 TP vs DP 实测做决策；PD 先 PoC 验证 KV 传输收益再全量。

---

## 10 连载衔接

上一章（第 38 章）优化了 decode 单步；本章把调度与并行放大到整机/集群——连续批、PD、TP/PP/DP/EP 的决策树都齐了。

下一章是编译器阶段的最后一战：【第 40 章】内核重构与编译级量化：硬件指令集与极致优化总纲。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #ContinuousBatching #PD分离 #张量并行 #MoE #推理优化 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 36 算子融合与计算图编译实战
- 37 动态 shape 与显存编译器优化
- 38 Speculative Decoding 投机推理进阶
- 39 批量调度与并行深度适配（本篇）
- 40 内核重构与编译级量化

第 5 阶段·全栈工程联调与项目落地
- 41 全链路串联：数据到上线一体化流程
- 42 领域大模型定制化落地
- 43 高并发生产适配与端侧部署
- 44 模型迭代升级与性能对标评测
- 45 线上问题闭环排查
- 46 工程化最佳实践汇总（手册终章）

---

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 40 章 内核重构与编译级量化*