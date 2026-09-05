# 【LLM全栈工程·第28章】vLLM 部署实战与参数调优：从启动到生产级

> 本篇为「LLM全栈工程」连载第 28 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① vLLM 不只是“跑起来”：核心参数逐项调优；② 显存利用率、并发上限、前缀缓存怎么配：vLLM 实战；③ 从一条命令到生产服务：vLLM 部署清单。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「基础部署与推理优化」第 2 篇：把 vLLM 从 Demo 级用到生产级 |
| 适用场景 | 个人学习（单卡起服务）；小团队上线对话服务；企业推理集群选参 |
| 核心知识点 | vLLM 核心机制；启动参数语义；并发/显存/前缀缓存调优；健康检查 |
| 技术选型 | 单卡/TP 多卡/Docker；OpenAI API 接入方式 |
| 分步实操 | 候选参数网格 → 逐个起服务 → 压测 → 汇总表 → 定生产配置 |
| 参数详解 | gpu-memory-utilization/max-model-len/max-num-seqs/max-num-batched-tokens/prefix caching |
| 踩坑排查 | OOM、max len 错配、并发上限过低、前缀缓存吃显存、模型名不一致 |
| 进阶优化 | 多 LoRA、PD 分离、投机解码、量化联动（后续章节展开） |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 29 章：SGLang 高性能推理落地 |

---

## 01 开篇导语

第 27 章把 vLLM 选为默认主力框架——因为它把“调度”做明白了：KV Cache 分页管理（PagedAttention）让显存不碎片化，连续批处理（Continuous Batching）让 GPU 一直有活干。

但“装好就能跑”和“生产级”之间，隔着一堆参数：显存利用率开多少？并发上限设多大？要不要开前缀缓存？这些参数直接影响 TTFT、吞吐与稳定性——调错了，同样的卡能差一倍性能。

本章从一条最小命令出发，逐项拆解核心参数，最后给一套“候选网格→压测→汇总→定稿”的调优流程。

> 一句话记住本章：**vLLM 调优 = 在“显存预算、并发上限、上下文长度、缓存策略”四个旋钮之间找平衡，并用压测数据定稿。**

---

## 02 白话原理：vLLM 的三个关键机制

### 2.1 PagedAttention：KV Cache 分页

传统推理为每条请求预分配整块 KV Cache，显存碎片严重；vLLM 像操作系统分页一样，把 KV 按块管理，需要时再分配——显存利用率大幅提升，能同时服务的请求更多。

### 2.2 Continuous Batching：有请求走就走

传统框架等一批请求全部生成完才换下一批；vLLM 每生成一个 token 就检查有没有新请求可以插入——GPU 不再空等，吞吐显著提升。

### 2.3 Prefix Caching：相同前缀直接复用

多条请求共享相同前缀（长 system prompt、RAG 上下文、多轮历史）时，vLLM 可以缓存并复用这些前缀的 KV，省掉重复 prefill。开启后对“长固定 prompt”场景收益巨大，但会占用额外显存做缓存。

> 记忆锚点：**PagedAttention 省显存、Continuous Batching 提吞吐、Prefix Caching 省重复计算——三个机制对应三类参数。**

## 03 一条生产级启动命令

以 helpdesk-sft-merged 为例（单卡 80GB/24GB 均可，按显存调参）：

~~~bash
vllm serve models/helpdesk-sft-merged \
  --served-model-name helpdesk \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 4096 \
  --enable-prefix-caching \
  --dtype auto
~~~

启动后验证：

~~~bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/health
curl http://localhost:8000/metrics | head -n 20
~~~

> 说明：部分参数名随 vLLM 版本演进（如 chunked prefill 的开关/尺寸字段），以你安装版本的 vllm serve --help 为准。上面是“语义正确”的生产基线。

---

## 04 核心参数逐个拆解

| 参数 | 作用 | 参考 | 调优逻辑 |
|---|---|---|---|
| --gpu-memory-utilization | 允许使用的显存比例 | 0.85–0.95 | 越高 KV 越多；太高易 OOM/无缓冲 |
| --max-model-len | 最大上下文长度 | 4096–131072 | 越长 KV 占用越大，压缩并发 |
| --max-num-seqs | 最大并发序列数 | 16–256 | 并发上限=吞吐与排队平衡 |
| --max-num-batched-tokens | 单批最多 token | 2048–8192 | 控制 prefill 块大小，配 chunked prefill |
| --enable-prefix-caching | 前缀 KV 缓存 | 视场景 | 共享前缀多就开，否则浪费显存 |
| --tensor-parallel-size | TP 卡数 | 1–8 | 卡间需 NVLink；不匹配会报错 |
| --dtype | 精度 | auto/bfloat16 | 与量化权重匹配 |
| --quantization | 量化方式 | awq/gptq/fp8… | 与模型权重配套（第 31 章） |
| --served-model-name | API 里的模型名 | 自定义 | 客户端必须一致 |
| --enforce-eager | 关 CUDA Graph | true/false | 调试用，生产默认关 |

### 4.1 四个最容易调错的组合

1. **max-model-len 与显存**：上下文要求 32K，但显存只够 8K 的 KV——启动会报错或疯狂 OOM；先算账再定长度；
2. **并发与排队**：max-num-seqs=8 时第 9 个请求只能排队，首字时延飙升；
3. **前缀缓存与显存**：开 prefix caching 会预留缓存空间，短 prompt 场景收益小、显存白占；
4. **TP 与互联**：tensor-parallel-size=4 但跨节点普通网卡，吞吐可能不升反降。

### 4.2 OpenAI 兼容调用

~~~python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="helpdesk",
    messages=[{"role": "user", "content": "打印机连不上怎么办？"}],
    stream=True,
    max_tokens=300,
)
for chunk in resp:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="")
~~~

> 结构化输出（JSON mode/工具调用）不同版本支持程度不同，生产使用前先查该版本文档并在评测集上验证格式。

## 05 分步实操：参数网格调优（约 1 小时）

### Step 1 生成候选启动命令（5 分钟）

保存 scripts/gen_vllm_candidates.py：

~~~python
# scripts/gen_vllm_candidates.py —— 输出 vLLM 参数候选命令
MODEL = "models/helpdesk-sft-merged"
NAME = "helpdesk"

candidates = [
    {"name": "base", "gpu": "0.90", "seqs": "64", "prefix": False},
    {"name": "prefix_on", "gpu": "0.90", "seqs": "64", "prefix": True},
    {"name": "seqs128", "gpu": "0.90", "seqs": "128", "prefix": True},
    {"name": "mem085", "gpu": "0.85", "seqs": "64", "prefix": True},
]

for i, c in enumerate(candidates):
    port = 8000 + i
    cmd = ["vllm serve", MODEL,
           "--served-model-name", NAME,
           "--port", str(port),
           "--max-model-len", "8192",
           "--gpu-memory-utilization", c["gpu"],
           "--max-num-seqs", c["seqs"],
           "--dtype", "auto"]
    if c["prefix"]:
        cmd.append("--enable-prefix-caching")
    print("# " + c["name"] + " -> http://localhost:" + str(port) + "/v1")
    print(" ".join(cmd) + " \\")
    print()
~~~

运行：

~~~bash
python scripts/gen_vllm_candidates.py
~~~

### Step 2 逐候选起服务并压测（40 分钟）

对每个候选：起服务（后台或另一终端）→ 用第 27 章压测脚本跑 → 记录后关掉，再起下一个：

~~~bash
# 候选 1：base
vllm serve models/helpdesk-sft-merged --served-model-name helpdesk --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 64 --dtype auto
python scripts/bench_frameworks.py base http://localhost:8000/v1 16 32
# 关掉后：候选 2 prefix_on（命令里加 --enable-prefix-caching）
python scripts/bench_frameworks.py prefix_on http://localhost:8000/v1 16 32
# ……依次跑 seqs128、mem085
~~~

> 压测提示：为了体现 prefix caching 差异，可把压测 PROMPT 改成“相同长 system+不同提问”，更能暴露缓存收益（第 27 章脚本可传参调整）。

### Step 3 汇总结果（5 分钟）

保存 scripts/collect_vllm_tune.py：

~~~python
# scripts/collect_vllm_tune.py —— 汇总 logs/bench_*.json 为对比表
import glob
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
rows = []
for path in sorted(glob.glob(str(root / "logs" / "bench_*.json"))):
    data = json.load(open(path, encoding="utf-8"))
    rows.append(data)

print("| 候选 | TTFT p95(ms) | 平均总时延(ms) | 近似吞吐(chars/s) |")
print("|---|---|---|---|")
for r in rows:
    print("| %s | %.1f | %.1f | %.1f |" % (r["name"], r["ttft_p95_ms"], r["total_avg_ms"], r["chars_per_sec"]))
~~~

运行：

~~~bash
python scripts/collect_vllm_tune.py
~~~

### Step 4 定稿生产配置（10 分钟）

1. 看吞吐拐点：seqs128 是否比 seqs64 明显提升？没有说明 GPU 已饱和，别再堆并发；
2. 看 TTFT：p95 是否满足业务（如 <1s）；
3. 看 prefix_on 与 base 的差距：差距大说明前缀复用场景多，保留缓存；差距小就关掉省显存；
4. 把胜出配置写进 deploy/vllm.prod.sh 并归档：

~~~bash
cd llm-demo
mkdir -p deploy
git add scripts logs deploy
git commit -m "vllm: tune grid (base/prefix/seqs/mem) -> prod config"
git tag vllm-tune-v1
~~~

> 重要提醒：压测通过只代表“性能达标”，上线前还要做稳定性验证（长稳压测、错误率、OOM 观察），那是第 34 章的内容。

## 06 参数详解：生产配置速查

| 场景 | 建议配置 |
|---|---|
| 24GB 单卡 7B（BF16） | gpu-memory-utilization 0.90，max-model-len 4096，max-num-seqs 16–32 |
| 80GB 单卡 7B | 0.90–0.92，max-model-len 8192–32768，max-num-seqs 64–128 |
| 长固定 system（RAG/Agent） | 开 enable-prefix-caching；system 保持完全一致才能命中 |
| 长输出任务 | 调大 max-num-batched-tokens/关注 chunked prefill |
| 多卡 7B | tensor-parallel-size=2~4（需 NVLink），单卡放不下时再用 |

> 每次只改一个参数并压测；把“参数组合+压测结果”写进实验台账，防止三个月后忘了为什么这么配。

---

## 07 高频踩坑排查

**坑 1：gpu-memory-utilization 设 0.99**
症状：跑一会就 CUDA OOM（驱动/上下文也要显存）。
解法：0.85–0.95 起步；观察 /metrics 里的显存使用再微调。

**坑 2：max-model-len 超过显存能力**
症状：启动报 “max model len ... larger than” 或运行中 OOM。
解法：先按第 12 章公式估算 KV 显存，再定长度；或开量化/降并发。

**坑 3：max-num-seqs 设太小**
症状：并发一高，请求全在排队，TTFT 爆炸。
解法：压测 16/32/64/128 找拐点；注意 CPU/调度也会随 seqs 上升。

**坑 4：前缀缓存开着但没收益**
症状：请求前缀各不相同，缓存命中率≈0，显存却少了一块。
解法：看 metrics 的 prefix cache hit rate；命中率低就关掉。

**坑 5：模型名不一致**
症状：客户端 404 model not found。
解法：--served-model-name 与客户端 model 完全一致；curl /v1/models 核对。

**坑 6：量化权重没配 quantization 参数**
症状：加载 AWQ/GPTQ 权重报错或结果异常。
解法：--quantization 与权重格式配套（第 31 章）；dtype 用 auto。

**坑 7：直接在生产跑未调优参数**
症状：上线后被真实流量打挂（并发/长度远超压测）。
解法：用线上真实流量分布压测后再定稿；留 10%–15% 显存余量。

---

## 08 进阶优化：vLLM 的进化方向

**① 多 LoRA 服务**：一个底座动态挂多个领域 adapter，按请求路由，省显存省管理（与第 14 章 adapter 策略呼应）。

**② Prefix Caching 深度使用**：把 system prompt/RAG 上下文固定成“规范前缀”，命中率可做到 90%+，长 prompt 场景吞吐翻倍。

**③ 投机解码**：vLLM 支持 speculative decoding（草稿模型），输出长文本时能再降时延（第 38 章展开）。

**④ 量化联动**：AWQ/GPTQ/FP8 权重 + vLLM 内核，显存减半、吞吐提升（第 31 章）。

**⑤ PD 分离/分布式**：生产大规模用 prefill-decode 分离架构与多实例编排（第 34/39 章），vLLM 的调度优势在集群层面继续放大。

---

## 09 本章核心总结（TOP3）

**TOP1**：vLLM 三机制对应三组参数：PagedAttention（显存/并发）、Continuous Batching（max-num-seqs/batched-tokens）、Prefix Caching（开关+前缀一致性）。

**TOP2**：启动基线 = gpu-memory-utilization 0.90 + max-model-len 按显存算 + max-num-seqs 压测定拐点 + prefix caching 按命中率开关；一次只改一个参数。

**TOP3**：生产定稿必须“压测+稳定性”双通过：记录版本与参数组合，用真实流量分布复测；上线前留显存余量并接 /metrics 监控（第 34 章）。

---

## 10 连载衔接

上一章（第 27 章）完成了框架选型；本章把默认主力 vLLM 调到了生产基线——启动命令、参数语义、压测网格全部就位。

下一章看 vLLM 的最强对手：【第 29 章】SGLang 高性能推理落地：RadixAttention 前缀复用、结构化输出与生产参数。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #vLLM #推理部署 #PagedAttention #参数调优 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 28 vLLM 部署实战与参数调优（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 29 章 SGLang 高性能推理落地*