# 【LLM全栈工程·第31章】模型量化实操：FP16/BF16/INT8/INT4 怎么选怎么用

> 本篇为「LLM全栈工程」连载第 31 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 显存不够先量化：GPTQ/AWQ/FP8 白话原理与实操；② INT4 会让模型变傻吗？量化质量评估方法；③ 从 BF16 到 INT4：精度、显存、吞吐的三方权衡。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「基础部署与推理优化」第 5 篇：量化选型、工具链与质量回归 |
| 适用场景 | 个人学习（单卡量化）；小团队降本；企业显存/吞吐优化 |
| 核心知识点 | 精度格式；权重量化 vs 激活量化；校准集；GPTQ/AWQ/FP8/GGUF；质量回归 |
| 技术选型 | BF16/INT8-W8A8/INT4-GPTQ-AWQ/FP8/GGUF 的适用矩阵 |
| 分步实操 | 造校准集 → AWQ INT4 量化 → vLLM 加载 → BF16 vs INT4 评测对比 → 归档 |
| 参数详解 | w_bit/group_size/zero_point/calib 样本数/量化版本 |
| 踩坑排查 | 校准分布错、小模型 INT4 掉点、量化后不评测、kernel 不支持、顺序搞反 |
| 进阶优化 | FP8、KV Cache 量化、量化感知训练、蒸馏+量化组合 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 32 章：模型剪枝与蒸馏压缩 |

---

## 01 开篇导语

7B 模型 BF16 权重约 14GB——24GB 卡能跑但余量不大；换成 INT4 后约 4GB，同样一张卡能塞更多上下文、更高并发，吞吐还能涨。

但“量化=白嫖性能”是错觉：它把连续浮点数压成少量离散档位，必然引入误差。误差大不大，取决于三点：

1. **量化方法**（GPTQ/AWQ/FP8 的误差控制差异很大）；
2. **校准数据**（用什么文本定缩放范围）；
3. **任务敏感度**（代码/数学/JSON 往往比闲聊更怕量化）。

本章讲清精度格式家族、主流方法原理，并给出一套“量化→部署→质量回归”的完整流程。铁律只有一条：**任何量化版本上线前，必须跑同一套评测对比。**

> 一句话记住本章：**量化是用“可控的精度损失”换“显存/吞吐收益”；校准集与质量回归决定损失可不可控。**

---

## 02 白话原理：精度格式家族

| 格式 | 位数 | 本质 | 典型用途 |
|---|---|---|---|
| FP32 | 32 | 训练主权重 | 训练内部 |
| BF16/FP16 | 16 | 半精度浮点 | 推理/训练基线 |
| INT8 | 8 | 定点整数（需 scale） | W8A8 加速（SmoothQuant 类） |
| FP8 | 8 | 8 位浮点 | Hopper+ 推理/训练 |
| INT4 | 4 | 超低位定点 | GPTQ/AWQ 权重量化 |

- **权重量化**：只把权重压到 INT4/INT8，激活仍高精度——显存收益大，速度收益看 kernel；
- **权重+激活量化（W8A8）**：计算也走 INT8——速度收益大，但更难做（激活有 outlier）；
- **Group Size**：每 128 个权重共享一个 scale，越小越准越贵。

> 记忆锚点：**INT4 主要省显存，W8A8 主要提速度，FP8 是“新硬件的平衡点”；量化误差主要在 outlier（异常大的权重/激活）上放大。**

## 03 主流量化方法与选型

### 3.1 方法速览

| 方法 | 类型 | 思路 | 适合 |
|---|---|---|---|
| GPTQ | INT4 权重量化 | 逐层用二阶信息补偿量化误差 | 离线一次性量化、通用 |
| AWQ | INT4 权重量化 | 按激活分布保护重要通道 | 通常比 GPTQ 更稳 |
| SmoothQuant | INT8 W8A8 | 把激活 outlier 迁移到权重 | 追求速度、有 kernel 支持 |
| FP8 | 8 位浮点 | 天然支持 Hopper+ | 新卡首选之一 |
| GGUF Q4_K_M | INT4（llama.cpp） | 分块量化 | 边缘/llama.cpp 生态 |

### 3.2 选型决策

| 场景 | 推荐 |
|---|---|
| 显存不够、单卡跑 7B | AWQ/GPTQ INT4（group 128） |
| 追求吞吐、新 NVIDIA 卡 | FP8 或 W8A8（看框架支持） |
| 边缘/CPU/llama.cpp | GGUF Q4_K_M/Q5_K_M |
| 质量敏感（代码/数学） | 先试 AWQ group 128，不行用 INT8/FP8 |
| 训练后微调 | 量化应在训练后做；QLoRA 训练是另一回事 |

### 3.3 质量回归三板斧

1. **同一评测集**：BF16 基线 vs 量化版跑第 20 章 eval_suite（领域题+通用题）；
2. **敏感任务专项**：数学计算、代码、JSON 格式、长文本各 20 题；
3. **量化噪声检查**：同 prompt 多次生成，观察是否出现乱码/重复/崩坏。

> 经验参考：7B 级模型 INT4 通常领域问答掉 0–3 个点、数学/代码可能掉 5+ 个点（因模型与任务而异）；掉点超红线就换更温和的量化档位。

## 05 分步实操：AWQ INT4 量化 + 质量回归（约 1 小时）

前置：pip install autoawq（版本以官方文档为准）。

### Step 1 造校准集（5 分钟）

保存 scripts/make_calib_prompts.py：

~~~python
# scripts/make_calib_prompts.py —— 量化校准集：领域+通用混合
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
prompts = []

# 领域：规则问题
for line in open(root / "domain" / "domain_out" / "domain_corpus.jsonl", encoding="utf-8"):
    rec = json.loads(line)
    prompts.append(rec["title"] + "，按公司规定怎么处理？")

# 领域：SFT 输入
for line in open(root / "data" / "sft_dataset_v1.jsonl", encoding="utf-8"):
    row = json.loads(line)
    if row.get("input"):
        prompts.append(row["instruction"] + "\n" + row["input"])

# 通用：中英混合（覆盖常见 token 分布）
general = [
    "请解释 Transformer 的注意力机制。",
    "Python 的列表和元组有什么区别？",
    "数据库索引为什么能加速查询？",
    "Explain the difference between TCP and UDP.",
    "Write a python function to check palindrome.",
]
prompts.extend(general)

# 去重并限长
seen = set()
rows = []
for p in prompts:
    if p not in seen and len(p) < 2000:
        seen.add(p)
        rows.append({"text": p})

with open(root / "data" / "calib_prompts.jsonl", "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("calib prompts:", len(rows))
~~~

运行：

~~~bash
python scripts/make_calib_prompts.py
~~~

> 校准集铁律：**必须贴近线上真实分布**（领域+格式+语种混合），否则缩放范围定错，量化质量崩。

### Step 2 AWQ 量化（10–20 分钟）

保存 scripts/quantize_awq.py：

~~~python
# scripts/quantize_awq.py —— AWQ INT4 量化
# 用法：python quantize_awq.py <模型目录> <输出目录> [校准条数]
import json
import pathlib
import sys

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = sys.argv[1]
quant_path = sys.argv[2] if len(sys.argv) > 2 else "models/helpdesk-awq-int4"
max_calib = int(sys.argv[3]) if len(sys.argv) > 3 else 128

calib = []
for line in open("data/calib_prompts.jsonl", encoding="utf-8"):
    calib.append(json.loads(line)["text"])
    if len(calib) >= max_calib:
        break

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.quantize(tokenizer, quant_config=quant_config, calib_data=calib)
model.save_quantized(quant_path)
print("AWQ INT4 saved ->", quant_path)
~~~

运行：

~~~bash
python scripts/quantize_awq.py models/helpdesk-sft-merged models/helpdesk-awq-int4 128
~~~

### Step 3 用 vLLM 加载 INT4 并评测（15 分钟）

~~~bash
# BF16 服务（8000）与 AWQ INT4 服务（8001）分别启动
vllm serve models/helpdesk-sft-merged --served-model-name helpdesk --port 8000 --max-model-len 4096
vllm serve models/helpdesk-awq-int4 --served-model-name helpdesk-int4 --port 8001 --max-model-len 4096 --quantization awq

# 质量回归（第 20 章 eval_suite）
python scripts/eval_suite.py helpdesk http://localhost:8000/v1 data/eval_test.jsonl
python scripts/eval_suite.py helpdesk-int4 http://localhost:8001/v1 data/eval_test.jsonl

# 性能对比（第 27 章 bench_frameworks）
python scripts/bench_frameworks.py bf16 http://localhost:8000/v1 8 32
python scripts/bench_frameworks.py int4 http://localhost:8001/v1 8 32
~~~

填表：

| 版本 | 领域题通过率 | 显存占用 | 吞吐 | 结论 |
|---|---|---|---|---|
| BF16 | ? | ? | ? | 基线 |
| AWQ INT4 | ? | ? | ? | ? |

**决策**：INT4 通过率掉点在红线内（如 ≤3%）且显存/吞吐收益明显 → 上线 INT4；掉点大 → 试 group 64、INT8/FP8 或换量化方法；**任何“看起来没掉点”都要用敏感任务（JSON/数学/代码）再验一次**。

### Step 4 GGUF 路线（可选，边缘用）

~~~bash
# llama.cpp 工具链
python llama.cpp/convert_hf_to_gguf.py models/helpdesk-sft-merged --outfile models/helpdesk-f16.gguf --outtype f16
llama.cpp/build/bin/llama-quantize models/helpdesk-f16.gguf models/helpdesk-q4km.gguf Q4_K_M
~~~

### Step 5 归档

~~~bash
cd llm-demo
git add data scripts logs
git commit -m "quant: AWQ INT4 (group128) + quality regression vs BF16"
git tag quant-awq-v1
~~~

> 生产提示：量化权重也是制品：记录校准集版本、量化工具版本、评测报告；模型微调升级后需要重新量化与回归。

## 06 参数详解：量化配置速查

| 参数 | 参考 | 说明 |
|---|---|---|
| w_bit | 4（主流）/8 | 越低越省显存，误差越大 |
| q_group_size | 128 起步 | 64 更准更慢；256 更省 |
| zero_point | True（对称量化可关） | AWQ/GPTQ 常用 True |
| 校准样本数 | 128–256 | 太少不稳定，太多耗时 |
| 校准集分布 | 领域+通用+格式混合 | 与线上一致 |
| 量化版本 | GEMM/GEMV | 按 kernel 支持选 |
| desc_act（GPTQ） | True/False | True 更准但慢 |

---

## 07 高频踩坑排查

**坑 1：校准集与线上分布不符**
症状：用新闻语料校准，线上全是代码/JSON，量化后崩。
解法：校准集采样自真实业务流量；每次业务分布变化重做校准。

**坑 2：小模型 INT4 掉点严重**
症状：3B 量化后明显变笨，7B 却还好。
解法：小模型先试 group 64/INT8/FP8；或先蒸馏再量化。

**坑 3：量化后不跑评测**
症状：perplexity 看着没变，一问业务题全崩。
解法：eval_suite 全量回归 + 敏感任务专项 + 生成噪声检查。

**坑 4：只看显存不看 kernel 支持**
症状：INT4 权重加载了，但推理 kernel 回退到慢路径。
解法：用框架支持的量化格式（vLLM 认 awq/gptq/fp8 等），跑 bench 验证收益。

**坑 5：先量化再微调/顺序混乱**
症状：量化后直接继续训练，误差被放大。
解法：训练用 BF16/QLoRA，训练完再量化部署；QLoRA 与部署量化是两件事。

**坑 6：GPTQ/AWQ 混用配置**
症状：AWQ 权重用 GPTQ 参数加载，结果错乱。
解法：权重格式、--quantization 参数、工具版本三者必须匹配。

**坑 7：只量化权重不量化 KV**
症状：长上下文显存还是不够。
解法：KV Cache 也可量化（INT8/FP8），与权重量化叠加（第 33 章）。

---

## 08 进阶优化：量化的进化方向

**① FP8 全面化**：Hopper+ 卡用 FP8 权重/激活，精度接近 BF16、吞吐高于 INT4 路线的常见选择。

**② KV Cache 量化**：长上下文场景对 KV 做 INT8/FP8 量化，显存再省 50%（第 33 章）。

**③ 量化感知训练（QAT）**：训练时就模拟量化误差，让模型“适应被量化”，比训练后量化（PTQ）更稳（成本高）。

**④ 蒸馏+量化组合**：先 7B→3B 蒸馏再 INT4（第 22 章），端侧/高并发性价比之王。

**⑤ 层敏感度自适应**：不同层对量化敏感度不同——敏感层用更高精度/更小 group，其余用 INT4，全局精度更优（进阶实验）。

---

## 09 本章核心总结（TOP3）

**TOP1**：精度家族按用途选：BF16 基线、INT4（GPTQ/AWQ）省显存、W8A8/FP8 提速度、GGUF 给边缘；权重量化与激活量化是两回事。

**TOP2**：量化质量由“方法+校准集+任务敏感度”决定：校准集必须贴近线上分布，group 128 起步，INT4 后必须跑同一套 eval_suite + 敏感任务专项。

**TOP3**：量化是制品不是一次性动作：记录校准集/工具版本/评测报告，模型升级后重新量化与回归；掉点超红线就换更温和档位。

---

## 10 连载衔接

上一章（第 30 章）把模型固化进 engine；本章给模型“减肥”——AWQ INT4 与质量回归流程就位，显存与吞吐有了新空间。

下一章继续压缩：【第 32 章】模型剪枝与蒸馏压缩：小模型化落地。量化动数值，剪枝动结构——两条路可以组合。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #模型量化 #INT4 #AWQ #GPTQ #FP8 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 31 模型量化实操（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 32 章 模型剪枝与蒸馏压缩*