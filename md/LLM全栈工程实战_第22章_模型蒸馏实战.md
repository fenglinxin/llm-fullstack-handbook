# 【LLM全栈工程·第22章】模型蒸馏实战：大模型萃取小模型的完整流程

> 本篇为「LLM全栈工程」连载第 22 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 大模型太贵？把 7B 的能力蒸馏给 3B：完整流程；② 黑盒蒸馏与白盒蒸馏：怎么选、怎么做；③ 蒸馏不是“抄答案”：数据收集、质量过滤与效果验收。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「微调与对齐体系」第 9 篇：用大模型（教师）训练小模型（学生），平衡质量与成本 |
| 适用场景 | 个人学习（小模型蒸馏实验）；小团队降本部署；企业端侧/高并发场景 |
| 核心知识点 | 黑盒/白盒蒸馏；教师数据收集与过滤；学生训练；效果 gap 验收与迭代 |
| 技术选型 | API 教师（黑盒）vs 同词表 logits 蒸馏（白盒）；学生规模与底座选择 |
| 分步实操 | 造提示池 → 教师批量生成 → 质量过滤 → 学生 SFT → 师生同测对比 → 困难样本迭代 |
| 参数详解 | 教师采样温度、每提示样本数、蒸馏温度 T、KL 权重、学生超参 |
| 踩坑排查 | 把教师幻觉教给学生、提示分布偏、只在教师数据上评测、词表不一致硬做 logits 蒸馏 |
| 进阶优化 | 多教师、迭代蒸馏、logits+序列混合、蒸馏+量化组合 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 23 章：奖励模型训练 |

---

## 01 开篇导语

你有一个效果很好的 7B 领域助手，但老板说：**并发要翻 10 倍、单卡要跑 4 路、时延要砍一半**——怎么办？

直接换小模型？效果掉太多。硬扛大模型？成本爆炸。中间的解法是**模型蒸馏**：让大模型当“教师”，把自己会的东西教给一个小模型“学生”。

蒸馏不是让小学生背大学生的学习笔记，而是让教师**生成大量高质量示范**（黑盒蒸馏），或在训练时直接传递“思考概率分布”（白盒蒸馏），让学生学会教师的格式、风格与行为。

本章走完一条完整流水线：

**提示池 → 教师生成 → 质量过滤 → 学生训练 → 师生同测 → 困难样本迭代**

以主线工程为例：7B 帮助台助手（教师）→ 3B 学生模型，跑通全流程并量化 gap。

> 一句话记住本章：**蒸馏的本质是“用教师的输出分布训练学生”，质量在数据、上限在教师、验收在独立评测集。**

---

## 02 白话原理：两种蒸馏方式

### 2.1 黑盒蒸馏（离线生成式，最常用）

教师只暴露“输入→输出”接口（本地 vLLM 或云 API）：

1. 准备一批多样化的提示（问题/指令/场景）；
2. 让教师生成答案（可多次采样）；
3. 过滤质量后作为 SFT 数据训练学生。

优点：不要求同词表、实现简单、能蒸馏任何模型（包括闭源 API）。
缺点：只能学到“答案”，学不到教师内部的概率细节。

### 2.2 白盒蒸馏（Logits/KL 蒸馏）

教师与学生同词表、可本地加载时，训练学生不仅拟合标准答案（hard label），还拟合教师的 token 概率分布（soft label）：

~~~text
loss = α × CE(学生, 标准答案)
     + (1-α) × KL(学生分布, 教师分布 / T) × T²
~~~

- T（温度）把教师分布“软化”，暴露更多“次优但合理的候选”；
- KL 权重 α 控制跟老师还是跟标准答案；
- 适合同家族模型（如 Qwen2.5-7B → Qwen2.5-3B），词表一致才能对齐位置。

### 2.3 蒸馏到底转移了什么

主要转移：**输出格式、行文风格、行为边界（拒答/澄清）、任务结构**。知识的上限仍受学生容量限制——学生学不会教师“不知道但能推理出来”的深层能力时，效果 gap 是正常的。

> 记忆锚点：**黑盒蒸馏=看教师交的作业学；白盒蒸馏=连教师的草稿纸（概率）一起学。**

## 03 蒸馏流水线与数据质量

### 3.1 六步流水线

~~~text
① 提示池（覆盖任务/难度/边界）
   → ② 教师批量生成（temperature 0.7，可多次采样）
   → ③ 质量过滤（空答/幻觉/格式错/重复）
   → ④ 学生 SFT 训练（3B LoRA 或全参）
   → ⑤ 师生同测（同一独立评测集）
   → ⑥ 困难样本回流（学生答错的题再让教师生成）
~~~

### 3.2 提示池设计

| 维度 | 要求 |
|---|---|
| 任务覆盖 | 与线上任务矩阵一致（问答/JSON/拒答/多轮） |
| 难度分布 | 简单 40% / 中等 40% / 困难 20% |
| 边界样本 | 拒答、信息不足、模糊提问各 5%–10% |
| 数量 | 起步 500–2000 条提示；质量优先 |

### 3.3 教师输出的质量过滤（关键）

教师也会胡说，必须过滤：

| 检查 | 规则示例 |
|---|---|
| 空答/过短 | 输出 <10 字丢弃 |
| 格式合规 | 要求 JSON 的必须可解析 |
| 幻觉实体 | 关键实体不在知识库中→丢弃（领域蒸馏） |
| 重复 | 同提示多次采样去重（MinHash） |
| 越界 | 拒答类提示教师没拒→人工/规则修正 |

> 生产经验：**教师数据按“通过率”分层**，只把教师做对且稳定的样本教给学生；把教师自己的错误教给学生是蒸馏最常见的翻车点。

---

## 04 技术选型：学生模型与蒸馏方式

| 决策 | 推荐 | 说明 |
|---|---|---|
| 学生规模 | 教师的 1/2~1/10 | 7B→3B 是性价比常见组合 |
| 学生底座 | 与教师同家族优先 | 白盒蒸馏需要同词表 |
| 蒸馏方式 | 黑盒起步 | 简单、通用、先验证收益 |
| 何时上白盒 | 黑盒有瓶颈且同词表 | 通常能再追回几个点 |
| 训练方式 | 学生 LoRA 起步 | 数据量上万再考虑全参 |

> 验收标准：学生要达到教师效果的 85%–95%（按业务指标），同时时延/成本下降 50%+——**gap 在可接受范围且收益显著，蒸馏才算成功**。

## 05 分步实操：7B 教师 → 3B 学生（约 1.5–2 小时）

沿用 llm-demo。教师=第 15 章 helpdesk-sft（7B，服务在 8000）；学生=Qwen2.5-3B-Instruct（需下载，或换 1.5B）。

### Step 1 下载学生底座（10 分钟，如未下载）

~~~bash
cd llm-demo
hf download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct
# 或国内：modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir models/Qwen2.5-3B-Instruct
~~~

### Step 2 造提示池（5 分钟）

保存 scripts/make_prompts.py：

~~~python
# scripts/make_prompts.py —— 蒸馏提示池：领域 + 通用
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
out_path = root / "data" / "distill_prompts.jsonl"

rows = []
for line in open(root / "data" / "sft_dataset_v1.jsonl", encoding="utf-8"):
    row = json.loads(line)
    if row.get("input"):
        rows.append({"instruction": row["instruction"], "input": row["input"],
                     "task_type": row["task_type"], "difficulty": row["difficulty"]})

general = [
    ("请用一句话解释 Transformer 的注意力机制。", "general", "easy"),
    ("Python 的列表和元组有什么区别？", "general", "easy"),
    ("数据库索引为什么能加速查询？", "general", "medium"),
    ("梯度下降的 learning rate 太大通常会怎样？", "general", "medium"),
    ("写一个 Python 函数判断字符串是否为回文。", "general", "hard"),
    ("解释一下什么是 KV Cache，它节省了什么计算？", "general", "hard"),
]
for q, task, diff in general:
    rows.append({"instruction": "", "input": q, "task_type": task, "difficulty": diff})

with open(out_path, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("prompts:", len(rows), "->", out_path)
~~~

运行：

~~~bash
python scripts/make_prompts.py
~~~

### Step 3 教师批量生成（10–20 分钟）

保存 scripts/distill_collect.py：

~~~python
# scripts/distill_collect.py —— 黑盒蒸馏：教师批量生成答案
# 用法：python distill_collect.py <teacher_url> <teacher_model> [采样数]
import json
import pathlib
import sys

from openai import OpenAI

teacher_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1"
teacher_model = sys.argv[2] if len(sys.argv) > 2 else "helpdesk-sft"
samples = int(sys.argv[3]) if len(sys.argv) > 3 else 1

root = pathlib.Path(__file__).resolve().parent.parent
client = OpenAI(base_url=teacher_url, api_key="EMPTY")
prompts = [json.loads(line) for line in open(root / "data" / "distill_prompts.jsonl", encoding="utf-8")]

out_rows = []
for p in prompts:
    user_content = p["instruction"] + ("\n" + p["input"] if p["instruction"] else p["input"])
    for _ in range(samples):
        resp = client.chat.completions.create(
            model=teacher_model,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.7,
            max_tokens=400,
        )
        answer = resp.choices[0].message.content.strip()
        if len(answer) < 10:
            continue  # 空答过滤
        out_rows.append({"instruction": p["instruction"], "input": p["input"],
                         "output": answer, "task_type": p["task_type"],
                         "difficulty": p["difficulty"], "teacher": teacher_model})

out_path = root / "data" / "distill_teacher_data.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for row in out_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("teacher samples:", len(out_rows), "->", out_path)
~~~

运行：

~~~bash
python scripts/distill_collect.py http://localhost:8000/v1 helpdesk-sft 1
head -n 2 data/distill_teacher_data.jsonl
~~~

> 提示：教师服务 temperature 用 0.7 保留多样性；关键任务（JSON/拒答）建议 temperature 0.2 再生成一份，避免格式飘。质量过滤可复用第 20 章 eval 的 JSON 校验逻辑。

### Step 4 学生训练（20–40 分钟）

在 data/dataset_info.json 注册 distill_teacher_data（Alpaca 三列同前），保存 configs/distill_student.yaml：

~~~yaml
model_name_or_path: models/Qwen2.5-3B-Instruct
template: qwen
stage: sft
finetuning_type: lora
dataset_dir: data
dataset: distill_teacher_data
val_size: 0.1
cutoff_len: 1024
per_device_train_batch_size: 2
per_device_eval_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 3.0e-4
num_train_epochs: 4.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
seed: 42
eval_strategy: steps
eval_steps: 10
logging_steps: 5
save_steps: 20
lora_rank: 16
lora_alpha: 32
output_dir: outputs/student-3b-distill
~~~

~~~bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/distill_student.yaml
llamafactory-cli export configs/export_student.yaml   # 指向 outputs/student-3b-distill
~~~

### Step 5 师生同测（10 分钟）

起两个服务：教师 helpdesk-sft（8000）、学生（8001，--served-model-name student-3b），用第 20 章 eval_suite 跑同一份 eval_test：

~~~bash
python scripts/eval_suite.py helpdesk-sft http://localhost:8000/v1 data/eval_test.jsonl
python scripts/eval_suite.py student-3b http://localhost:8001/v1 data/eval_test.jsonl
~~~

填对比表：

| 指标 | 教师 7B | 学生 3B | gap |
|---|---|---|---|
| 领域题通过率 | ? | ? | ? |
| JSON/拒答任务 | ? | ? | ? |
| 首 token 时延（参考） | ? | ? | ? |
| 吞吐/成本（参考） | ? | ? | ? |

**判断**：gap 可接受（如 ≤10%）且成本/时延显著下降 → 蒸馏成功；否则进入 Step 6。

### Step 6 困难样本迭代（10 分钟）

1. 把学生答错的评测题收集成 fail_prompts.jsonl；
2. 让教师重新生成（可 temperature 0.3 生成更稳的答案）并人工确认；
3. 追加进 distill_teacher_data，重新训练学生；
4. 重复直到 gap 达标或连续两轮无改善（说明学生容量到顶，需换更大学生或白盒蒸馏）。

### Step 7 归档

~~~bash
cd llm-demo
git add data configs outputs scripts
git commit -m "distill: teacher 7B -> student 3B v1 (gap report)"
git tag distill-v1
~~~

> 生产提示：蒸馏数据要按月刷新（教师升级后重新生成），学生跟着教师版本走；记录 teacher 版本进数据 manifest。

## 06 参数详解：蒸馏速查

| 参数 | 参考取值 | 说明 |
|---|---|---|
| 教师采样温度 | 0.2（稳定）/0.7（多样） | 关键格式任务用低温 |
| 每提示采样数 | 1–4 | 采样越多越稳，成本越高 |
| 提示池规模 | 500–5000 条起步 | 覆盖任务矩阵 |
| 学生规模 | 教师 1/2~1/10 | 7B→3B 常用 |
| 白盒温度 T | 2–4 | 软化教师分布 |
| KL 权重 (1-α) | 0.3–0.7 | 同词表白盒时用 |
| 学生 LR | LoRA 2e-4~4e-4 | 比正常 SFT 略高可试 |
| 学生 epoch | 3–5 | 教师数据量大时 2–3 |

---

## 07 高频踩坑排查

**坑 1：把教师的幻觉教给学生**
症状：学生学会了教师编造的型号/政策。
解法：教师输出必须过“知识库实体校验+人工抽检”；教师答错的样本不要进训练。

**坑 2：提示池分布偏**
症状：全是简单问答，学生一遇困难/边界题就崩。
解法：提示池按任务矩阵与难度分层；包含拒答/模糊/多轮场景。

**坑 3：只在教师生成的数据上评测**
症状：学生“看起来和教师一样”，一上独立评测就露馅。
解法：评测用独立集（第 20 章 eval_test），且评测题不与蒸馏提示池重叠。

**坑 4：词表不同硬做白盒蒸馏**
症状：KL 对齐报错或位置错乱。
解法：白盒蒸馏只用于同词表同家族模型；跨词表用黑盒蒸馏。

**坑 5：学生太小硬塞**
症状：3B 学不会 70B 的推理，gap 巨大且不再下降。
解法：明确蒸馏转移的是格式/行为；推理能力 gap 大时换更大学生或保留大模型做复杂任务路由。

**坑 6：学生训练数据里混入教师旧版本**
症状：教师升级后学生还在学旧答案，行为滞后。
解法：蒸馏数据带 teacher 版本；教师升级后重新生成。

**坑 7：只做一次蒸馏**
症状：学生答错的题永远答错。
解法：困难样本迭代闭环：学生失败 → 教师重生成 → 追加训练，直到 gap 达标。

---

## 08 进阶优化：蒸馏的进化方向

**① 白盒 Logits 蒸馏**：同词表场景用 KL(学生||教师/T) 补充 hard label，通常比纯黑盒再追 2–5 个点；实现要点是温度软化与 α 平衡。

**② 多教师蒸馏**：领域教师（7B SFT）+ 通用教师（更大底座）按任务路由生成，学生各取所长；答案冲突时用规则/置信度仲裁。

**③ 迭代/自蒸馏**：学生超过教师一部分任务后，把学生的强项样本也加入蒸馏集（self-distillation），形成持续提升。

**④ 蒸馏+量化组合**：先 7B→3B 蒸馏，再对 3B 做 INT4/INT8 量化（第 31 章），端侧/高并发场景的终极性价比组合。

**⑤ 在线蒸馏**：训练学生时实时采样教师（同 batch），避免离线数据分布过期；成本高，适合大厂训练平台。

---

## 09 本章核心总结（TOP3）

**TOP1**：蒸馏转移的是格式/风格/行为边界，不是“变出教师没有的知识”；黑盒蒸馏通用简单，白盒 logits 蒸馏同词表时更优。

**TOP2**：流水线 = 提示池（覆盖矩阵）→ 教师生成（控温+多采样）→ 质量过滤（幻觉/格式/重复）→ 学生训练 → 独立评测 → 困难样本迭代；教师自己的错误绝不能进训练集。

**TOP3**：验收标准是“学生达到教师 85%–95% 效果 + 成本/时延显著下降”；gap 不达标先查提示池与过滤，再考虑白盒蒸馏或更大学生。

---

## 10 连载衔接

上一章（第 21 章）让模型能连续对话；本章给了模型“变小”的能力——主线工程现在可以按场景选择 7B 教师或 3B 学生，成本与质量开始有了调度空间。

下一章进入“好坏评判”训练：【第 23 章】奖励模型训练：偏好数据、训练与防 Reward Hacking。对齐时代的大门正式打开。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #模型蒸馏 #知识蒸馏 #小模型 #降本部署 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 22 模型蒸馏实战（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 23 章 奖励模型训练*