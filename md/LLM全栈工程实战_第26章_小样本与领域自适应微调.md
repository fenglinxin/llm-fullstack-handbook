# 【LLM全栈工程·第26章】小样本与领域自适应微调：冷启动方案

> 本篇为「LLM全栈工程」连载第 26 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 只有 100 条数据怎么微调？冷启动实战路线；② 先别急着训：少样本场景的四条路线对比；③ 小数据 + LoRA + 数据增强：领域助手的冷启动配方。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「微调与对齐体系」收官篇：数据少、算力少时的冷启动方法与决策框架 |
| 适用场景 | 个人学习（百条级数据）；小团队 PoC；企业新领域从 0 起步 |
| 核心知识点 | 冷启动决策树；少样本路线（提示/RAG/微调/领域预训练）；种子集质量；数据增强与防过拟合 |
| 技术选型 | 上下文少样本 vs LoRA/QLoRA vs 领域自适应预训练（DAPT） |
| 分步实操 | 造 50 条种子集 → 0-shot/3-shot 基线 → LoRA-50 训练 → 增强到 200 再训 → 四路线对比表 |
| 参数详解 | 种子集规模、增强倍数、LoRA rank、dropout、通用回放比例 |
| 踩坑排查 | 数据少硬上全参、增强注水、小验证集噪声、只训不对比、先微调后 RAG |
| 进阶优化 | DAPT 冷启动、迁移复用、主动学习、多任务预训练底座 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 27 章：部署框架横向对比（进入部署阶段） |

---

## 01 开篇导语

前面章节动不动就“准备 5k–20k 条数据”——但真实世界最常见的问题是：**我只有 100 条问答、1 张卡、2 周时间，怎么做领域助手？**

答案是：先别急着全参微调。冷启动的正确姿势是“由轻到重”走四条路线：

1. **提示/RAG 路线**：先不训练，用 prompt + 检索把知识喂进去；
2. **上下文少样本**：给模型 3–10 个示例再回答；
3. **LoRA/QLoRA 微调**：50–500 条高质量数据训“格式与行为”；
4. **领域自适应预训练（DAPT）**：有无标注领域文本时，先 CPT 再 SFT。

绝大多数冷启动项目，第 1–2 步就能解决 70% 需求；微调是用来解决“格式/行为必须稳定”的剩余 30%。本章给你完整的决策框架与可执行对比流程。

> 一句话记住本章：**冷启动 = 先用最便宜的手段验证需求，再用 50–500 条“高纯度种子数据”做轻量微调；数据质量比数量重要十倍。**

---

## 02 冷启动决策框架：先回答三个问题

### 2.1 三问决策树

| 问题 | 答案 | 路线 |
|---|---|---|
| 知识会变吗？ | 会（政策/价格频繁更新） | RAG，别微调 |
| 只改风格/格式？ | 是 | prompt + 少样本即可 |
| 需要稳定格式+私有知识？ | 是 | LoRA/QLoRA 微调 |
| 有大量无标注领域文本？ | 有 | 先 DAPT/CPT 再 SFT |

### 2.2 数据量分档建议

| 数据量 | 建议 |
|---|---|
| <50 条 | 先别微调：prompt/RAG/少样本验证 |
| 50–500 条 | LoRA/QLoRA + 高质量种子 + 适量增强 |
| 500–5000 条 | LoRA 主力，必要时全参精修 |
| 5k+ 且任务复杂 | 全参或 DAPT+SFT |

> 核心反直觉结论：**50 条高质量、覆盖任务矩阵的数据，常常比 5000 条重复同质数据更有用**——小数据拼的是“纯度”与“覆盖”，不是数量。

## 03 四条路线的工程要点

### 3.1 路线 A：提示/RAG（0 训练）

把领域知识放进 system/prompt 或检索结果。适合“知识会变、格式要求低”的场景。成本最低、可随时改知识，缺点是长知识塞不下、复杂格式不稳定。

### 3.2 路线 B：上下文少样本（0 训练，带示例）

每次请求带 3–10 个“问题-标准答案”示例。适合格式稳定的小场景，示例即“临时微调”。注意：示例越多 token 越贵；示例要与当前问题同分布。

### 3.3 路线 C：LoRA/QLoRA 微调（50–500 条种子）

小数据微调的成功配方：

1. **种子集高纯度**：宁可 50 条专家逐条改，不要 500 条“看起来对”；
2. **覆盖任务矩阵**：问答/格式/拒答都要有（第 16 章）；
3. **LoRA rank 8–16**：小数据别用大 rank，防过拟合；
4. **dropout 0.1 + 通用回放 10%–30%**；
5. **早停盯 val**：小数据 2–4 epoch 就到顶，别硬跑。

### 3.4 路线 D：DAPT（领域自适应预训练）

手头有几十万到几千万 token 的**无标注领域文本**（文档/工单原文）时，先用第 11 章 CPT 方法让底座“懂领域”，再用小标注集 SFT。DAPT 补“知识底色”，SFT 补“格式行为”，小标注集的压力会小很多。

| 场景 | 推荐组合 |
|---|---|
| 只有 100 条 QA | B 或 C（LoRA-50~100） |
| 100 条 QA + 大量领域文档 | D（DAPT）+ C（LoRA SFT） |
| 格式要求高、知识会变 | B + RAG |
| 预算充足、效果优先 | C 起步 → 数据扩到 5k → 全参 |

## 05 分步实操：四条路线对比（约 1.5 小时）

沿用 llm-demo：种子数据来自第 16 章 sft_dataset_v1（演示约 30 条，真实项目用 50–500 条专家数据）；评测用第 20 章 eval_test。

### Step 1 切种子集 + 模板增强（10 分钟）

保存 scripts/prepare_coldstart.py：

~~~python
# scripts/prepare_coldstart.py —— 冷启动：train/val 切分 + 模板增强
import json
import pathlib
import random

root = pathlib.Path(__file__).resolve().parent.parent
rows = [json.loads(line) for line in open(root / "data" / "sft_dataset_v1.jsonl", encoding="utf-8")]
rows = [r for r in rows if r.get("input")]

random.seed(2026)
random.shuffle(rows)
n_val = min(10, max(4, len(rows) // 5))
train_rows = rows[:len(rows) - n_val]
val_rows = rows[len(rows) - n_val:]

def write_rows(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 模板增强：只对 direct 类问题加问法前缀（保持语义）
PREFIX = ["请问：{}", "求助：{}", "{}，谢谢", "帮我看看：{}"]
aug = []
for row in train_rows:
    if row.get("task_type") == "direct_qa":
        for tpl in PREFIX:
            new_input = tpl.format(row["input"])
            if new_input != row["input"]:
                copy = dict(row)
                copy["input"] = new_input
                aug.append(copy)

write_rows(root / "data" / "cold_train.jsonl", train_rows)
write_rows(root / "data" / "cold_train_aug.jsonl", train_rows + aug)
write_rows(root / "data" / "cold_val.jsonl", val_rows)
print("train:", len(train_rows), "aug:", len(train_rows) + len(aug), "val:", len(val_rows))
~~~

运行：

~~~bash
python scripts/prepare_coldstart.py
~~~

> 评测集说明：cold_val.jsonl 是种子切出的验证问答；最终效果用第 20 章 eval_test（关键词/JSON 判分）对比，避免自评。

### Step 2 少样本基线评测脚本（10 分钟）

保存 scripts/coldstart_eval.py：

~~~python
# scripts/coldstart_eval.py —— 0-shot / N-shot 评测（OpenAI 兼容）
# 用法：python coldstart_eval.py <model> <url> [examples数]
import json
import pathlib
import sys

from openai import OpenAI

model = sys.argv[1]
base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000/v1"
n_examples = int(sys.argv[3]) if len(sys.argv) > 3 else 0
root = pathlib.Path(__file__).resolve().parent.parent
client = OpenAI(base_url=base_url, api_key="EMPTY")

cases = [json.loads(line) for line in open(root / "data" / "eval_test.jsonl", encoding="utf-8")]
train = [json.loads(line) for line in open(root / "data" / "cold_train.jsonl", encoding="utf-8")][:n_examples]

def ask(question):
    messages = [{"role": "system", "content": "你是公司 IT 帮助台助手，严格按公司规定回答。"}]
    for ex in train:
        user_text = ex["instruction"] + ("\n" + ex["input"] if ex.get("input") else "")
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": question})
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2, max_tokens=300)
    return resp.choices[0].message.content

passed = 0
for case in cases:
    answer = ask(case["question"])
    ok = False
    if case.get("expect_json"):
        try:
            obj = json.loads(answer)
            ok = bool(obj.get("steps"))
        except Exception:
            ok = False
    else:
        ok = all(k in answer for k in case.get("keywords", []))
    if ok:
        passed += 1

report = {"model": model, "examples": n_examples, "pass": passed, "total": len(cases)}
out = root / "logs" / ("cold_eval_" + model + "_ex" + str(n_examples) + ".json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(report)
~~~

### Step 3 训练两组 LoRA（30–60 分钟）

注册数据集 cold_train/cold_train_aug（Alpaca 三列），保存两个配置（差异只有 dataset 与输出目录）：

~~~yaml
# configs/cold_lora_50.yaml（小数据防过拟合配方）
model_name_or_path: models/Qwen2.5-7B-Instruct
template: qwen
stage: sft
finetuning_type: lora
dataset_dir: data
dataset: cold_train
val_size: 0.15
cutoff_len: 1024
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2.0e-4
num_train_epochs: 4.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
seed: 42
eval_strategy: steps
eval_steps: 5
logging_steps: 5
save_steps: 10
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.1
output_dir: outputs/cold-lora-50
~~~

cold_lora_aug.yaml 复制一份：dataset: cold_train_aug，output_dir: outputs/cold-lora-aug。然后：

~~~bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/cold_lora_50.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/cold_lora_aug.yaml
~~~

### Step 4 四路线对比（15 分钟）

~~~bash
# 基线：底座 0-shot 与 3-shot（同一服务）
python scripts/coldstart_eval.py base http://localhost:8000/v1 0
python scripts/coldstart_eval.py base http://localhost:8000/v1 3

# LoRA-50 与 LoRA-aug 模型（导出后各起服务，或依次评测）
python scripts/coldstart_eval.py cold-50 http://localhost:8001/v1 0
python scripts/coldstart_eval.py cold-aug http://localhost:8002/v1 0
~~~

填对比表：

| 路线 | eval_test 通过率 | 成本/耗时 | 结论 |
|---|---|---|---|
| 底座 0-shot | ? | 0 | 基线 |
| 底座 3-shot | ? | 仅 token | 提示工程收益 |
| LoRA-50 | ? | 低 | 微调收益 |
| LoRA-50+增强 | ? | 低 | 增强收益 |

**决策规则**：3-shot 已达标 → 先上线提示方案；LoRA 明显更好且格式稳定 → 上微调；增强版比原版好 → 保留增强管线。

### Step 5 归档

~~~bash
cd llm-demo
git add data configs outputs scripts logs
git commit -m "coldstart: 4-route compare (0/3-shot vs lora50 vs aug)"
git tag coldstart-v1
~~~

> 冷启动的正确结束姿势：不是“训完就完”，而是“用最少资源验证哪条路有效，并把数据管线留下”——后续数据增长时无缝切换到第 15–18 章的标准流程。

## 06 参数详解：冷启动速查

| 参数 | 参考取值 | 说明 |
|---|---|---|
| 种子集规模 | 50–500 条 | 覆盖任务矩阵即可，别贪多 |
| 增强倍数 | 2–5× | 超过 5× 边际收益快速下降 |
| LoRA rank | 8–16 | 小数据低 rank 防过拟合 |
| dropout | 0.1 | 小数据可加高 |
| epoch | 2–4 | 小数据 4 轮内到顶 |
| 通用回放 | 10%–30% | 保通用能力 |
| 少样本示例数 | 3–10 | 超过 10 收益递减且费 token |
| 验证集 | ≥20 条独立题 | 太少无法判断 |

---

## 07 高频踩坑排查

**坑 1：50 条数据上全参微调**
症状：把 50 条背得滚瓜烂熟，换题全崩。
解法：LoRA rank 8 + dropout 0.1 + 早停；全参等数据到 5k+ 再说。

**坑 2：增强=注水**
症状：把同一条数据改个标点算 10 条，模型没学到新东西。
解法：增强要换“问法/场景/表述”，不换答案事实；增强后做去重与质检。

**坑 3：小验证集噪声大**
症状：验证集 5 条，改个 seed 结论就反转。
解法：验证集 20 条+、多次切分或 Bootstrap；结论看趋势不看单次。

**坑 4：跳过提示/RAG 直接微调**
症状：知识每周更新，却训进参数里，改一次训一次。
解法：先按决策树走：知识会变→RAG；格式不稳定→再微调。

**坑 5：种子集质量不把关**
症状：50 条里有 10 条答案是编的，模型全学走。
解法：种子集必须专家逐条审（第 16 章规范），数量可以少，错不能有。

**坑 6：只对比“训前 vs 训后”**
症状：没比 3-shot 提示方案，微调收益被高估。
解法：四路线同表对比（0-shot/3-shot/LoRA/LoRA+增强），用同一评测集。

**坑 7：冷启动数据管线没留下**
症状：Demo 验证成功，正式数据来了却要从零搭流程。
解法：冷启动阶段就把“数据格式+训练配置+评测脚本”沉淀进仓库，后续只是换数据。

---

## 08 进阶优化：冷启动的进化方向

**① DAPT 冷启动**：有大量无标注领域文档时，先 CPT（第 11 章）再小样本 SFT——知识底色越厚，小标注集越省。

**② 迁移复用**：相似领域（如“IT 帮助台→HR 帮助台”）先拿旧模型/旧 adapter 初始化，再小样本适配，比从通用底座起步快。

**③ 主动学习选种子**：用模型对未标注池预测，挑“最不确定”的 100 条优先人工标注——同样的标注预算，覆盖更有效。

**④ 少样本评测先行**：正式微调前先用 3-shot 提示跑业务用例，收集“提示方案答不好的 badcase”，让种子集精准覆盖这些缺口。

**⑤ 数据飞轮**：冷启动上线后收集真实问答回流，按周扩充种子集——冷启动只是开始，数据会越滚越大（第 44 章迭代流程）。

---

## 09 本章核心总结（TOP3）

**TOP1**：冷启动四路线由轻到重：提示/RAG → 上下文少样本 → LoRA/QLoRA → DAPT+SFT；先用最便宜的手段验证需求，再决定要不要训练。

**TOP2**：小数据拼纯度不拼数量：50–500 条专家级种子 + 覆盖任务矩阵 + LoRA rank 8–16 + dropout 0.1 + 通用回放，比 5000 条重复数据有用。

**TOP3**：用同一评测集做“0-shot / 3-shot / LoRA-50 / LoRA+增强”四路线对比再决策；冷启动阶段就把数据管线沉淀下来，正式迭代时直接换数据。

---

## 10 连载衔接

上一章（第 25 章）让偏好数据可以规模化；本章给“只有 100 条数据”的团队指了条明路——至此，**微调与对齐体系（14–26）13 章全部收官**：从技术拆解、SFT、数据、模板、调参、拟合、评估、多轮、蒸馏、RM、PPO、RLAIF 到冷启动，模型能力层完整闭环。

下一阶段转向工程落地：【第 27 章】部署框架横向对比：vLLM/SGLang/TensorRT-LLM/llama.cpp。训练好的模型，终于要见用户了。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #小样本微调 #冷启动 #LoRA #领域自适应 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 26 小样本与领域自适应微调（本篇）

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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 27 章 部署框架横向对比*