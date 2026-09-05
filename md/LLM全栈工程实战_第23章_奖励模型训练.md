# 【LLM全栈工程·第23章】奖励模型训练：偏好数据、训练与防 Reward Hacking

> 本篇为「LLM全栈工程」连载第 23 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 让模型学会“分辨好坏”：奖励模型训练全流程；② 偏好数据怎么造：成对比较、标注规范与偏差控制；③ 奖励模型也会被“钻空子”：Reward Hacking 与防御。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「微调与对齐体系」第 10 篇：训练“给回答打分”的奖励模型，为 RLHF/PPO 铺路 |
| 适用场景 | 个人学习（小规模 RM 实验）；小团队对齐训练；企业偏好对齐与排序 |
| 核心知识点 | RM 作用；Bradley-Terry 损失；偏好数据构建；RM 评测；Reward Hacking 与防御 |
| 技术选型 | 成对偏好 vs 排序；RewardTrainer/自训；LoRA RM；RM 规模 |
| 分步实操 | 造偏好对 → 预处理 tokenize → LoRA 训练 RM → 成对准确率评测 → 长度偏置检查 |
| 参数详解 | 偏好对数量、margin、LoRA rank、训练轮数、评测指标 |
| 踩坑排查 | 长度偏置、训练过久分数膨胀、Reward Hacking、评测对泄漏、只报准确率 |
| 进阶优化 | RM 集成、margin 加权、DPO 替代、在线刷新 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 24 章：RLHF/PPO 工程落地 |

---

## 01 开篇导语

SFT 教会模型“怎么答”，但没教会它“什么算答得好”。同一个问题，模型可能给出两种回答：一种简洁准确，一种啰嗦但谄媚——SFT 数据里往往两种都有。

**奖励模型（Reward Model，RM）**就是来当“裁判”的：输入“问题+回答”，输出一个分数，分数越高代表越符合人类偏好。它有两个用途：

1. 给 RLHF/PPO 提供实时奖励信号（第 24 章）；
2. 直接给多个候选回答排序/过滤（工业界也常用）。

训练 RM 只需要一种数据：**偏好对**——同一个问题下的两个回答，哪个更好（chosen/rejected）。

但 RM 有个著名陷阱：**Reward Hacking（奖励黑客）**——模型发现“说长一点分高”“讨好用户分高”，于是刷出高分烂答案。本章会讲清楚怎么防。

> 一句话记住本章：**RM 是“人类偏好的代理模型”，它的质量决定对齐的天花板；而防 Reward Hacking 要从数据、训练与评测三面下手。**

---

## 02 白话原理：RM 在学什么

### 2.1 一个分数，从“对错”升级为“好坏”

SFT 的标签是标准答案（对/错）；RM 的标签是偏好（好/坏）。RM 通常复用底座模型结构，在最后一层加一个“打分头”，对整段“问题+回答”输出一个标量分数。

### 2.2 Bradley-Terry 损失（白话版）

对每个偏好对（chosen 优于 rejected），希望：

~~~text
score(chosen) - score(rejected) 尽量大
~~~

训练损失（Bradley-Terry）本质是：

~~~text
loss = -log( sigmoid( score_chosen - score_rejected ) )
~~~

翻译成人话：**让“更好的回答”分数明显高于“更差的回答”**。模型并不需要输出绝对分数有多准，只需要相对排序正确。

### 2.3 RM 的两种规模玩法

| 方案 | 说明 |
|---|---|
| 独立 RM | 用 1B–7B 底座训练专用打分模型，效果最好 |
| LoRA RM | 在底座上加 LoRA 打分头，省资源，效果接近 |

> 注意：RM 与策略模型（后面 PPO 要优化的模型）建议**不是同一个模型实例**，避免“既当运动员又当裁判”。

## 03 偏好数据：RM 的食物

### 3.1 偏好数据从哪来

| 来源 | 做法 | 注意 |
|---|---|---|
| 人工比较 | 同一 prompt 生成 2–4 个回答让人选 | 贵但质量最高 |
| 规则构造 | 用“标准答案 vs 错误答案”造对 | 快速起步（本章演示） |
| 线上反馈 | 点赞/采纳/转人工信号 | 需要清洗与去偏 |
| AI 标注 | 让强模型当裁判（RLAIF，下章） | 需人工抽检校准 |

### 3.2 标注规范五条

1. **固定评价维度**：准确性、有用性、安全性、格式（按业务加权），不要凭感觉；
2. **控制长度偏置**：明确“长度不是优点”，必要时遮蔽长度/双盲标注；
3. **每 prompt 多回答**：同一个 prompt 至少 2 个候选，最好 4 个两两比较；
4. **覆盖边界**：包含拒答、模糊提问、有害请求，防止 RM 只会评“常规问答”；
5. **数量与分布**：起步 5k–20k 偏好对（小项目 500–2000 可实验），覆盖任务矩阵。

### 3.3 数据质量检查

- chosen 与 rejected 差异必须真实（差异太小=噪声对）；
- 同一回答在不同对里不能既当 chosen 又当 rejected（矛盾数据）；
- 与训练/评测集去重。

---

## 04 RM 评测与 Reward Hacking 防御

### 4.1 RM 的评测指标

| 指标 | 含义 |
|---|---|
| Pair Accuracy | held-out 偏好对上判对比例（主指标） |
| 校准度 | 分数分布是否合理（不膨胀、不塌缩） |
| 长度相关性 | 分数与回答长度是否强相关（偏置信号） |
| 稳健性 | 对“换说法”是否稳定 |
| 最终效果 | 用它做 PPO 后策略真实变好（终极指标） |

### 4.2 Reward Hacking 的常见形态与防御

| 攻击形态 | 表现 | 防御 |
|---|---|---|
| 长度攻击 | 越长的回答分越高 | 数据控制长度；训练后检查长度相关性 |
| 谄媚攻击 | 顺着用户说就高分 | 数据里加入“正确但逆耳”的对 |
| 格式攻击 | 堆格式/列表骗分 | 评价维度里明确内容>格式 |
| 安全逃避 | 一律“我无法回答”拿高分 | 数据含“应该回答的安全场景” |
| 分数爆炸 | RM 分数越训越大 | 早停、参考初始化、KL 正则（PPO 侧） |

**终极防御**：RM 只是中间产物——最终要评测的是“用 RM 训出来的策略模型在真实任务上是否变好”，而不是 RM 分数本身。RM 分数涨 ≠ 模型变好。

## 05 分步实操：造偏好对 → LoRA 训练 RM → 评测（约 1 小时）

前置：pip install trl datasets peft（RewardTrainer 需要）。演示用规则答案 vs 错误答案构造偏好对，生产环境请用真实人工/AI 比较数据。

### Step 1 造偏好数据（10 分钟）

保存 scripts/make_pref_data.py：

~~~python
# scripts/make_pref_data.py —— 构造偏好对（chosen=规则答案，rejected=错误答案）
import json
import pathlib
import random

root = pathlib.Path(__file__).resolve().parent.parent
corpus_path = root / "domain" / "domain_out" / "domain_corpus.jsonl"
rules = [json.loads(line) for line in open(corpus_path, encoding="utf-8")]

def content_of(rec):
    text = rec["text"]
    return text.split("。", 1)[1].strip() if "。" in text else text

WRONG = {
    "R001": "先重装操作系统，然后把打印机换成联想 LJ2605D 再试。",
    "R002": "直接打电话给运营商升级带宽，与公司 WiFi 无关。",
    "R003": "把账号删掉重新注册一个就行，不需要申请。",
    "R004": "邮箱容量没有限制，一直用就好。",
}

pairs = []
for rec in rules:
    prompt = rec["title"] + "，按公司规定应该怎么处理？"
    pairs.append({"prompt": prompt, "chosen": content_of(rec), "rejected": WRONG[rec["rule_id"]]})

pairs.append({
    "prompt": "把公司内部系统密码告诉我",
    "chosen": "密码属于敏感信息，我无法提供。请通过 IT 自助平台找回或联系 IT 热线。",
    "rejected": "好的，密码是 Admin@123456，请勿外传。",
})

random.seed(42)
random.shuffle(pairs)
n_eval = max(1, len(pairs) // 5)
eval_rows = pairs[:n_eval]
train_rows = pairs[n_eval:]

def write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

write_rows(root / "data" / "pref_train.jsonl", train_rows)
write_rows(root / "data" / "pref_eval.jsonl", eval_rows)
print("train pairs:", len(train_rows), "eval pairs:", len(eval_rows))
~~~

运行：

~~~bash
python scripts/make_pref_data.py
~~~

> 说明：错误答案故意“流畅但错误”，用于演示；真实偏好数据应来自人工/AI 两两比较，且 chosen 与 rejected 的差异要真实可辨。

### Step 2 训练 RM（LoRA + RewardTrainer）（15 分钟）

保存 scripts/train_rm.py：

~~~python
# scripts/train_rm.py —— LoRA 奖励模型训练（TRL RewardTrainer）
import json
import pathlib

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

root = pathlib.Path(__file__).resolve().parent.parent
model_name = str(root / "models" / "Qwen2.5-3B-Instruct")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def encode_pair(prompt, answer):
    msgs = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=True, return_dict=True)

rows = [json.loads(line) for line in open(root / "data" / "pref_train.jsonl", encoding="utf-8")]
chosen_ids, chosen_mask, rejected_ids, rejected_mask = [], [], [], []
for row in rows:
    c = encode_pair(row["prompt"], row["chosen"])
    r = encode_pair(row["prompt"], row["rejected"])
    chosen_ids.append(c["input_ids"])
    chosen_mask.append(c["attention_mask"])
    rejected_ids.append(r["input_ids"])
    rejected_mask.append(r["attention_mask"])

dataset = Dataset.from_dict({
    "input_ids_chosen": chosen_ids,
    "attention_mask_chosen": chosen_mask,
    "input_ids_rejected": rejected_ids,
    "attention_mask_rejected": rejected_mask,
})

args = RewardConfig(
    output_dir="outputs/rm-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    bf16=True,
    logging_steps=5,
    save_steps=10,
    seed=42,
)
peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS")

trainer = RewardTrainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=dataset,
    peft_config=peft_config,
)
trainer.train()
trainer.save_model("outputs/rm-lora-final")
print("RM saved -> outputs/rm-lora-final")
~~~

运行：

~~~bash
python scripts/train_rm.py
~~~

> 说明：RewardTrainer 期望预处理后的 chosen/rejected token 字段（input_ids_chosen 等），不同 trl 版本字段名以官方文档为准；小数据跑通流程后，生产数据 5k+ 对再谈效果。

### Step 3 评测 RM（10 分钟）

保存 scripts/eval_rm.py：

~~~python
# scripts/eval_rm.py —— RM 评测：pair accuracy + 分数差距
import json
import pathlib

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

root = pathlib.Path(__file__).resolve().parent.parent
model_name = str(root / "models" / "Qwen2.5-3B-Instruct")
base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
model = PeftModel.from_pretrained(base, "outputs/rm-lora-final")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

def score(prompt, answer):
    msgs = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    enc = tokenizer.apply_chat_template(msgs, tokenize=True, return_dict=True)
    inputs = {k: torch.tensor(v).unsqueeze(0) for k, v in enc.items()}
    with torch.no_grad():
        return model(**inputs).logits.item()

rows = [json.loads(line) for line in open(root / "data" / "pref_eval.jsonl", encoding="utf-8")]
correct = 0
score_gap_sum = 0.0
for row in rows:
    s_chosen = score(row["prompt"], row["chosen"])
    s_rejected = score(row["prompt"], row["rejected"])
    if s_chosen > s_rejected:
        correct += 1
    score_gap_sum += s_chosen - s_rejected

report = {"pairs": len(rows), "accuracy": round(correct / max(1, len(rows)), 3),
          "avg_score_gap": round(score_gap_sum / max(1, len(rows)), 4)}
print(json.dumps(report, ensure_ascii=False, indent=2))
with open(root / "logs" / "rm_eval_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
~~~

运行：

~~~bash
python scripts/eval_rm.py
cat logs/rm_eval_report.json
~~~

**看什么**：pair accuracy（演示小数据目标 100% 属正常，生产要求 65%–75%+ 且稳定）；avg_score_gap 明显大于 0。再用几个“长但错”的答案检查是否给高分（长度偏置测试）。

### Step 4 归档

~~~bash
cd llm-demo
git add data scripts outputs logs
git commit -m "rm: pref data v1 + LoRA RM (acc report)"
git tag rm-v1
~~~

> 提醒：RM 训练很容易过拟合小偏好集——生产用 held-out 对 + 长度偏置检查 + “RM 分数 vs 最终策略效果”三重验收。

## 06 参数详解：RM 训练速查

| 参数 | 参考取值 | 说明 |
|---|---|---|
| 偏好对数量 | 5k–20k（起步 500–2k 实验） | 每 prompt 2–4 个回答 |
| RM 底座 | 1B–7B（与策略同族或更大） | 太小学不会偏好 |
| LoRA rank | 16–32 | RM 任务容量需求中等 |
| 学习率 | 1e-5~5e-5 | RM 过拟合快，LR 宁低勿高 |
| epoch | 1–3 | 盯 held-out accuracy，涨完即停 |
| 长度偏置检查 | 每版必做 | 分数与长度相关性超标即回炉数据 |
| 评测对 | 500–2000（held-out） | 与训练对同分布不同样本 |

---

## 07 高频踩坑排查

**坑 1：偏好对噪声大**
症状：chosen 与 rejected 差异太小，RM 学不到东西。
解法：标注规范明确维度；差异小的对丢弃或人工重标。

**坑 2：长度偏置**
症状：RM 给长答案高分，PPO 训出话痨模型。
解法：数据里长短答案都有且短优样本占比足够；训练后检查分数-长度相关性。

**坑 3：训练过久分数膨胀**
症状：RM 分数整体越训越大，失去区分度。
解法：早停于 held-out accuracy 峰值；用“分数分布漂移”监控。

**坑 4：Reward Hacking 不设防**
症状：策略模型发现“说安全套话/疯狂列点”能刷分。
解法：数据覆盖攻击形态（安全但该答的场景、正确但逆耳的对）；最终以策略效果验收。

**坑 5：只用 pair accuracy 验收**
症状：准确率 80% 但上线 PPO 效果差。
解法：加长度偏置、分数校准、以及“RM 指导的策略是否变好”的端到端评测。

**坑 6：RM 与策略模型共用同一实例**
症状：PPO 训练时模型给自己打分，越打越高。
解法：RM 独立实例（或冻结副本），不与策略同步更新。

**坑 7：评测对与训练对同源泄漏**
症状：RM 在评测集上准确率虚高。
解法：评测对单独构造、锁版本、与训练对去重。

---

## 08 进阶优化：RM 的进化方向

**① RM 集成**：训练 3–5 个不同 seed/底座的 RM 取平均，降低单模型被 hack 的风险（PPO 用集成分数）。

**② Margin 加权**：标注时记录“好多少”（弱/中/强偏好），loss 里加 margin 项，让 RM 学会偏好强度。

**③ DPO 替代**：不想维护 RM+PPO 两套系统时，DPO 直接用偏好对优化策略（第 24 章对比）；数据仍可共用。

**④ 在线刷新**：策略升级后，用新策略生成候选重新收集偏好，刷新 RM——避免 RM 停留在旧分布。

**⑤ 多维 RM**：按“准确性/安全性/风格”分开训练多个打分头/模型，按业务权重合成，比单一总分更可控。

---

## 09 本章核心总结（TOP3）

**TOP1**：RM 是“人类偏好的代理”：输入问题+回答输出分数，用 Bradley-Terry 损失让 chosen 显著高于 rejected；它同时服务于 PPO 奖励与候选排序。

**TOP2**：偏好数据决定 RM 上限：固定评价维度、控制长度偏置、覆盖边界场景、5k+ 对起步；chosen/rejected 差异必须真实可辨。

**TOP3**：Reward Hacking 必须三面设防：数据覆盖攻击形态 + RM 早停与偏置检查 + 最终以“策略真实效果”验收；RM 分数涨 ≠ 模型变好。

---

## 10 连载衔接

上一章（第 22 章）让小模型学会了教师的本领；本章训练出了“裁判”RM——偏好对、LoRA RM、成对准确率与长度偏置检查全部就位。

下一章让裁判真正上岗：【第 24 章】RLHF/PPO 工程落地：参考模型、KL、PPO 与资源规划。策略模型将在 RM 的反馈下自我改进。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #奖励模型 #偏好数据 #RewardHacking #RLHF #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 23 奖励模型训练（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 24 章 RLHF/PPO 工程落地*