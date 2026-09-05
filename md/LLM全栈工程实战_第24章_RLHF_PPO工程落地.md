# 【LLM全栈工程·第24章】RLHF/PPO 工程落地：参考模型、KL、PPO 与资源规划

> 本篇为「LLM全栈工程」连载第 24 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① SFT 是“教规矩”，RLHF 是“练偏好”：PPO 工程落地；② 为什么 PPO 要四个模型？RLHF 白话原理与资源规划；③ KL 惩罚、奖励黑客与训练不稳定：PPO 实战避坑。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「微调与对齐体系」第 11 篇：把第 23 章的 RM 用起来，完成 PPO 强化对齐 |
| 适用场景 | 个人学习（小模型 PPO 实验）；小团队对齐训练；企业 RLHF 平台规划 |
| 核心知识点 | RLHF 三段式；PPO 四模型架构；KL 约束；GAE/优势；资源规划与稳定性 |
| 技术选型 | TRL PPOv2 / OpenRLHF / verl；LoRA 策略；DPO/GRPO 何时替代 PPO |
| 分步实操 | 造 RL 提示集 → 资源规划 → 最小 PPO 训练 → 指标监控 → 前后评测与回滚 |
| 参数详解 | kl_coef、LR、batch、clip、GAE、response_length |
| 踩坑排查 | Reward Hacking、KL 失控、critic 不稳、OOM、策略发散、长度膨胀 |
| 进阶优化 | GRPO、PPO offload、OpenRLHF/verl、在线 RM 刷新 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 25 章：RLAIF：AI 反馈自动对齐 |

---

## 01 开篇导语

SFT 让模型“会按指令回答”，RM 让机器“能判断好坏”。现在把它们接起来：**让模型自己生成回答、让 RM 打分、用分数当反馈更新模型**——这就是 RLHF（人类反馈强化学习）的核心，主流实现是 PPO。

RLHF 不是必须的：数据少、任务简单时 SFT 就够；但当你要优化“开放性偏好”（回答更受欢迎、更安全、更少废话）时，RLHF 往往比堆 SFT 数据更高效。

本章回答四个工程问题：

1. PPO 为什么要四个模型、各自干什么；
2. KL 惩罚为什么必不可少；
3. 资源怎么规划（显存/卡数）；
4. 训练不稳定与 Reward Hacking 怎么防。

> 一句话记住本章：**PPO = “生成→打分→修正”的循环；RM 给方向，KL 拴缰绳，评测做刹车。**

---

## 02 白话原理：RLHF 三段式与 PPO 循环

### 2.1 RLHF 三段式

~~~text
SFT：让模型会说话（第 15 章）
  → RM：训练偏好裁判（第 23 章）
  → PPO：用裁判反馈优化策略
~~~

### 2.2 PPO 循环的每一步

对每个 prompt：

1. **Rollout（生成）**：当前策略模型生成回答；
2. **打分**：RM 给回答打分（reward）；
3. **计算 KL**：回答在“当前策略 vs 参考策略”下的概率差异；
4. **修正奖励**：final_reward = rm_score − β × KL——防止模型为了高分跑偏；
5. **更新**：PPO 用优势估计更新策略参数，让高分行为概率上升；
6. **循环**：新策略继续生成→打分→更新。

### 2.3 四个模型各司其职

| 模型 | 是否更新 | 作用 |
|---|---|---|
| Policy（策略） | ✅ | 被训练的主角 |
| Reference（参考） | ❌ 冻结 | 提供 KL 基准，防跑偏 |
| Reward（奖励） | ❌ 冻结 | 给回答打分 |
| Critic/Value（价值） | ✅（可选） | 估计状态价值，算优势 |

> 为什么需要参考模型：如果只追求 RM 高分，模型很快会“钻空子”（说长话、拍马屁、套模板）。KL 惩罚让新策略不敢离 SFT 策略太远——**KL 是防 Reward Hacking 的第一道锁。**

## 03 技术选型与资源规划

### 3.1 PPO 与替代方案怎么选

| 方案 | 需要 RM | 显存/算力 | 特点 | 适合 |
|---|---|---|---|---|
| PPO | ✅ | 高（四模型） | 在线采样、上限高、工程复杂 | 效果优先、有算力 |
| DPO | ❌ | 低（两模型） | 离线偏好直接优化、简单 | 中小团队首选入门 |
| GRPO | ✅/规则 | 中 | 组相对奖励、无 critic | 推理类任务流行 |
| RLAIF | 视实现 | 同 PPO/DPO | AI 生成偏好 | 下章展开 |

结论：**先 DPO 验证数据与偏好，效果不够再上 PPO；生产大模型对齐常用 PPO/GRPO 系。**

### 3.2 框架选型

| 框架 | 特点 | 适合 |
|---|---|---|
| TRL（PPOv2） | 与 HF 生态一体、上手快 | 小模型实验、单机多卡 |
| OpenRLHF | 高性能、支持 70B 级 | 生产 RLHF |
| verl | 字节开源，RL 训练框架 | 大规模、推理类 |

### 3.3 显存规划（先算账）

PPO 峰值要同时容纳：策略（+LoRA 状态）、参考、RM、Critic。粗略估算：

- 全参 7B 四模型 ≈ 4×16GB/模型量级 → 至少 8×80GB；
- **LoRA 策略 + 冻结参考/RM**：策略主权重可共享加载，参考与 RM 各一份权重；3B 级四件套在 4×24GB 或 2×48GB 可实验；
- 显存不够的降级路线：更小模型（1.5B/3B）→ CPU offload 参考/RM → 用 DPO 替代 PPO。

> 生产经验：**先在小模型上把 PPO 流程与超参跑通，再放大**——PPO 的不稳定源（KL、LR、reward scale）在小模型上同样会暴露。

## 05 分步实操：PPO 最小工程闭环（实验级）

### Step 1 造 RL 提示集（5 分钟）

保存 scripts/make_rl_prompts.py：

~~~python
# scripts/make_rl_prompts.py —— PPO rollout 提示集（领域+通用）
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
out_path = root / "data" / "rl_prompts.jsonl"

rows = []
for line in open(root / "data" / "sft_dataset_v1.jsonl", encoding="utf-8"):
    row = json.loads(line)
    if row.get("input"):
        prompt = row["instruction"] + ("\n" + row["input"] if row["instruction"] else row["input"])
        rows.append({"prompt": prompt, "task_type": row["task_type"]})

general = [
    "请用一句话解释 Transformer 的注意力机制。",
    "Python 的列表和元组有什么区别？",
    "写一个 Python 函数判断字符串是否为回文。",
]
for q in general:
    rows.append({"prompt": q, "task_type": "general"})

with open(out_path, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("rl prompts:", len(rows))
~~~

运行：

~~~bash
python scripts/make_rl_prompts.py
~~~

### Step 2 资源规划脚本（5 分钟）

保存 scripts/plan_rlhf.py：

~~~python
# scripts/plan_rlhf.py —— PPO 显存粗估与建议
# 用法：python plan_rlhf.py <模型B> <full|lora> <卡数> <单卡GB>
import sys

model_b = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
mode = sys.argv[2] if len(sys.argv) > 2 else "lora"
n_gpu = int(sys.argv[3]) if len(sys.argv) > 3 else 4
vram = int(sys.argv[4]) if len(sys.argv) > 4 else 24

GB = 1024 ** 3
params = model_b * 1e9
if mode == "full":
    total = params * 16 * 4   # policy+ref+RM+critic（全参，未含激活）
    note = "全参四模型；激活与采样开销大，按 1.5-2x 预留"
else:
    total = params * 2 * 3 + params * 2 * 0.3  # 3 份冻结权重 + 策略 LoRA 开销（粗估）
    note = "LoRA 策略 + 冻结 ref/RM/critic；实际以框架 profiling 为准"

per_gpu = total / n_gpu / GB
print("模型 %.1fB 模式 %s 卡数 %d 单卡 %dGB" % (model_b, mode, n_gpu, vram))
print("粗估均摊每卡: %.1f GB（未含激活/采样缓存）" % per_gpu)
print("说明:", note)
if per_gpu < vram * 0.6:
    print("结论: 预算充足，可跑；建议先小 batch 验证")
elif per_gpu < vram * 0.95:
    print("结论: 偏紧；开 activation checkpointing / offload / 降 batch")
else:
    print("结论: 放不下；换 LoRA/小模型/DPO，或加卡")
~~~

试算：

~~~bash
python scripts/plan_rlhf.py 3 lora 4 24
python scripts/plan_rlhf.py 7 full 8 80
~~~

### Step 3 最小 PPO 训练脚本（形态参考 TRL PPOv2）

保存 scripts/ppo_v2_demo.py：

~~~python
# scripts/ppo_v2_demo.py —— PPOv2 最小调用（字段以 trl 当日官方文档为准）
import json
import pathlib

from datasets import Dataset
from transformers import AutoTokenizer
from trl import PPOv2Config, PPOv2Trainer

root = pathlib.Path(__file__).resolve().parent.parent
policy_model = str(root / "models" / "Qwen2.5-1.5B-Instruct")   # 演示用小模型
ref_model = policy_model
reward_model = "outputs/rm-lora-final"   # 第 23 章 RM（或独立 RM 服务地址）

rows = [{"prompt": json.loads(line)["prompt"]}
        for line in open(root / "data" / "rl_prompts.jsonl", encoding="utf-8")]
dataset = Dataset.from_list(rows)
tokenizer = AutoTokenizer.from_pretrained(policy_model)

config = PPOv2Config(
    model_name=policy_model,
    reward_model=reward_model,
    ref_model=ref_model,
    learning_rate=1e-6,
    batch_size=4,
    kl_coef=0.05,
    # 其余字段（response_length/epochs/offload 等）以官方文档与硬件为准
)

trainer = PPOv2Trainer(config=config, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()
trainer.save_pretrained("outputs/policy-ppo")
~~~

> 重要说明：PPO 需要同时加载策略/参考/RM（+critic），3B 级建议 4×24GB 或 2×48GB；TRL 各版本字段有差异，运行前先查官方示例。1.5B/3B 跑通后，再按第 12 章方法放大。

### Step 4 训练监控四件套（必做）

| 指标 | 看什么 | 危险信号 |
|---|---|---|
| reward（原始 RM 分） | 是否上升 | 暴涨=可能 hacking |
| kl（与参考的散度） | 是否可控 | 飙升=策略跑偏 |
| policy loss | 是否稳定下降 | 震荡=LR 太大 |
| response length | 是否膨胀 | 持续变长=长度 hacking |

> 训练中每 N 步保存一次 checkpoint（第 13 章规范）；**每次 PPO 实验都保留 SFT 起点**——跑崩了直接回滚，而不是从零再来。

### Step 5 前后评测与验收（10 分钟）

1. 用第 20 章 eval_suite + 通用冒烟集分别测 SFT 起点与 PPO 后模型；
2. 人工看 20 条：是否更简洁、更安全、更少废话（PPO 的收益往往在“开放性偏好”上，关键词分可能不变）；
3. 检查 Reward Hacking：长答案是否变多、拒答是否滥用、格式是否注水；
4. 通过则归档：

~~~bash
cd llm-demo
git add data scripts outputs logs
git commit -m "ppo: v1 experiment (kl=0.05) + eval & hacking checks"
git tag ppo-v1
~~~

> 生产提示：PPO 的效果与 RM 质量强相关——RM 不准时，PPO 会把 RM 的偏见放大。RM 每轮升级后，PPO 要重跑对比。

## 06 参数详解：PPO 核心参数速查

| 参数 | 参考取值 | 说明 |
|---|---|---|
| kl_coef (β) | 0.01–0.2 | 越大越保守；从 0.05 起步 |
| policy LR | 1e-7~1e-6（全参）/ 1e-6~5e-6（LoRA） | PPO 更新必须小步 |
| batch_size | 4–32 prompts | 每次 rollout 的提示数 |
| mini_batch | 1–8 | 更新粒度，影响稳定性 |
| clip_range | 0.1–0.3 | PPO 裁剪范围 |
| GAE lambda | 0.95–0.99 | 优势估计折扣 |
| response_length | 128–1024 | 与任务输出长度匹配 |
| rollout 次数/epoch | 1–4 | 同一批回答复用次数，太多过拟合 |

> PPO 调参第一原则：**任何参数都别一次改太多**；先固定 kl 与 LR，用“reward 曲线 + KL 曲线”判断方向。

---

## 07 高频踩坑排查

**坑 1：Reward Hacking**
症状：reward 暴涨但回答质量变差（变长/谄媚/套模板）。
解法：加大 kl_coef、数据里加反例、RM 集成；最终用真实评测验收而不是 reward。

**坑 2：KL 失控**
症状：策略快速偏离参考模型，输出开始乱。
解法：降 LR、升 kl_coef、检查 reward scale（RM 分数应先归一化）。

**坑 3：Critic/Value 不稳**
症状：value loss 震荡，优势估计失真。
解法：价值模型独立初始化、LR 更低、GAE lambda 调低。

**坑 4：显存 OOM**
症状：四模型同时加载放不下。
解法：LoRA 策略、冻结模型 offload、小 batch；或改用 DPO/GRPO。

**坑 5：rollout 与更新分布不一致**
症状：同一批回答反复用多次，策略过拟合该批。
解法：限制 reuse 次数；提示集要足够大且多样。

**坑 6：reward scale 未归一化**
症状：RM 分数整体漂移，kl_coef 失效。
解法：对 reward 做标准化/减基线；监控 reward 分布。

**坑 7：只信 reward 不信评测**
症状：reward 涨了直接上线，用户反馈变差。
解法：离线评测+人工抽检+线上 A/B 三关都过才算数。

---

## 08 进阶优化：PPO 的进化方向

**① GRPO**：去掉 critic，用一组采样回答的相对奖励算优势——实现更简单，推理/代码类任务表现好，成为近年主流之一。

**② PPO offload/量化**：参考与 RM 用 INT8/offload 加载，策略 LoRA 训练，单机多卡也能跑 7B 级实验。

**③ OpenRLHF / verl**：生产规模（70B+、多节点）用专门 RL 框架，处理 rollout 并行、混合引擎与断点续训。

**④ 在线 RM 刷新**：PPO 跑几轮后，用新策略生成样本刷新偏好数据与 RM，避免“旧裁判评新选手”。

**⑤ RLHF 数据飞轮**：把线上用户反馈（点赞/举报/采纳）脱敏回流为偏好数据，持续刷新 RM 与 PPO——对齐是运营出来的，不是一次训出来的。

---

## 09 本章核心总结（TOP3）

**TOP1**：PPO 循环 = rollout → RM 打分 → KL 约束 → 优势更新；四个模型中 policy 更新、ref/RM 冻结，KL 惩罚是防 Reward Hacking 的第一道锁。

**TOP2**：资源先算账：全参四模型 7B 需 8×80GB 级，LoRA 策略+冻结模型 offload 可把门槛降到单机多卡；先小模型跑通再放大。

**TOP3**：验收永远看“真实效果”而不是 reward：reward 暴涨+KL 飙升+回答变长=典型 hacking；离线评测、人工抽检、线上 A/B 三关缺一不可。

---

## 10 连载衔接

上一章（第 23 章）训练出了 RM 裁判；本章让策略在裁判反馈下自我进化——RLHF/PPO 的工程骨架与避坑清单已经就位。

下一章回答“没有人工偏好数据怎么办”：【第 25 章】RLAIF：AI 反馈自动对齐与偏好合成。让强模型当标注员，把对齐成本打下来。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #RLHF #PPO #强化学习 #KL散度 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 24 RLHF/PPO 工程落地（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 25 章 RLAIF：AI 反馈自动对齐*