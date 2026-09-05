# 【LLM全栈工程·第21章】多轮对话微调专项：长上下文、多轮数据与稳定性

> 本篇为「LLM全栈工程」连载第 21 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 单轮评测全过、一聊多轮就崩？多轮微调专项；② 让 AI“记得上一句”：多轮数据构造与训练要点；③ 越聊越傻怎么破：多轮稳定性评测与优化。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「微调与对齐体系」第 8 篇：解决多轮对话场景的数据、训练与稳定性问题 |
| 适用场景 | 个人学习（多轮 Demo）；小团队做客服/助手类产品；企业多轮 Agent 落地 |
| 核心知识点 | 多轮数据构造；上下文截断；角色一致性；多轮评测；“越聊越傻”排查 |
| 技术选型 | ShareGPT 格式；多轮数据比例；截断策略（保尾截头/摘要）；评测场景设计 |
| 分步实操 | 生成多轮训练集 → 注册与训练 → 多轮场景自动评测 → 长会话压测 → 归档 |
| 参数详解 | cutoff_len、轮数分布、单多轮比例、packing、截断保留轮数 |
| 踩坑排查 | 只训单轮、截断切半句、角色错乱、system 漂移、长会话遗忘、复读 |
| 进阶优化 | 摘要记忆、工具调用多轮、多轮 DPO、长上下文评测 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 22 章：模型蒸馏实战 |

---

## 01 开篇导语

很多模型单轮问答表现不错，一进多轮对话就露馅：

- 第二轮问“那它呢？”，模型不知道“它”指打印机还是 WiFi；
- 用户说“按你说的试了还是不行”，模型又把第一轮答案重复一遍；
- 聊到第 8 轮，角色开始混乱，甚至替用户说话；
- 上下文一长，最早的约定全忘光。

多轮能力不是单轮能力的简单叠加——模型需要学会：**记住上下文、跟随话题、分清角色、在长对话中保持稳定**。这需要专门的数据与训练设计。

本章解决三件事：

1. 多轮训练数据怎么构造（指代、追问、话题切换、收尾）；
2. 训练时上下文怎么截断才不伤多轮能力；
3. 多轮效果怎么评测（含“越聊越傻”压测）。

> 一句话记住本章：**多轮微调 = 构造有“记忆需求”的数据 + 保尾截断 + 角色一致性训练 + 长会话评测。**

---

## 02 白话原理：多轮对话对模型提出了什么新要求

### 2.1 单轮 vs 多轮的能力差

| 能力 | 单轮 | 多轮 |
|---|---|---|
| 指代消解 | 无 | 知道“它/那台机器”指什么 |
| 上下文利用 | 无 | 用前文信息修正回答 |
| 话题管理 | 无 | 能追问、能切换、能回到旧话题 |
| 角色保持 | 弱 | 长时间不串角色 |
| 状态一致 | 无 | 前后回答不矛盾 |

### 2.2 多轮数据里的四种关键结构

| 结构 | 例子 | 训练价值 |
|---|---|---|
| 追问 | 用户继续问同一主题 | 学会不重复已回答内容 |
| 指代 | “那台打印机呢？” | 学会消解指代 |
| 话题切换 | 问完打印机问 WiFi | 学会切换与保持人设 |
| 澄清/收尾 | “我说的不是这个意思” | 学会修正理解 |

> 多轮数据的“灵魂”是让**后面的回答必须依赖前面说过的话**——如果每轮都能独立回答，模型就学不会记忆。

## 03 多轮数据构造与训练要点

### 3.1 数据构造五条军规

1. **轮数分布**：2–8 轮都要有，2 轮 30%、3–5 轮 50%、6–8 轮 20%（按业务调整）；
2. **必须含指代与追问**：至少 30% 样本的后续轮次不能独立回答；
3. **system 全程一致**：每轮之间 system 不重复注入（模板自动处理），人设不漂移；
4. **收尾自然**：包含“谢谢/再见”类短轮，避免模型只会长篇大论；
5. **答案只出现在 assistant**：ShareGPT 格式天然区分角色，loss 只算 assistant 回答。

### 3.2 截断策略：保尾截头，不切半句

长对话必然超 cutoff_len，截断原则：

| 策略 | 做法 | 适用 |
|---|---|---|
| 保尾截头（推荐） | 从最早轮次开始丢，保留 system+最近 N 轮 | 默认 |
| 摘要旧轮 | 把早期轮次压缩成摘要再保留 | 超长会话 |
| 窗口对话 | 只给最近 K 轮（产品层控制） | 与记忆管理配合 |

**红线**：截断必须在“轮次边界”进行——宁可丢整轮，也不能把某一轮 assistant 回答切成两半（模型会学到半截话）。token 预算建议：system 5% + 历史 60% + 当前轮 35%。

### 3.3 训练配置差异

| 配置 | 单轮 | 多轮 |
|---|---|---|
| 数据格式 | Alpaca | ShareGPT |
| cutoff_len | 1024 | 2048–8192（看业务） |
| 单:多轮混合 | 100:0 | 建议 30:70 或全多轮+通用单轮回放 |
| 截断 | 截答案尾 | 轮次边界截断 |
| 评测 | 单轮题 | 多轮场景题 |

> 如果用 LlamaFactory：formatting=sharegpt + tags 映射正确，loss 掩码与模板拼接由框架处理——但你要抽查 1–2 条“预处理后的样本”，确认 assistant 轮次是完整保留的。

## 05 分步实操：多轮数据 → 训练 → 场景评测（约 1.5 小时）

### Step 1 生成多轮训练数据（15 分钟）

保存 scripts/make_multiturn_train.py：

~~~python
# scripts/make_multiturn_train.py —— 由领域规则生成多轮 ShareGPT 数据
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
corpus_path = root / "domain" / "domain_out" / "domain_corpus.jsonl"
out_path = root / "data" / "multiturn_train.jsonl"

rules = [json.loads(line) for line in open(corpus_path, encoding="utf-8")]
SYSTEM = {"from": "system", "value": "你是公司 IT 帮助台助手，严格按公司规定回答，不编造。"}
GREETING = {"from": "gpt", "value": "你好，我是 IT 帮助台助手，请描述你遇到的问题。"}
CLOSING = {"from": "gpt", "value": "不客气，有其他问题随时找我。"}

def answer_of(rec):
    text = rec["text"]
    return text.split("。", 1)[1].strip() if "。" in text else text

def question_of(rec):
    return rec["title"] + "，按公司规定应该怎么处理？"

ESCALATION = {
    "R001": "请提交 IT 工单并注明打印机 IP 与错误码，工程师会远程排查。",
}
def escalate(rec):
    return ESCALATION.get(rec["rule_id"], "请提交 IT 工单或拨打 IT 热线，工程师会协助处理。")

convs = []
for rec in rules:
    ans = answer_of(rec)
    q = question_of(rec)
    # 8 轮标准链：问候->提问->回答->追问->升级->感谢->收尾
    convs.append({
        "conversations": [
            SYSTEM,
            {"from": "human", "value": "你好，在吗？"}, GREETING,
            {"from": "human", "value": q}, {"from": "gpt", "value": ans},
            {"from": "human", "value": "按你说的试了还是不行。"}, {"from": "gpt", "value": escalate(rec)},
            {"from": "human", "value": "好的，明白了，谢谢。"}, CLOSING,
        ]
    })
    # 4 轮短链
    convs.append({
        "conversations": [
            SYSTEM,
            {"from": "human", "value": q}, {"from": "gpt", "value": ans},
            {"from": "human", "value": "谢谢"}, CLOSING,
        ]
    })

# 话题切换链：问完 A 再问 B（6 轮）
for i, rec in enumerate(rules):
    other = rules[(i + 1) % len(rules)]
    convs.append({
        "conversations": [
            SYSTEM,
            {"from": "human", "value": question_of(rec)}, {"from": "gpt", "value": answer_of(rec)},
            {"from": "human", "value": "明白了。再问一个，" + question_of(other)},
            {"from": "gpt", "value": answer_of(other)},
            {"from": "human", "value": "都清楚了，谢谢！"}, CLOSING,
        ]
    })

with open(out_path, "w", encoding="utf-8") as f:
    for conv in convs:
        f.write(json.dumps(conv, ensure_ascii=False) + "\n")
print("multiturn samples:", len(convs))
turns = [len(x["conversations"]) for x in convs]
print("turn distribution:", {n: turns.count(n) for n in sorted(set(turns))})
~~~

运行：

~~~bash
python scripts/make_multiturn_train.py
head -n 1 data/multiturn_train.jsonl
~~~

> 检查要点：抽 2 条人工展开，确认“追问轮必须依赖前文才能回答”而不是万能回答；生产数据建议 30%+ 带指代（“那它呢？”）。

### Step 2 注册数据集并训练（20–40 分钟）

在 data/dataset_info.json 注册（同第 17 章 multiturn_demo 的结构，文件名换成 multiturn_train），然后保存 configs/sft_multiturn.yaml：

~~~yaml
model_name_or_path: models/Qwen2.5-7B-Instruct
template: qwen
stage: sft
finetuning_type: lora
dataset_dir: data
dataset: multiturn_train
val_size: 0.15
cutoff_len: 2048
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2.0e-4
num_train_epochs: 3.0
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
output_dir: outputs/helpdesk-multiturn
~~~

~~~bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/sft_multiturn.yaml
~~~

> 训练前先跑第 17 章 template_doctor 的多轮渲染检查；训练中盯 val loss，过长轮次样本可能造成单样本 loss 波动，属正常，看整体趋势。

### Step 3 多轮场景自动评测（15 分钟）

保存 scripts/multiturn_eval.py：

~~~python
# scripts/multiturn_eval.py —— 多轮场景评测：追问/话题切换/复读检查
import json
import pathlib

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "helpdesk-mt"

def ask(messages):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=300,
    )
    return resp.choices[0].message.content

SCENARIOS = [
    {
        "name": "printer_followup",
        "turns": [
            {"q": "打印机连不上怎么办？", "keywords": ["HP LaserJet M405"]},
            {"q": "按你说的重启了还是不行。", "keywords": ["工单"]},
            {"q": "好的，谢谢。", "max_len": 80},
        ],
    },
    {
        "name": "topic_switch",
        "turns": [
            {"q": "公司 WiFi 连不上怎么办？", "keywords": ["Corp-WiFi"]},
            {"q": "好的。再问一下，VPN 账号被锁定了怎么办？", "keywords": ["30 分钟"]},
        ],
    },
    {
        "name": "long_session",
        "turns": [
            {"q": "打印机卡纸了怎么处理？", "keywords": ["卡纸"]},
            {"q": "取出来之后还是报错怎么办？", "keywords": ["工单"]},
            {"q": "那 WiFi 呢，也连不上了。", "keywords": ["Corp-WiFi"]},
            {"q": "好的，我再试试。", "max_len": 80},
        ],
    },
]

report = []
for scenario in SCENARIOS:
    history = [{"role": "system", "content": "你是公司 IT 帮助台助手，严格按公司规定回答，不编造。"}]
    passed = 0
    prev_answer = ""
    for idx, turn in enumerate(scenario["turns"]):
        history.append({"role": "user", "content": turn["q"]})
        answer = ask(history)
        history.append({"role": "assistant", "content": answer})
        ok = True
        reasons = []
        for kw in turn.get("keywords", []):
            if kw not in answer:
                ok = False
                reasons.append("missing:" + kw)
        if "max_len" in turn and len(answer) > turn["max_len"]:
            ok = False
            reasons.append("too_long")
        if answer.strip().lower().startswith(("user:", "human:", "assistant:")):
            ok = False
            reasons.append("role_leak")
        if idx > 0 and answer == prev_answer:
            ok = False
            reasons.append("repeat_prev")
        prev_answer = answer
        if ok:
            passed += 1
    report.append({"scenario": scenario["name"], "passed": passed, "total": len(scenario["turns"])})
    print(scenario["name"], passed, "/", len(scenario["turns"]))

summary = {"total_scenarios": len(report), "pass_rate": round(sum(r["passed"] for r in report) / max(1, sum(r["total"] for r in report)), 3)}
out = pathlib.Path(__file__).resolve().parent.parent / "logs" / "multiturn_eval_report.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "scenarios": report}, f, ensure_ascii=False, indent=2)
print("summary:", summary)
~~~

合并导出（configs/export 指向 outputs/helpdesk-multiturn）、vLLM 起服务后运行：

~~~bash
llamafactory-cli export configs/export_mt.yaml
vllm serve models/helpdesk-mt-merged --served-model-name helpdesk-mt --port 8000 --max-model-len 8192
python scripts/multiturn_eval.py
cat logs/multiturn_eval_report.json
~~~

**验收线**：追问轮能给出新信息（不重复第一轮答案）；话题切换正常；收尾轮简短；全程无角色泄漏。任一场景挂掉，先查数据（有没有对应结构）再查截断。

### Step 4 归档

~~~bash
cd llm-demo
git add data configs scripts logs
git commit -m "multiturn: train data v1 + mt eval (followup/topic_switch/long)"
git tag mt-v1
~~~

> 生产进阶：把多轮 badcase（聊崩的线上对话）脱敏后回流训练集，形成“越聊越傻”的持续修复闭环。

## 06 参数详解：多轮微调速查

| 参数 | 参考取值 | 说明 |
|---|---|---|
| cutoff_len | 2048–8192 | 必须大于“system+历史+当前轮” |
| 轮数分布 | 2 轮 30% / 3–5 轮 50% / 6+ 轮 20% | 按业务场景调整 |
| 指代样本占比 | ≥30% | 没有指代就学不会记忆 |
| 单:多轮比例 | 30:70 或全多轮+通用回放 | 纯多轮会损失单轮泛化 |
| 截断策略 | 轮次边界保尾截头 | 禁止切半句 |
| token 预算 | system 5% / 历史 60% / 当前 35% | 长会话按此留预算 |
| 评测会话长度 | 4–12 轮 | 覆盖短期与中期记忆 |

---

## 07 高频踩坑排查

**坑 1：训练集全是单轮**
症状：多轮评测一塌糊涂，模型不会利用前文。
解法：训练数据混入 30%+ 多轮样本，且多轮样本必须“依赖前文”。

**坑 2：截断切在轮次中间**
症状：某些训练样本的 assistant 回答只剩半句，模型学成半截话。
解法：按轮次边界截断；截断策略记录进 manifest。

**坑 3：多轮数据没有指代**
症状：用户说“那它呢”，模型答非所问。
解法：构造“前文实体 + 指代提问”样本，检查后续回答是否依赖前文。

**坑 4：角色混乱**
症状：聊久了模型开始替用户说话或自称 user。
解法：ShareGPT 角色映射正确；loss 只算 assistant；评测加角色泄漏检查。

**坑 5：system 漂移**
症状：训练样本有的带 system 有的不带，线上加了 system 行为不一致。
解法：训练与线上 system 文案一致；样本统一带 system。

**坑 6：长会话遗忘**
症状：聊到第 10 轮忘了第 2 轮说的型号。
解法：数据覆盖长会话；产品层做记忆管理/摘要（第 25 章）；评测包含“跨轮信息利用”题。

**坑 7：复读机**
症状：用户追问时模型把上一轮答案原样重复。
解法：数据里构造“追问-新信息”对；评测检测相邻轮重复；推理时可加重复惩罚（第 34 章）。

---

## 08 进阶优化：多轮微调的进化方向

**① 摘要记忆**：超长会话把早期轮次压成摘要再续聊，让模型“记得重点但不用背全文”（第 25 章记忆管理展开）。

**② 工具调用多轮**：客服/Agent 场景加入“请求工具→返回结果→继续对话”的多轮结构，训练模型在合适时机调工具（第 21 章数据格式扩展）。

**③ 多轮 DPO**：对整段多轮对话做偏好优化（哪段回答更好），比单轮 DPO 更能改善“越聊越差”（第 24–25 章）。

**④ 长上下文评测**：用“跨轮信息定位”题（第 1 轮埋信息、第 8 轮提问）与长会话压测做门禁，防止只优化了短对话。

**⑤ 数据回流闭环**：线上多轮对话脱敏后按“聊崩类型”打标（复读/遗忘/角色乱/答非所问），回流训练——多轮质量靠持续喂养真实场景。

---

## 09 本章核心总结（TOP3）

**TOP1**：多轮能力 = 指代消解 + 上下文利用 + 话题管理 + 角色保持；数据里至少 30% 样本“后面的回答必须依赖前面”，否则学不会记忆。

**TOP2**：截断只按轮次边界保尾截头；system 训练线上一致；loss 只算 assistant；多轮用 ShareGPT 格式，别手拼字符串。

**TOP3**：多轮评测要覆盖追问、话题切换、长会话与复读/角色泄漏检查；线上聊崩样本回流训练集，形成“越聊越傻”的持续修复闭环。

---

## 10 连载衔接

上一章（第 20 章）建立了效果法庭；本章把法庭扩展到多轮战场——主线工程从“单轮问答助手”升级为“能连续对话的领域助手”。

下一章开始“变小”的魔法：【第 22 章】模型蒸馏实战：大模型萃取小模型的完整流程。大模型太贵太大？把它的能力蒸馏进小模型，是成本与部署的破局点。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #多轮对话 #多轮微调 #ShareGPT #长上下文 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 21 多轮对话微调专项（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 22 章 模型蒸馏实战*