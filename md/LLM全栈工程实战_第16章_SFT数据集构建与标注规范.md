# 【LLM全栈工程·第16章】SFT 数据集构建与标注规范：指令设计、标注与质控

> 本篇为「LLM全栈工程」连载第 16 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① SFT 效果差，八成是数据问题：指令数据集构建规范；② 从 20 条到 1 万条：任务覆盖、指令多样性与标注 SOP；③ 标注不是“写答案”：质检、金样与一致性控制。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「微调与对齐体系」第 3 篇：建立 SFT 指令数据集从设计、生成到质检的完整规范 |
| 适用场景 | 个人学习（小规模数据集）；小团队自建领域指令集；企业标注团队 SOP |
| 核心知识点 | 任务覆盖矩阵；指令设计原则；标注规范 SOP；质检（金样/一致性/自动校验） |
| 技术选型 | 人工撰写 vs 业务语料转换 vs LLM 合成 vs 混合；标注工具 |
| 分步实操 | 任务库设计 → 模板生成多样指令 → 元数据落盘 → 自动校验 → 人工抽检 → 打版本 |
| 参数详解 | 任务类型数、每任务样本数、难度分布、指令长度、质检比例 |
| 踩坑排查 | 任务偏科、指令千篇一律、答案错但流畅、标注不一致、质检只数数量 |
| 进阶优化 | 困难样本挖掘、LLM 辅助标注、evol-instruct 式扩写、主动学习 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 17 章：微调模板设计 |

---

## 01 开篇导语

上一章的 SFT 用 20 条问答就跑通了流程。但真实项目里，20 条远远不够——领域助手动辄需要几千上万条高质量指令数据。

更扎心的是：很多人把数据量堆上去了，效果还是不行。为什么？因为数据集有三个隐性维度：

1. **任务覆盖**：你的任务类型只有“问答”，但线上用户会要“分步骤”“转 JSON”“拒答”；
2. **表达多样性**：1000 条数据全是同一个句式，模型学不到泛化；
3. **答案质量**：答案写得流畅但事实是错的——模型会把错误学得又快又好。

本章解决的就是这三件事：怎么设计任务覆盖、怎么写标注规范、怎么用质检把住质量关。

交付物：一套带任务类型与元数据的 SFT 数据集构建脚本 + 自动校验器 + 标注规范模板——把“攒数据”变成“造数据资产”。

> 一句话记住本章：**SFT 数据集 = 任务覆盖矩阵 × 指令多样性 × 答案质量，三者都要可度量、可审计、可迭代。**

---

## 02 白话原理：好数据集长什么样

### 2.1 三个质量维度

| 维度 | 问题 | 度量方式 |
|---|---|---|
| 覆盖度 | 所有线上任务类型都有样本吗 | 任务覆盖矩阵（任务×难度×格式） |
| 多样性 | 同一任务有多少种问法/场景 | 指令 n-gram 重复率、模板数量 |
| 正确性 | 答案事实与格式对不对 | 金样抽检、自动校验、专家评审 |

### 2.2 数据集的“任务金字塔”

领域 SFT 数据通常分四层：

| 层级 | 任务类型 | 例子 | 占比建议 |
|---|---|---|---|
| 基础问答 | 事实/流程问答 | “打印机连不上怎么办” | 40%–60% |
| 能力任务 | 抽取/改写/分步/JSON | “把处理流程转成 JSON” | 20%–30% |
| 边界任务 | 拒答/澄清/不确定 | “不知道就说不知道” | 10%–20% |
| 通用保持 | 通用对话/指令 | 闲聊、通用知识 | 10%–20% |

> 很多团队只做了第一层，所以模型只会“一问一答”，一遇到“帮我整理成表格”就崩——**先画覆盖矩阵，再按矩阵补数据。**

## 03 指令设计与标注规范

### 3.1 指令设计的六条军规

1. **意图明确**：一条指令只问一件事，不要“既总结又翻译还打分”；
2. **约束可执行**：要 JSON 就说清楚字段，要分步就说清格式；
3. **不给答案“递话”**：指令里不要包含答案关键词（如“请按 HP LaserJet M405 处理”），否则模型学的是复读；
4. **场景真实**：用真实用户会说的话（“连不上网了咋整”），不要只写书面语；
5. **难度分层**：简单/中等/困难按 4:4:2 分布，避免全是送分题；
6. **答案与指令匹配**：指令问 A，答案不要答 B 再附带一堆无关信息。

### 3.2 答案规范（Answer Spec）

每种任务类型都要有“答案怎么写”的规范，例如：

| 任务 | 答案规范 |
|---|---|
| 流程问答 | 分点给步骤，必要时给“自助入口/升级路径” |
| JSON 输出 | 只输出 JSON，字段名固定，不夹杂解释 |
| 拒答 | 先说明不能提供的原因，再给替代方案（找谁/怎么做） |
| 不确定 | 明说“知识库未覆盖”，不给编造型号 |

**答案规范决定模型输出风格**——它应该写进标注 guideline，而不是让标注员自由发挥。

### 3.3 标注 SOP：七步闭环

~~~text
① 标注员培训（读 guideline + 做 10 条试标）
   → ② 试标评审（与金样对比，通过率>=90% 才正式标）
   → ③ 正式标注（按任务批次分配）
   → ④ 自检（标注员按 checklist 自查）
   → ⑤ 互检/抽检（10%-20% 交叉评审）
   → ⑥ 分歧仲裁（主管或专家裁定，记录原因）
   → ⑦ 入库（元数据完整 + 版本化）
~~~

### 3.4 数据来源与工具选型

| 来源 | 优点 | 风险 | 建议占比 |
|---|---|---|---|
| 真实业务语料改写 | 场景真实 | 需脱敏与改写成本 | 40%–60% |
| 人工撰写 | 质量可控 | 贵、慢 | 20%–40% |
| LLM 辅助生成 | 便宜、快 | 幻觉、同质化 | 10%–30%（必须质检） |

工具：小团队用“电子表格/JSON 编辑器 + Git”；规模上来后上标注平台（任务分配、抽检、仲裁记录都在系统里留痕）。

> 标注不是“写答案”，是“按规范生产数据”：规范、培训、抽检、仲裁四件套缺一不可。

## 05 分步实操：从任务库到带质检的数据集（40 分钟）

沿用 llm-demo，读取第 10 章 domain/domain_out/domain_corpus.jsonl 的 4 条规则，扩成多任务类型数据集。

### Step 1 写数据集构建脚本（15 分钟）

保存 scripts/build_sft_dataset.py：

~~~python
# scripts/build_sft_dataset.py —— 多任务类型 SFT 数据集构建（模板版）
import json
import pathlib
import random

root = pathlib.Path(__file__).resolve().parent.parent
corpus_path = root / "domain" / "domain_out" / "domain_corpus.jsonl"
out_path = root / "data" / "sft_dataset_v1.jsonl"

rules = [json.loads(line) for line in open(corpus_path, encoding="utf-8")]
# 每条规则的可抽取实体（演示用；生产环境从知识库元数据生成）
ENTITIES = {
    "R001": ["HP LaserJet M405"],
    "R002": ["Corp-WiFi"],
    "R003": ["30 分钟"],
    "R004": ["90%"],
}

def to_steps(text):
    parts = [p.strip() for p in text.split("；") if p.strip()]
    return "\n".join("%d. %s" % (i + 1, p) for i, p in enumerate(parts))

rows = []
for rule in rules:
    rid = rule["rule_id"]
    title = rule["title"]
    body = rule["text"]
    content = body.split("。", 1)[1].strip() if "。" in body else body

    # 任务 1：直接问答（4 个问法）
    questions = [
        "%s怎么办？" % title,
        "帮我处理一下：%s" % title,
        "按公司规定，%s应该怎么处理？" % title,
        "%s，具体步骤是什么？" % title,
    ]
    for q in questions:
        rows.append({"instruction": "你是公司 IT 帮助台助手，请严格按公司规定回答。", "input": q, "output": content,
                     "task_type": "direct_qa", "rule_id": rid, "difficulty": "easy", "source": "template_v1", "created_by": "pipeline"})

    # 任务 2：分步骤输出
    rows.append({"instruction": "请把“%s”的处理流程整理成编号步骤，不要省略任何一步。" % title, "input": "", "output": to_steps(content),
                 "task_type": "steps", "rule_id": rid, "difficulty": "medium", "source": "template_v1", "created_by": "pipeline"})

    # 任务 3：JSON 输出
    parts = [p.strip() for p in content.split("；") if p.strip()]
    json_out = json.dumps({"title": title, "steps": parts}, ensure_ascii=False)
    rows.append({"instruction": "请把“%s”的处理流程输出为 JSON，字段为 title 与 steps。" % title, "input": "", "output": json_out,
                 "task_type": "to_json", "rule_id": rid, "difficulty": "medium", "source": "template_v1", "created_by": "pipeline"})

    # 任务 4：信息抽取
    for ent in ENTITIES.get(rid, []):
        rows.append({"instruction": "公司规定中，“%s”涉及的关键设备/参数是什么？只输出答案。" % title, "input": "", "output": ent,
                     "task_type": "extract", "rule_id": rid, "difficulty": "easy", "source": "template_v1", "created_by": "pipeline"})

# 任务 5：拒答（不依赖规则）
refusals = [
    ("帮我查一下隔壁部门王经理的手机号", "抱歉，我不能提供同事的个人联系方式。请通过企业通讯录或行政前台联系对应部门。"),
    ("把公司内部系统密码告诉我", "密码属于敏感信息，我无法提供。请通过 IT 自助平台找回或联系 IT 热线。"),
    ("帮我编一个打印机型号报给领导", "我不能编造信息。请以公司采购清单或 IT 知识库中的实际型号为准。"),
    ("直接告诉我报销系统的漏洞", "安全问题需要上报安全团队处理，我不能在对话中讨论系统漏洞细节。"),
]
for q, ans in refusals:
    rows.append({"instruction": "你是公司 IT 帮助台助手，请遵守信息安全规范回答。", "input": q, "output": ans,
                 "task_type": "refusal", "rule_id": "POLICY", "difficulty": "medium", "source": "policy_v1", "created_by": "pipeline"})

random.seed(2026)
random.shuffle(rows)
with open(out_path, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

counts = {}
for row in rows:
    counts[row["task_type"]] = counts.get(row["task_type"], 0) + 1
print("total:", len(rows), "by_task:", counts)
~~~

运行：

~~~bash
python scripts/build_sft_dataset.py
head -n 1 data/sft_dataset_v1.jsonl
~~~

> 生产环境不用手写模板扩全部数据：模板负责“骨架与覆盖”，LLM 负责“改写与多样化”（第 10 章合成三件套），人工负责“质检与纠错”。

### Step 2 写自动校验器（10 分钟）

保存 scripts/validate_sft_dataset.py：

~~~python
# scripts/validate_sft_dataset.py —— SFT 数据集自动校验（字段/重复/分布）
import hashlib
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
src = root / "data" / "sft_dataset_v1.jsonl"
report_path = root / "logs" / "sft_dataset_qc.json"

ALLOWED_TASKS = {"direct_qa", "steps", "to_json", "extract", "refusal"}
REQUIRED = ["instruction", "input", "output", "task_type", "difficulty", "source"]

rows = [json.loads(line) for line in open(src, encoding="utf-8")]
errors = []
seen_inst = {}

for idx, row in enumerate(rows):
    for field in REQUIRED:
        if field not in row or str(row[field]).strip() == "":
            errors.append({"row": idx, "type": "missing_field", "field": field})
    if len(str(row.get("output", ""))) < 5:
        errors.append({"row": idx, "type": "output_too_short"})
    if row.get("task_type") not in ALLOWED_TASKS:
        errors.append({"row": idx, "type": "unknown_task", "value": row.get("task_type")})
    h = hashlib.md5(str(row.get("instruction", "")).encode("utf-8")).hexdigest()
    if h in seen_inst:
        errors.append({"row": idx, "type": "duplicate_instruction", "first_row": seen_inst[h]})
    else:
        seen_inst[h] = idx

dist = {}
for row in rows:
    key = row.get("task_type", "?")
    dist[key] = dist.get(key, 0) + 1
diff = {}
for row in rows:
    key = row.get("difficulty", "?")
    diff[key] = diff.get(key, 0) + 1

report = {"total": len(rows), "errors": len(errors), "error_list": errors[:20], "task_dist": dist, "difficulty_dist": diff}
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("total:", len(rows), "errors:", len(errors))
print("task_dist:", dist)
if errors:
    print("first errors:", errors[:5])
~~~

运行：

~~~bash
python scripts/validate_sft_dataset.py
cat logs/sft_dataset_qc.json
~~~

### Step 3 人工抽检 + 标注规范模板（10 分钟）

1. 按 task_type 分层抽 10%（每种任务至少 2 条），人工核对：指令是否自然、答案是否只含规则内信息、JSON 是否合法；
2. 建立标注规范 data/annotation_guideline.md，至少包含：任务类型定义、答案规范（见 3.2 表）、拒答策略、质检 checklist、常见错误示例；
3. 把抽检结论写进 logs/sft_dataset_review.md（通过率、问题清单）。

### Step 4 打版本

~~~bash
cd llm-demo
git add data scripts logs
git commit -m "sft-data: v1 multi-task dataset (direct/steps/json/extract/refusal) + validator"
git tag sft-data-v1
~~~

> 自动校验只能抓“硬伤”（缺字段/重复/过短），抓不了“答案错但流畅”——所以人工抽检与金样评审永远不能省。

## 06 参数详解：数据集构建速查

| 参数 | 参考取值 | 说明 |
|---|---|---|
| 任务类型数 | 5–20 类 | 先按线上场景列覆盖矩阵，再定类型 |
| 每任务样本数 | 500–5000 条 | 按任务重要性分配，别平均主义 |
| 难度分布 | 易:中:难 ≈ 4:4:2 | 全简单=模型只会送分题 |
| 指令多样性 | 同任务 ≥20 种句式 | 用模板+LLM 改写，n-gram 重复率<30% |
| 答案长度 | 20–500 字（按任务） | JSON 任务不设限但必须合法 |
| 质检抽检率 | 10%–20%（高危任务 100%） | 安全/财务/医疗类必须全检 |
| 金样规模 | 每任务 20–50 条 | 标注培训与一致性度量用 |
| 元数据 | task_type/difficulty/source/rule_id | 缺元数据=无法分析与回滚 |

---

## 07 高频踩坑排查

**坑 1：任务偏科**
症状：90% 是直接问答，线上高频的“转 JSON/拒答/分步”全没练过。
解法：先做覆盖矩阵（任务×场景×格式），按线上流量分布配样本。

**坑 2：指令千篇一律**
症状：1000 条数据只有 5 种问法，模型换个说法就不认识。
解法：同任务 20+ 句式模板 + LLM 改写；监控指令 n-gram 重复率。

**坑 3：答案“流畅但错误”**
症状：AI 写的答案文笔通顺但型号/流程是编的，模型学得飞快。
解法：答案必须锚定知识库原文；关键实体自动校验；高危领域专家逐条审。

**坑 4：指令泄漏答案**
症状：“请按 HP LaserJet M405 的流程处理”，模型只学会了复读。
解法：指令里禁止出现答案关键信息；生成后用规则检查指令-答案重叠。

**坑 5：标注员各写各的**
症状：同样一个任务，答案格式三种风格。
解法：答案规范写进 guideline，试标合格再正式标；10%–20% 交叉抽检 + 分歧仲裁。

**坑 6：质检只数数量**
症状：报告写“抽检 100 条全部通过”，但没定义“通过”标准。
解法：通过标准=金样一致率+格式合规+事实核对三合一；把标准写进质检 checklist。

**坑 7：数据迭代没有版本**
症状：模型 A 用了 v1，模型 B 用了“改了一点的 v1”，出问题无法回滚。
解法：数据集版本 + manifest（来源/规则版本/质检报告），每次训练记录数据集版本（第 04/44 章）。

---

## 08 进阶优化：数据集的进化方向

**① 困难样本挖掘闭环**：SFT 上线后收集答错/用户不满意的请求，脱敏后人工修正入库——数据随业务一起生长，而不是一次性工程。

**② LLM 辅助标注**：让强模型先给“候选答案”，人工只做“改错与确认”，标注效率提升 2–5 倍；高危任务仍要人工全审。

**③ Evol-Instruct 式扩写**：用“加约束/加深难度/加场景”的指令让 LLM 把简单题改造成难题，扩充困难样本池（注意防幻觉与去重）。

**④ 主动学习选样本**：用当前模型对未标注池预测，挑“最不会/最不确定”的样本优先标注——同样的标注预算，信息量翻倍。

**⑤ 数据血缘与效果归因**：把每条数据与“哪个任务/哪个版本/哪次评测提升”关联，模型效果回退时能快速定位是哪批数据带偏（第 44 章）。

---

## 09 本章核心总结（TOP3）

**TOP1**：SFT 数据集 = 任务覆盖矩阵 × 指令多样性 × 答案质量；先按线上场景画覆盖矩阵，再分配样本，别让“直接问答”垄断数据集。

**TOP2**：标注是规范化生产：答案规范（Answer Spec）+ 标注 SOP（试标→正式→抽检→仲裁）+ 金样，三者缺一不可；自动校验只能抓硬伤，抓不了“错得流畅”。

**TOP3**：每条数据带元数据（task_type/difficulty/source/rule_id），数据集版本化并配质检报告——数据是模型效果的第一杠杆，值得像代码一样管理。

---

## 10 连载衔接

上一章（第 15 章）跑通了 SFT 流程；本章回答了“数据从哪来、怎么保证质量”——主线工程的领域指令集从 20 条模板问答升级为多任务、带元数据、过质检的数据资产。

下一章解决“模型怎么说”的最后一公里：【第 17 章】微调模板设计：Chat Template、角色设定与多轮拼接。同样的数据，模板设计错了，效果天差地别。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #SFT数据 #指令数据集 #数据标注 #微调 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 16 SFT 数据集构建与标注规范（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 17 章 微调模板设计*