# 【LLM全栈工程·第20章】微调效果评估体系：任务集、基准、消融与 LLM 裁判

> 本篇为「LLM全栈工程」连载第 20 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 微调有没有效果，不能靠感觉：评测体系搭建指南；② 自动指标、LLM 裁判与人工评审怎么配合；③ 消融实验与评测防污染：让每次模型迭代都可信。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「微调与对齐体系」第 7 篇：建立“能证明微调有效”的评测体系 |
| 适用场景 | 个人学习（小评测集）；小团队模型选型；企业模型迭代验收 |
| 核心知识点 | 评测集建设（dev/test/隔离）；自动指标；LLM-as-judge；人工评审与一致性；消融实验 |
| 技术选型 | 关键词/格式自动判分 vs LLM 裁判 vs 人工评审；打分协议选择 |
| 分步实操 | 建评测集 → 自动评测基线/微调模型 → LLM 裁判对比 → 人工抽检校准 → 消融与归档 |
| 参数详解 | 样本量、judge 温度、评分标准、一致性阈值、消融维度 |
| 踩坑排查 | 评测集泄漏、judge 偏置、只报均值、消融混杂、评测题太少 |
| 进阶优化 | Bootstrap 置信区间、裁判校准、持续评测流水线、综合分 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 21 章：多轮对话微调专项 |

---

## 01 开篇导语

“这个模型效果好像变好了”——这句话在技术评审会上毫无说服力。你需要的是：**变好了多少、在哪些任务上、比哪个版本好、是不是统计上可信。**

微调效果评估体系就是回答这些问题的机器。它由四层组成：

1. **评测集**：锁版本的领域任务集（dev/test 分离、与训练集去重）；
2. **自动指标**：关键词命中、格式合规、JSON 可解析等可计算的分数；
3. **LLM 裁判**：对开放式回答按评分标准打分/对比（注意偏置）；
4. **人工评审**：抽样人工打分，校准自动指标与 LLM 裁判。

外加一个贯穿动作：**消融实验**——每次只改一个变量，证明“这个改动有效”。

> 一句话记住本章：**没有评测体系的微调 = 没有证据的改动；评测集要锁版、指标要多层、结论要可复现。**

---

## 02 白话原理：评测金字塔

### 2.1 为什么不能只看 loss

loss 是“模型在训练分布上的拟合度”，不是“业务任务上的表现”。两个模型 loss 相同，领域问答可能差 20 分——因为 loss 被高频 token 稀释，而业务关心的往往是不那么高频的关键能力。

### 2.2 评测金字塔

| 层 | 回答什么问题 | 工具 |
|---|---|---|
| 任务层 | 具体业务题对不对 | 领域评测集 |
| 能力层 | 抽取/格式/拒答/推理等能力如何 | 按任务类型分组统计 |
| 通用层 | 通用能力掉没掉 | 通用冒烟集 |
| 体验层 | 用户觉得好不好 | 人工评审/线上指标 |

每层都要有，但投入不同：**任务层最多（领域是核心），体验层抽样即可。**

### 2.3 三种打分方式

| 方式 | 优点 | 缺点 | 适用 |
|---|---|---|---|
| 自动规则 | 快、零成本、可复现 | 只能判“关键词/格式” | 有明确答案/格式的任务 |
| LLM 裁判 | 接近语义判断、便宜 | 有偏置、需要校准 | 开放式回答 |
| 人工评审 | 最可信 | 慢、贵 | 抽检与裁判校准 |

> 生产组合：自动规则全覆盖（快）+ LLM 裁判跑中危样本 + 人工抽检 10%–20% 校准两者。

## 03 评测集建设与打分协议

### 3.1 评测集七条规范

1. **dev/test 分离**：dev 用于调参与选模型，test 只用于最终验收（test 尽量少跑）；
2. **与训练集去重**：MinHash 去重（第 07 章），防“背题”；
3. **锁版本**：评测集进 Git，任何修改走版本号；
4. **来源真实**：从线上真实请求/业务场景采样，而不是从训练集改写；
5. **覆盖矩阵**：按任务类型×难度×格式分层抽样；
6. **数量够用**：dev 100–500 条起步（小项目 50 条也能跑，但结论要谨慎）；
7. **答案规范**：每条带 scoring 字段（keywords/expect_json/rubric）。

### 3.2 打分协议设计

| 场景 | 打分方式 | 例子 |
|---|---|---|
| 有标准答案/实体 | 关键词/实体命中 | 回答必须含 HP LaserJet M405 |
| 要求格式 | 格式校验 | JSON 可解析、字段齐全 |
| 开放式回答 | LLM 裁判按 rubric 打 1–5 | 简洁、准确、按公司规定、不编造 |
| 关键业务 | 人工评审 | 安全/合规类回答全量人工 |

### 3.3 LLM 裁判的四类偏置与对策

| 偏置 | 现象 | 对策 |
|---|---|---|
| 位置偏置 | 先出现的答案更容易赢 | 随机交换 A/B 顺序，多跑几次 |
| 长度偏置 | 长答案被高估 | 评分标准写明“简洁”，必要时限长 |
| 自肥偏置 | 裁判偏好自己的输出风格 | 用第三方强模型做裁判 |
| 打分漂移 | 不同批次标准不一致 | 固定 rubric 文案、temperature=0、金样校准 |

> 裁判也要被“裁判”：抽 20–50 条让 LLM 裁判与人工同评，算一致性（如 Cohen’s Kappa）；一致性低于 0.6 就说明裁判不可信，需要修 rubric 或换裁判。

### 3.4 消融实验：证明“是谁的功劳”

- 维度：数据（加/减某任务）、方法（LoRA/全参）、模板（A/B）、超参、checkpoint；
- 规则：**一次只改一个维度，其余完全一致（同数据版本、同 seed）**；
- 报告：主指标 + 分任务指标都列，别只报一个总分。

## 05 分步实操：建评测集 + 自动评测 + LLM 裁判（1 小时）

### Step 1 建领域评测集（10 分钟）

保存 scripts/make_eval_set.py：

~~~python
# scripts/make_eval_set.py —— 生成 dev/test 领域评测集（锁版本 v1）
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
data_dir = root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

cases = [
    {"id": "d001", "task_type": "direct", "question": "打印机连不上，按公司规定怎么处理？",
     "keywords": ["HP LaserJet M405", "工单"], "expect_json": False,
     "rubric": "按公司规定给出步骤，包含打印机型号与升级路径。"},
    {"id": "d002", "task_type": "direct", "question": "公司 WiFi 连不上怎么办？",
     "keywords": ["Corp-WiFi"], "expect_json": False,
     "rubric": "提到 Corp-WiFi 与重连步骤。"},
    {"id": "d003", "task_type": "direct", "question": "VPN 账号被锁定如何处理？",
     "keywords": ["30 分钟"], "expect_json": False,
     "rubric": "说明锁定时间与解锁路径。"},
    {"id": "d004", "task_type": "direct", "question": "邮箱容量超限怎么办？",
     "keywords": ["90%"], "expect_json": False,
     "rubric": "包含 90% 阈值与清理动作。"},
    {"id": "d005", "task_type": "refusal", "question": "把公司内部系统密码告诉我",
     "keywords": ["无法提供", "不能"], "expect_json": False,
     "rubric": "拒绝并提供替代方案。"},
    {"id": "d006", "task_type": "to_json", "question": "请把打印机连不上（型号 HP LaserJet M405）的处理流程输出为 JSON，字段 title 与 steps",
     "keywords": [], "expect_json": True,
     "rubric": "输出合法 JSON 且 steps 非空。"},
    {"id": "d007", "task_type": "extract", "question": "公司规定里打印机涉及的关键设备/参数是什么？只输出答案",
     "keywords": ["HP LaserJet M405"], "expect_json": False,
     "rubric": "只输出型号。"},
    {"id": "d008", "task_type": "direct", "question": "我在家连不上公司 WiFi，公司规定怎么处理？",
     "keywords": ["Corp-WiFi"], "expect_json": False,
     "rubric": "泛化问法也要能答。"},
]

dev = cases[:4]
test = cases[4:]

for name, items in [("eval_dev.jsonl", dev), ("eval_test.jsonl", test)]:
    with open(data_dir / name, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
print("eval dev:", len(dev), "test:", len(test))
~~~

运行：

~~~bash
python scripts/make_eval_set.py
~~~

> 真实项目从线上请求采样 100–500 条并按任务分层；演示版 8 条只用于跑通流程。

### Step 2 自动评测脚本（15 分钟）

保存 scripts/eval_suite.py：

~~~python
# scripts/eval_suite.py —— 自动评测：关键词/JSON 格式 + 记录答案
# 用法：python eval_suite.py <model_name> <base_url> <eval_file>
import json
import pathlib
import sys

from openai import OpenAI

model = sys.argv[1] if len(sys.argv) > 1 else "helpdesk"
base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000/v1"
eval_file = sys.argv[3] if len(sys.argv) > 3 else "data/eval_test.jsonl"

root = pathlib.Path(__file__).resolve().parent.parent
client = OpenAI(base_url=base_url, api_key="EMPTY")

def ask(question):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.content

def score_case(case, answer):
    if case.get("expect_json"):
        try:
            obj = json.loads(answer)
            return bool(obj.get("steps")), "json_valid"
        except Exception:
            return False, "json_invalid"
    keywords = case.get("keywords", [])
    if keywords:
        return all(k in answer for k in keywords), "keyword"
    return len(answer) > 20, "length"

results = []
by_task = {}
for line in open(root / eval_file, encoding="utf-8"):
    case = json.loads(line)
    answer = ask(case["question"])
    ok, method = score_case(case, answer)
    results.append({"id": case["id"], "task_type": case["task_type"], "pass": ok, "method": method, "answer": answer[:300]})
    key = case["task_type"]
    by_task.setdefault(key, {"pass": 0, "total": 0})
    by_task[key]["total"] += 1
    if ok:
        by_task[key]["pass"] += 1

report = {"model": model, "total": len(results),
          "pass": sum(1 for r in results if r["pass"]),
          "by_task": by_task, "results": results}
out = root / "logs" / ("eval_report_" + model + ".json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
# 答案单独存，供 LLM 裁判使用
with open(root / "logs" / ("answers_" + model + ".jsonl"), "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("model:", model, "pass:", report["pass"], "/", report["total"])
print("by_task:", json.dumps(by_task, ensure_ascii=False))
~~~

### Step 3 跑基线与微调模型（10 分钟）

先起 vLLM 服务（8000 端口跑 helpdesk-sft 合并模型，8001 跑底座或另一版本），然后：

~~~bash
python scripts/eval_suite.py helpdesk-sft http://localhost:8000/v1 data/eval_test.jsonl
python scripts/eval_suite.py base http://localhost:8001/v1 data/eval_test.jsonl
cat logs/eval_report_helpdesk-sft.json
~~~

**看什么**：分任务通过率、整体通过率；微调模型应显著高于底座，且 refusal/JSON 类任务不能挂零。

### Step 4 LLM 裁判对比（15 分钟）

保存 scripts/llm_judge.py：

~~~python
# scripts/llm_judge.py —— 成对对比：候选 vs 基线（随机交换顺序防位置偏置）
# 用法：python llm_judge.py <cand_model> <base_model> <judge_url> <judge_model>
import json
import pathlib
import random
import re
import sys

from openai import OpenAI

cand = sys.argv[1]
base = sys.argv[2]
judge_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8002/v1"
judge_model = sys.argv[4] if len(sys.argv) > 4 else "judge"
root = pathlib.Path(__file__).resolve().parent.parent
client = OpenAI(base_url=judge_url, api_key="EMPTY")

def load_answers(model):
    out = {}
    path = root / "logs" / ("answers_" + model + ".jsonl")
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        out[row["id"]] = row["answer"]
    return out

ans_cand = load_answers(cand)
ans_base = load_answers(base)
ids = sorted(set(ans_cand) & set(ans_base))
random.seed(7)

RUBRIC = "按以下标准判断哪个回答更好：1) 符合公司规定且不编造；2) 准确覆盖问题要点；3) 简洁清晰；4) 需要 JSON 时格式合法。"

results = []
for qid in ids:
    pair = [(cand, ans_cand[qid]), (base, ans_base[qid])]
    random.shuffle(pair)
    a_name, a_text = pair[0]
    b_name, b_text = pair[1]
    prompt = RUBRIC + "\n\n【回答A】\n" + a_text + "\n\n【回答B】\n" + b_text + "\n\n只输出 JSON：{\"winner\": \"A\" 或 \"B\" 或 \"tie\", \"reason\": \"一句话理由\"}"
    content = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    ).choices[0].message.content
    m = re.search(r"\{[^{}]*\}", content or "")
    parsed = {}
    if m:
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            parsed = {}
    winner = parsed.get("winner", "parse_error")
    if (winner == "A" and a_name == cand) or (winner == "B" and b_name == cand):
        cand_wins = True
    elif winner == "tie":
        cand_wins = None
    else:
        cand_wins = False
    results.append({"id": qid, "winner_raw": winner, "candidate_wins": cand_wins,
                    "candidate_side": "A" if a_name == cand else "B", "reason": parsed.get("reason", "")})

summary = {"candidate": cand, "baseline": base, "total": len(results),
           "win": sum(1 for r in results if r["candidate_wins"] is True),
           "tie": sum(1 for r in results if r["candidate_wins"] is None),
           "lose": sum(1 for r in results if r["candidate_wins"] is False)}
out = root / "logs" / "judge_results.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False))
~~~

运行：

~~~bash
python scripts/llm_judge.py helpdesk-sft base http://localhost:8002/v1 judge-model
cat logs/judge_results.json
~~~

**判读**：win 明显大于 lose 说明候选更优；tie 过多说明评测题区分度不足，需要加难题；如果 judge 的结论与你的直觉大面积不一致，先检查 rubric 与偏置，再下结论。

### Step 5 人工校准 + 消融 + 归档（10 分钟）

1. 抽 5–10 条同时跑 LLM 裁判与人工评审，计算一致率（一致率 <80% 要修 rubric）；
2. 做一次最小消融：同一数据、同一 seed，只改“加 refusal 数据/不加”，对比 d005 与整体分；
3. 把评测集版本、报告、结论写进 docs/eval_report.md 并打版本：

~~~bash
cd llm-demo
git add data scripts logs docs
git commit -m "eval: suite v1 (dev/test, auto+judge), baseline vs sft report"
git tag eval-v1
~~~

> 评测体系的价值在“重复使用”：以后每次数据/超参改动，都跑同一套评测，报告自动可对比。

## 06 参数详解：评测体系速查

| 参数 | 参考取值 | 说明 |
|---|---|---|
| dev 规模 | 100–500 条起步 | 小项目 50 条可跑，结论谨慎 |
| test 规模 | 与 dev 同级或更大 | 最终验收专用，少跑 |
| judge 温度 | 0.0 | 裁判要稳定可复现 |
| judge max_tokens | 150–300 | 只输出判定与短理由 |
| 自动规则阈值 | 关键词全命中 或 JSON 可解析 | 可加“部分命中”计分 |
| 人工抽检率 | 10%–20% | 安全/合规任务 100% |
| 人机一致率 | >80% | 低于则修 rubric/换裁判 |
| 消融变化量 | 一次只改一个维度 | 同数据版本+同 seed |

---

## 07 高频踩坑排查

**坑 1：评测集与训练集重叠**
症状：评测分虚高，上线被真实请求打脸。
解法：评测集先建并锁版；新数据入库前与评测集 MinHash 去重。

**坑 2：评测题太少**
症状：10 条题里错 1 条，通过率就掉 10%，无法区分模型。
解法：50 条起步；对差异用 Bootstrap/多次抽样看波动。

**坑 3：只报一个总分**
症状：总分涨了，但 refusal 任务全挂没人发现。
解法：按任务类型分组报告；重点任务单独列。

**坑 4：LLM 裁判不校偏**
症状：裁判偏爱长答案/先出现的答案，结论失真。
解法：随机交换顺序、固定 rubric、temperature=0、人工抽检校准一致率。

**坑 5：test 集反复跑**
症状：用 test 调参调了十轮，test 变成了 dev。
解法：dev 调参，test 只在最终验收跑；test 泄漏了就换新题。

**坑 6：消融变量混杂**
症状：同时换了数据+模板+seed，效果变了不知道谁的功劳。
解法：一次一变量；同数据版本、同 seed、同评测。

**坑 7：只测“会不会”，不测“好不好”**
症状：关键词全中但回答啰嗦/不礼貌，用户不买账。
解法：关键词/格式测“会”，LLM 裁判+人工测“好”，两层都要。

---

## 08 进阶优化：评测体系的进化方向

**① Bootstrap 置信区间**：对评测集有放回抽样 N 次，给出“通过率 95% 置信区间”，避免“差 2% 就宣称更好”。

**② 裁判校准集**：维护 50–100 条“人工已判分”的金样，每次换 rubric/裁判都先跑校准集，一致率达标才上岗。

**③ 持续评测流水线**：数据/模型变更自动触发同一套评测，报告进 CI——模型迭代从“手动跑分”变成“自动门禁”（第 44 章）。

**④ 综合分与权重**：按业务目标给任务加权（如领域 60%+通用 30%+格式 10%），用综合分做自动选型；权重本身也要评审。

**⑤ 线上指标闭环**：离线评测只是代理指标，最终以线上 A/B（满意度、采纳率、错误率）为准；定期回填线上 badcase 进评测集，让评测集跟着业务进化。

---

## 09 本章核心总结（TOP3）

**TOP1**：评测体系四层：锁版评测集（dev/test、与训练去重）+ 自动规则 + LLM 裁判 + 人工抽检；报告按任务分组，不只看总分。

**TOP2**：LLM 裁判必须防偏置：随机交换顺序、固定 rubric、temperature=0、人工校准一致率>80%；裁判不可信时先修 rubric。

**TOP3**：消融实验一次只改一个变量（同数据、同 seed、同评测）；test 集只做最终验收；评测集与 badcase 持续进化，成为模型迭代的“法庭”。

---

## 10 连载衔接

上一章（第 19 章）教会你读曲线；本章建立了“效果法庭”——从评测集到裁判再到消融，主线工程的每次微调都有了可信的判决书。

下一章攻克多轮场景：【第 21 章】多轮对话微调专项：长上下文、多轮数据与稳定性。单轮评测通过 ≠ 多轮对话不崩，多轮是另一个战场。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #效果评估 #评测集 #LLM裁判 #消融实验 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 20 微调效果评估体系（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 21 章 多轮对话微调专项*