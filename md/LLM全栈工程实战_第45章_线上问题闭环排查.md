# 【LLM全栈工程·第45章】线上问题闭环排查：监控、告警与应急手册

> 本篇为「LLM全栈工程」连载第 45 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 模型上线后怎么盯？LLM 专属监控指标与 SLO；② 线上翻车怎么救？症状→排查→应急手册；③ 从告警到复盘：线上问题闭环与 badcase 回流。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「全栈工程联调与项目落地」第 5 篇：线上监控、告警与应急闭环 |
| 适用场景 | 小团队上线运维；企业 LLM 平台 SRE；值班工程师 |
| 核心知识点 | 可观测三支柱；LLM 指标；SLO/告警；故障排查手册；复盘闭环 |
| 技术选型 | Prometheus/Grafana 类监控；日志与 trace；告警渠道 |
| 分步实操 | 定义 SLO → 样例指标合规分析 → 生成应急手册 → 故障演练与复盘 |
| 参数详解 | TTFT/TPOT 阈值、错误率、SLO 窗口、告警级别 |
| 踩坑排查 | 只看平均、告警轰炸、没有 runbook、故障不复盘 |
| 进阶优化 | 自动回滚、质量漂移检测、badcase 自动回流 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 46 章：工程化最佳实践汇总（手册终章） |

---

## 01 开篇导语

模型上线后，最怕的不是出问题，而是**出了问题没人知道、知道了不知道怎么查、查完了不改进**。

LLM 服务的问题分层很清晰：

- **平台层**：OOM、掉卡、队列爆；
- **性能层**：TTFT/TPOT 劣化、排队；
- **质量层**：胡言乱语、拒答率飙升、空回复；
- **成本层**：重试循环、token 暴涨。

本章建立三层机制：

1. **监控**：指标+日志+trace 三支柱，LLM 专属指标；
2. **告警与应急**：SLO 阈值与症状→排查手册；
3. **闭环**：复盘→badcase 回流→回归测试（接第 44 章迭代）。

> 一句话记住本章：**监控让你“看得见”，告警让你“来得及”，手册让你“查得对”，复盘让你“不再犯”。**

---

## 02 白话原理：LLM 可观测三支柱

| 支柱 | 看什么 | 例子 |
|---|---|---|
| 指标 | 数字趋势 | TTFT P95、错误率、KV 用量 |
| 日志 | 单请求细节 | 请求/响应/错误栈 |
| Trace | 跨服务链路 | 网关→推理→RAG 各段耗时 |

LLM 专属指标：

- 性能：TTFT/TPOT/ITL/排队时延；
- 容量：KV Cache 用量、GPU 利用率、max-num-seqs 打满率；
- 质量：空回复率、拒答率、格式错误率、负反馈率；
- 成本：token 数、单请求成本、重试率。

> 记忆锚点：**通用监控看“挂没挂”，LLM 监控还要看“答得好不好、贵不贵”——质量与成本指标是 LLM 特有的第四维。**

## 03 SLO、告警与症状速查

### 3.1 SLO 示例（按业务调整）

| 指标 | SLO | 告警线 |
|---|---|---|
| 可用性 | 99.9% | <99.5% 持续 5min |
| TTFT P95 | <800ms | >1.5s 持续 5min |
| TPOT P95 | <60ms/token | >120ms |
| 错误率 | <1% | >2% |
| 空回复/拒答率 | <3% | >5% |
| 单请求成本 | <预算线 | 超线 20% |

### 3.2 告警分级

| 级别 | 响应 | 例子 |
|---|---|---|
| P0 | 立即 | 全站不可用/质量崩溃 |
| P1 | 15 分钟 | 错误率飙升/TTFT 劣化 |
| P2 | 当天 | 成本超线/个别模型问题 |
| P3 | 记录 | 轻微漂移 |

### 3.3 高频症状速查

| 症状 | 先查 | 常见根因 |
|---|---|---|
| 全站超时 | 部署/回滚 | 新版本/实例全挂 |
| TTFT 飙升 | 队列/实例数 | 流量突增/实例不足 |
| 输出变慢 | GPU/TPOT | 掉卡/量化回退/热降频 |
| 胡言乱语 | 模型版本/模板/量化 | 发错模型/配置错 |
| 拒答率飙升 | 安全策略/prompt | 规则误伤/模型漂移 |
| 成本暴涨 | 重试/死循环 | 客户端重试/超长输出 |

> 铁律：**先恢复服务（回滚/扩容），再查根因；复盘时把根因变成回归测试与 badcase，进入第 44 章迭代闭环。**

## 05 分步实操：SLO 合规分析与应急手册（约 30 分钟）

### Step 1 SLO 合规分析脚本（10 分钟）

保存 scripts/llm_slo.py：

~~~python
# scripts/llm_slo.py —— LLM 指标 SLO 合规分析
# 用法：python llm_slo.py [指标csv]（缺省生成样例）
import csv
import json
import pathlib
import random
import statistics
import sys

root = pathlib.Path(__file__).resolve().parent.parent
csv_path = sys.argv[1] if len(sys.argv) > 1 else str(root / "logs" / "metrics_sample.csv")
random.seed(9)

if not pathlib.Path(csv_path).exists():
    rows = []
    for _ in range(1000):
        rows.append({"ttft_ms": max(50, random.gauss(300, 120)),
                     "error": 1 if random.random() < 0.008 else 0,
                     "empty": 1 if random.random() < 0.01 else 0})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ttft_ms", "error", "empty"])
        writer.writeheader()
        writer.writerows(rows)

rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
n = len(rows)
ttfts = sorted(float(r["ttft_ms"]) for r in rows)
errors = sum(int(r["error"]) for r in rows)
empties = sum(int(r["empty"]) for r in rows)

def pct(values, p):
    return values[min(len(values) - 1, int(len(values) * p))]

report = {
    "total": n,
    "availability": round(1 - errors / max(1, n), 4),
    "error_rate": round(errors / max(1, n), 4),
    "empty_rate": round(empties / max(1, n), 4),
    "ttft_p50_ms": round(pct(ttfts, 0.5), 1),
    "ttft_p95_ms": round(pct(ttfts, 0.95), 1),
    "slo": {"ttft_p95_ok": pct(ttfts, 0.95) < 800,
            "error_ok": errors / max(1, n) < 0.02,
            "empty_ok": empties / max(1, n) < 0.03},
}
with open(root / "logs" / "slo_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
~~~

运行：

~~~bash
python scripts/llm_slo.py
cat logs/slo_report.json
~~~

> 生产用法：把 Prometheus 等监控的指标导出为 CSV/直接查询，按同一套阈值算 SLO；阈值写在配置里而不是脚本里。

### Step 2 应急手册生成器（10 分钟）

保存 scripts/runbook_gen.py：

~~~python
# scripts/runbook_gen.py —— 按症状生成应急步骤
# 用法：python runbook_gen.py <latency|quality|outage|cost>
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent
symptom = sys.argv[1] if len(sys.argv) > 1 else "latency"

RUNBOOK = {
    "outage": ["确认部署版本与实例状态", "回滚到上一 release tag（第 41 章）", "恢复后查根因：新模型/配置/依赖"],
    "latency": ["看队列深度与实例数", "扩容/摘除慢实例", "查 TTFT/TPOT 分段（网关/推理/网络）", "看是否掉卡/热降频/量化回退"],
    "quality": ["确认线上模型版本与评测版本一致", "抽查日志里的输入输出（模板/量化）", "触发安全策略? 查拒答率", "收集 badcase 进第 44 章迭代"],
    "cost": ["查重试率与死循环", "查输出 token 分布/超长输出", "查是否多条 prompt 重复请求", "设成本告警与上限"],
}

lines = ["# 线上应急手册", "", "## 症状：" + symptom, ""]
for step in RUNBOOK.get(symptom, RUNBOOK["latency"]):
    lines.append("- [ ] " + step)
lines.append("")
lines.append("## 复盘要求", "- 时间线/根因/影响/改进项", "- badcase 与回归测试回流", "")

out = root / "docs" / "incident_runbook.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("->", out)
~~~

运行并生成多个症状手册：

~~~bash
python scripts/runbook_gen.py latency
python scripts/runbook_gen.py quality
cat docs/incident_runbook.md
~~~

### Step 3 故障演练与复盘（10 分钟）

1. 用第 34 章“kill 实例”与第 44 章“Canary 回滚”做两次演练，对照手册打勾；
2. 演练后写复盘（时间线/根因/改进项），把根因变成评测集新题（第 20 章）与回归测试；
3. 归档：

~~~bash
cd llm-demo
git add scripts logs docs
git commit -m "ops: SLO checker + runbook gen + drills"
git tag ops-v1
~~~

> 复盘不是追责：**目标是把“这次踩的坑”变成“下次不会踩的测试与手册”。**

## 06 参数详解：监控与应急速查

| 参数 | 参考 | 说明 |
|---|---|---|
| 可用性 SLO | 99.9% | 按业务调 |
| TTFT P95 | <800ms | 业务敏感可更严 |
| TPOT P95 | <60ms/token | 流式体验 |
| 空回复/拒答率 | <3% | 质量红线 |
| 告警窗口 | 持续 5min 才告 | 防抖动轰炸 |
| 复盘时限 | 48h 内 | 趁热打铁 |

---

## 07 高频踩坑排查

**坑 1：只看平均**
症状：均值正常，P99 已崩。
解法：P50/P95/P99 全看；排队与错误率分开。

**坑 2：告警轰炸**
症状：一抖动就告警，值班人麻木。
解法：持续窗口+分级+收敛；告警也要演练。

**坑 3：没有 runbook**
症状：半夜告警，值班人现查文档。
解法：高频症状预写手册（本章生成器），每年演练两次。

**坑 4：只监控平台不监控质量**
症状：服务 100% 可用，回答全是胡话。
解法：加质量指标（空回复/拒答/负反馈/抽样评测）。

**坑 5：故障不复盘**
症状：同一个坑一年踩三次。
解法：48h 复盘+改进项进迭代与测试。

**坑 6：回滚太慢**
症状：明知新版本有问题还排查两小时。
解法：先回滚再排查；回滚演练常态化。

**坑 7：指标与评测脱节**
症状：线上指标好但评测集没更新，漂移看不见。
解法：线上 badcase 定期回流评测集（第 44 章）。

---

## 08 进阶优化：线上运营进化方向

**① 自动回滚**：核心指标触发阈值自动切回上一 release，人工只需确认。

**② 质量漂移检测**：对线上输出做抽样自动评测（关键词/LLM 裁判），质量分低于基线自动告警。

**③ badcase 自动回流**：负反馈/低分回答自动进入数据管道，下一轮训练自动包含（闭环第 44 章）。

**④ 成本实时看板**：按模型/租户/请求类型拆成本，超预算自动限流。

**⑤ 全链路 Trace**：request_id 贯穿网关→推理→RAG，任何“慢在哪/错在哪”一键定位。

---

## 09 本章核心总结（TOP3）

**TOP1**：LLM 监控 = 平台指标 + 性能指标 + 质量指标 + 成本指标；SLO 要有 P50/P95/P99 与错误率，别只看平均。

**TOP2**：告警分级 + 持续窗口 + 预写 runbook；故障处理顺序是“先恢复（回滚/扩容），再查根因”。

**TOP3**：复盘必须闭环：48 小时内写时间线/根因/改进项，根因变成评测题与回归测试——让同样的故障只发生一次。

---

## 10 连载衔接

上一章（第 44 章）建立了迭代升级纪律；本章让线上“看得见、来得及、查得对、不再犯”——运营闭环完成。

下一章是 46 章旅程的终点：【第 46 章】工程化最佳实践汇总：把整条生产线沉淀成团队可复用的手册与 SOP。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #监控告警 #SLO #应急手册 #可观测性 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 39 批量调度与并行深度适配
- 40 内核重构与编译级量化

第 5 阶段·全栈工程联调与项目落地
- 41 全链路串联：数据到上线一体化流程
- 42 领域大模型定制化落地
- 43 高并发生产适配与端侧部署
- 44 模型迭代升级与性能对标评测
- 45 线上问题闭环排查（本篇）
- 46 工程化最佳实践汇总（手册终章）

---

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 46 章 工程化最佳实践汇总（手册终章）*