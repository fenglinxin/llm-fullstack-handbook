# 【LLM全栈工程·第44章】模型迭代升级与性能对标评测：让模型持续变好且可证明

> 本篇为「LLM全栈工程」连载第 44 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 模型不是训一次就完：迭代升级闭环；② 新模型凭什么替换旧模型？离线评测+在线 A/B 双证据；③ 模型注册表、影子流量与回滚：升级工程的纪律。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「全栈工程联调与项目落地」第 4 篇：模型版本迭代、对标评测与安全上线 |
| 适用场景 | 小团队模型迭代；企业 LLM 平台；算法负责人评审 |
| 核心知识点 | 迭代闭环；模型注册与 stage；离线/在线双评测；Canary/影子流量；回滚 |
| 技术选型 | 评测套件（离线）+ A/B/影子（在线）+ 模型注册表 |
| 分步实操 | 冻结基线 → 生成模型卡 → 候选评测 → 在线 A/B 分析 → Canary 上线 |
| 参数详解 | canary 比例、样本量、指标集、回滚阈值 |
| 踩坑排查 | 基线没冻结、评测集漂移、Canary 太短、只看离线、指标挑选 |
| 进阶优化 | 自动 Canary、Champion/Challenger、漂移检测、注册表 API |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 45 章：线上问题闭环排查 |

---

## 01 开篇导语

模型上线不是终点，而是持续迭代的起点：业务在变、badcase 在积累、底座模型在升级——**模型必须像软件一样有版本、有评测、有灰度、有回滚。**

本章把迭代变成一条纪律严明的流水线：

1. **离线证据**：同一套锁版评测集，候选 vs 基线；
2. **在线证据**：影子流量或小流量 A/B，用真实用户指标说话；
3. **升级纪律**：Canary 比例、回滚阈值、模型卡与 stage 管理。

> 一句话记住本章：**换模型的唯一理由是“双证据通过”：离线评测不倒退、在线指标显著更好；没有证据的升级叫赌博。**

---

## 02 白话原理：迭代闭环与双证据

### 2.1 迭代闭环

~~~text
线上 badcase/反馈
  → 归因（数据缺？格式错？知识旧？）
  → 更新数据/配置
  → 训练候选 v2
  → 离线评测（v2 vs v1）
  → 影子/A-B（在线）
  → Canary 上线 → 监控 → 全量/回滚
~~~

### 2.2 双证据原则

| 证据 | 回答 | 工具 |
|---|---|---|
| 离线评测 | 能力是否倒退 | 锁版 eval_suite |
| 在线实验 | 用户是否更满意 | A/B/影子流量 |
| 性能成本 | 划不划算 | bench + 成本模型 |

> 记忆锚点：**离线证明“不变差”，在线证明“变更好”，成本证明“值得换”——三条都过才发布。**

## 03 模型注册与升级计划

### 3.1 模型 stage

| Stage | 含义 | 谁可用 |
|---|---|---|
| dev | 实验模型 | 算法团队 |
| staging | 候选通过离线评测 | 测试/评审 |
| production | 线上服役 | 全量 |

每次候选升级：dev→staging 要离线闸门；staging→production 要在线证据。

### 3.2 升级检查单

- [ ] 基线版本冻结（v1 模型卡+评测报告存档）
- [ ] 候选 v2 离线评测：领域≥v1、通用不掉、安全通过
- [ ] 性能成本：时延/吞吐/单次成本可接受
- [ ] 在线实验设计：样本量、指标、时长
- [ ] Canary 计划与回滚预案

### 3.3 Canary 与 A/B 设计

| 参数 | 参考 |
|---|---|
| Canary 比例 | 5%→20%→50%→100% |
| 观察时长 | 至少覆盖一个业务周期（如 3–7 天） |
| 核心指标 | 采纳率/满意度/错误率/时延/成本 |
| 回滚阈值 | 核心指标显著变差即回滚 |

> 样本量要先算：想检测 2% 的指标提升，需要足够的样本；实验太短/太少得出的“更好”不可信。

## 05 分步实操：模型卡 + 在线 A/B 分析（约 40 分钟）

### Step 1 生成模型卡（10 分钟）

保存 scripts/model_card.py：

~~~python
# scripts/model_card.py —— 生成模型卡（版本/stage/评测摘要）
# 用法：python model_card.py <模型名> <dev|staging|production>
import datetime
import json
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent
name = sys.argv[1]
stage = sys.argv[2] if len(sys.argv) > 2 else "dev"

eval_path = root / "logs" / ("eval_report_" + name + ".json")
lines = ["# 模型卡：" + name, "", "- 版本: " + name, "- Stage: " + stage,
         "- 生成时间: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), ""]

if eval_path.exists():
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    lines.append("## 离线评测")
    lines.append("- 通过率: %d/%d" % (data.get("pass", 0), data.get("total", 0)))
    lines.append("- 分任务: " + json.dumps(data.get("by_task", {}), ensure_ascii=False))
else:
    lines.append("## 离线评测")
    lines.append("- 未找到评测报告：请先运行 eval_suite")

lines += ["", "## 说明", "- 进入 production 前需：离线通过 + 在线实验证据 + 回滚预案", ""]
out = root / "docs" / ("model_card_" + name + ".md")
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:12]))
print("->", out)
~~~

运行：

~~~bash
python scripts/model_card.py helpdesk-sft-v1 staging
cat docs/model_card_helpdesk-sft-v1.md
~~~

> 模型卡是迭代的“身份证”：v1 冻结后，v2 的每一版升级都先更新自己的卡，再进评审。

### Step 2 在线 A/B 分析脚本（15 分钟）

保存 scripts/ab_compare.py：

~~~python
# scripts/ab_compare.py —— A/B 指标 Bootstrap 对比
# 用法：python ab_compare.py [csv路径]  （无 csv 时生成示例）
import csv
import json
import pathlib
import random
import sys

root = pathlib.Path(__file__).resolve().parent.parent
csv_path = sys.argv[1] if len(sys.argv) > 1 else str(root / "logs" / "ab_sample.csv")
random.seed(7)

if not pathlib.Path(csv_path).exists():
    rows = [("control", 1 if random.random() < 0.70 else 0) for _ in range(300)]
    rows += [("variant", 1 if random.random() < 0.74 else 0) for _ in range(300)]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "metric"])
        writer.writerows(rows)
    print("生成示例数据 ->", csv_path)

groups = {"control": [], "variant": []}
with open(csv_path, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["variant"] in groups:
            groups[row["variant"]].append(float(row["metric"]))

def bootstrap_diff(a, b, n=2000):
    diffs = []
    for _ in range(n):
        sa = [random.choice(a) for _ in a]
        sb = [random.choice(b) for _ in b]
        diffs.append(sum(sb) / len(sb) - sum(sa) / len(sa))
    diffs.sort()
    return diffs[int(n * 0.025)], diffs[int(n * 0.975)]

mean_c = sum(groups["control"]) / len(groups["control"])
mean_v = sum(groups["variant"]) / len(groups["variant"])
lo, hi = bootstrap_diff(groups["control"], groups["variant"])
report = {"control_mean": round(mean_c, 4), "variant_mean": round(mean_v, 4),
          "diff": round(mean_v - mean_c, 4), "ci95": [round(lo, 4), round(hi, 4)],
          "significant": not (lo <= 0 <= hi)}
with open(root / "logs" / "ab_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
~~~

运行：

~~~bash
python scripts/ab_compare.py
cat logs/ab_report.json
~~~

**判读**：diff>0 且 95% CI 不含 0 → variant 显著更好；CI 含 0 → 差异不显著，别急着发布；真实实验请用线上分流日志替换示例数据。

### Step 3 Canary 与回滚演练（10 分钟）

1. 候选 v2 过离线闸门后，按 5%→20%→50% 放量；
2. 每档观察核心指标（采纳率/错误率/时延），用 ab_compare 对比基线与候选；
3. 任一指标触发回滚阈值 → 切回 v1（第 41 章 release tag）；
4. 演练后归档：

~~~bash
cd llm-demo
git add scripts logs docs
git commit -m "iter: model card + AB bootstrap + canary drill"
git tag iter-v1
~~~

> 纪律：**Canary 期间禁止同时改 prompt/数据/框架**——一次只验证一个变量，否则无法归因。

## 06 参数详解：迭代评测速查

| 参数 | 参考 | 说明 |
|---|---|---|
| Canary 比例 | 5→20→50→100% | 每档观察 1–3 天 |
| 观察周期 | ≥一个业务周期 | 避免周内波动误判 |
| 核心指标 | 2–4 个 | 太多会“挑着说” |
| 回滚阈值 | 核心指标显著变差 | 预先写死 |
| Bootstrap 次数 | 1000–2000 | 稳定 CI |
| 评测集锁版 | 每次候选同套 | 变更=重跑基线 |

---

## 07 高频踩坑排查

**坑 1：基线没冻结**
症状：v1 模型和评测报告找不到了，无法对比。
解法：v1 发布时冻结模型卡+报告+tag（第 41 章）。

**坑 2：评测集漂移**
症状：每轮偷偷加题，v2 高分其实是题变简单。
解法：评测集版本化；变更必须重跑 v1。

**坑 3：只看离线**
症状：离线+5 分，上线用户不买账。
解法：离线+在线双证据；以线上指标为最终裁判。

**坑 4：Canary 太短**
症状：周二上线周五宣布成功，没覆盖周末峰值。
解法：至少覆盖一个完整业务周期。

**坑 5：指标挑选**
症状：10 个指标里挑 2 个涨的说成功。
解法：核心指标预注册；全部指标进报告。

**坑 6：样本量不足**
症状：100 个样本差 2% 就宣称显著。
解法：Bootstrap/显著性检验；先算样本量。

**坑 7：Canary 期间乱改**
症状：同时换了模型和 prompt，无法归因。
解法：一次只变一个变量。

---

## 08 进阶优化：迭代体系进化方向

**① 自动 Canary**：指标达标自动放量、触发阈值自动回滚——把升级变成 CI/CD 的一步。

**② Champion/Challenger**：线上常驻冠军，新模型作为挑战者自动评测，赢者上位。

**③ 漂移检测**：监控输入分布/指标漂移，触发“该重新训练了”的信号。

**④ 注册表 API**：模型注册表提供 stage 流转与部署 API，服务按 stage 拉取模型，杜绝手工拷贝。

**⑤ badcase 自动回流**：线上低分回答自动进数据管道（脱敏→标注→下一轮训练），迭代闭环完全自动化。

---

## 09 本章核心总结（TOP3）

**TOP1**：升级需要双证据：离线锁版评测不倒退 + 在线 A/B 显著更好，再叠加性能成本可接受——三条都过才发布。

**TOP2**：模型像软件一样管理：模型卡+stage（dev/staging/production）+release tag；Canary 5→100% 每档观察，指标预注册、回滚阈值写死。

**TOP3**：Canary 期间一次只变一个变量；样本量不足不宣称显著；评测集版本化，变更必须重跑基线。

---

## 10 连载衔接

上一章（第 43 章）搭好生产拓扑；本章让模型进入“可证明的持续升级”循环——v1、v2、Canary、回滚，一切有据可查。

下一章处理“出问题时怎么办”：【第 45 章】线上问题闭环排查：监控、告警与应急手册。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #模型迭代 #A/B测试 #Canary #模型注册 #评测 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 44 模型迭代升级与性能对标评测（本篇）
- 45 线上问题闭环排查
- 46 工程化最佳实践汇总（手册终章）

---

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 45 章 线上问题闭环排查*