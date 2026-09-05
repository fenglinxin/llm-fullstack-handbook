# 【LLM全栈工程·第13章】断点续训、收敛判断与训练失败排查：把“训练翻车”变成流程

> 本篇为「LLM全栈工程」连载第 13 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 训练到一半挂了怎么办？checkpoint 设计与断点续训全流程；② loss spike、NaN、不收敛：训练日志排查手册；③ 大模型训练不是“跑完就赢”：收敛判断与失败预案。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「预训练工程·数据核心层」收官篇：建立 checkpoint、收敛判断与失败排查三套机制 |
| 适用场景 | 个人学习（单卡实验防白跑）；小团队 CPT/微调；企业预训练稳定性保障 |
| 核心知识点 | checkpoint 该存什么；断点续训的状态完整性；收敛判断指标；训练日志分析与高频故障排查 |
| 技术选型 | 保存格式与频率；日志工具（wandb/TensorBoard/自建 jsonl）；resume 方式对比 |
| 分步实操 | 设计 checkpoint 目录 → 生成样例日志 → 日志分析脚本诊断 → 断电恢复演练 → 打版本 |
| 参数详解 | 保存间隔、保留份数、异常检测阈值、收敛判据 |
| 踩坑排查 | NaN/Inf、loss spike、吞吐骤降、OOM、训练挂起、checkpoint 静默损坏 |
| 进阶优化 | 异步保存、故障自动恢复、loss 预测、健康检查与告警 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 14 章：全维度微调技术拆解 |

---

## 01 开篇导语

预训练/CPT 有一个共同特点：**贵**。一次 7B 全参训练可能烧掉几十万卡时，如果跑到第 3 天因为一个 NaN 崩了、而你没有 checkpoint——前面的钱全部白花。

更隐蔽的问题是“假成功”：loss 在降，但模型其实在背数据；loss spike 后自己恢复了，没人发现数据里混进毒样本；训练挂起 8 小时，日志里只有一行 NCCL timeout……

本章把训练稳定性拆成三件事：

1. **断点续训**：挂了能无损续跑，不丢步、不丢数据状态；
2. **收敛判断**：知道“该停了”“该调了”“该回滚了”；
3. **失败排查**：日志异常能快速归因，而不是重启硬试。

交付物：一套 checkpoint 目录规范 + 一个训练日志分析脚本 + 一次“断电恢复演练”的完整流程。

> 一句话记住本章：**训练系统要按“一定会挂、一定会出异常”来设计——checkpoint 是保险，日志是黑匣子，告警是安全带。**

---

## 02 白话原理：一次训练 = 一个“可恢复的状态机”

### 2.1 训练状态不止是“模型权重”

很多人以为 checkpoint 就是存模型，其实一次训练的运行状态包括：

| 状态 | 作用 | 丢了会怎样 |
|---|---|---|
| 模型权重 | 学到的知识 | 回到上次保存点 |
| 优化器状态 | 动量/方差 | 学习率效果打折、可能不稳定 |
| 学习率调度器 | 当前 LR 位置 | 退火节奏错乱 |
| 随机数状态 | 数据顺序/增强 | 数据重复/无法复现 |
| 数据游标 | 读到第几个样本/epoch | 样本重复或漏读 |
| 梯度缩放器 | FP16 loss scaling | FP16 训练数值异常 |

**“断点续训”= 把以上状态全部恢复，让训练像没中断过一样继续**——只存权重再加载，叫“热启动”，不叫“续训”。

### 2.2 为什么数据游标和随机状态重要

预训练通常小于 1 epoch，且要求数据不重复。如果恢复时从头重读数据，前面读过的样本会被再学一遍，等于悄悄改变了数据配比；如果 shuffle 种子不一致，后续样本顺序全变，实验不可复现。所以**工业级 checkpoint 必须包含数据加载器状态**（或等效的确定性重放方案）。

> 记住：**checkpoint 的验收标准不是“能加载”，而是“续训后的 loss 曲线与不中断时连续”。**

## 03 Checkpoint 设计：目录、频率与保留策略

### 3.1 一个规范的 checkpoint 目录

~~~text
checkpoints/exp001/
├── step-000500/
│   ├── model/                 # 模型权重（HF/Megatron 格式）
│   ├── optimizer/             # 优化器状态
│   ├── scheduler.json         # LR 位置
│   ├── rng_state.pt           # 各卡随机状态
│   ├── dataloader_state.json  # 数据游标/epoch/样本索引
│   ├── scaler.pt              # FP16 loss scaler（如使用）
│   ├── config.json            # 训练配置快照（含 Git commit）
│   └── metadata.json          # step/loss/时间/完整性校验值
├── step-001000/
└── latest/                    # 软链到最新 checkpoint
~~~

metadata.json 至少包含：step、global_batch 累计、train loss、LR、保存时间、训练代码 commit、数据版本——**没有元数据的 checkpoint 等于匿名备份**。

### 3.2 保存频率与保留策略

| 项 | 建议 |
|---|---|
| 保存间隔 | 按时间（如每 30–60 分钟）+ 按 step（如每 500–2000 步）双触发 |
| 保留份数 | 最近 N 份 + 里程碑份（如每 10% 总步数一份） |
| 异步保存 | 保存放后台线程/独立进程，避免阻塞训练 |
| 完整性校验 | 保存后立即读回校验（hash/可加载），防“静默损坏” |
| 存储位置 | 与训练机分离（对象存储/NFS），防机器一起挂 |

### 3.3 续训的两种模式

| 模式 | 做法 | 适用 |
|---|---|---|
| 精确续训 | 恢复全部状态（推荐） | 长训练、数据不能重读 |
| 热启动 | 只加载权重，重新配数据 | 换数据/换任务、微调 |

> 分布式训练（ZeRO/TP/PP）的 checkpoint 还要保存各并行维度的分片与索引，恢复时按同样的并行配置加载——**并行配置变了，旧 checkpoint 往往不能直接续训**（需先转换）。

---

## 04 收敛判断：什么时候“该停、该调、该回滚”

### 4.1 判断收敛的四条曲线

| 曲线 | 看什么 | 异常信号 |
|---|---|---|
| train loss | 拟合进度 | 不降/反弹 |
| val loss（held-out） | 泛化进度 | 与 train 拉开=过拟合/背题 |
| grad norm | 训练健康 | 爆炸/消失 |
| 吞吐与 MFU | 效率 | 骤降=IO/通信/硬件问题 |

### 4.2 三条实战经验

1. **先定“预期 loss”再开跑**：用小规模数据外推或同规模历史实验估计目标区间；实际 loss 明显偏离预期，先怀疑数据/配置而不是“多跑跑就好”；
2. **用评测闸门而非 loss 决定停点**：预训练/CPT 期间周期性跑轻量评测（每 5%–10% 步数），选“评测最优”的 checkpoint，而不是最后一步；
3. **loss 平台期不等于收敛**：平台可能来自 LR 过低、数据重复、容量不足——先查 grad norm 与数据，再决定加步数还是调参。

### 4.3 停机决策清单

- [ ] val loss 连续 N 步不降且 LR 已退火到位？
- [ ] 评测分不再上升或有回退？
- [ ] 继续训练的成本 vs 预期收益还划算？
- [ ] checkpoint 与实验记录已归档？

> 一句话：**loss 负责“告诉我还在学”，评测负责“告诉我学得值不值”，checkpoint 负责“让我随时可以重来”。**

## 05 分步实操：日志分析 + 断电恢复演练（30 分钟）

### Step 1 生成一份“带事故”的样例训练日志（5 分钟）

保存 scripts/make_sample_log.py：

~~~python
# scripts/make_sample_log.py —— 生成含 loss spike 与 NaN 的样例日志
import json
import math
import pathlib
import random

root = pathlib.Path(__file__).resolve().parent.parent
log_dir = root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
out_path = log_dir / "sample_train.jsonl"

random.seed(7)
rows = []
peak_lr = 3e-4
for step in range(3001):
    base = 4.0 * math.exp(-step / 1500.0) + 0.9
    loss = base + random.gauss(0, 0.02)
    grad = max(0.1, 1.0 + random.gauss(0, 0.1))
    if step == 1200:
        loss = base + 3.0      # 模拟 loss spike
        grad = 25.0            # 伴随梯度尖峰
    if step >= 2500:
        loss = None            # 模拟 NaN（JSON 存 null）
        grad = None
    rows.append({
        "step": step,
        "loss": loss,
        "grad_norm": grad,
        "lr": peak_lr * (1.0 - step / 3000.0),
        "tokens_per_s": 5200 + random.randint(-120, 120),
    })

with open(out_path, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("sample log ->", out_path, "lines:", len(rows))
~~~

运行：

~~~bash
python scripts/make_sample_log.py
~~~

> 真实训练日志请统一输出这种 JSONL 结构（step/loss/grad_norm/lr/tokens_per_s），一个字段一行、机器可读——这是“黑匣子”的最低要求。WandB/TensorBoard 可以照常记录，但 JSONL 永远保留一份原始副本。

### Step 2 写日志分析脚本（10 分钟）

保存 scripts/analyze_log.py：

~~~python
# scripts/analyze_log.py —— 训练日志异常检测（简易版）
import json
import pathlib
import statistics

root = pathlib.Path(__file__).resolve().parent.parent
log_path = root / "logs" / "sample_train.jsonl"

rows = []
for line in open(log_path, encoding="utf-8"):
    rows.append(json.loads(line))

def clean(rows, key):
    return [r[key] for r in rows if r.get(key) is not None]

# 用前 200 步建立“正常基线”
base_loss = clean(rows[:200], "loss")
base_grad = clean(rows[:200], "grad_norm")
loss_mean = statistics.mean(base_loss)
loss_std = max(statistics.stdev(base_loss), 1e-6)
grad_mean = statistics.mean(base_grad)
tps = clean(rows, "tokens_per_s")
tps_mean = statistics.mean(tps)

events = []
for r in rows:
    step = r["step"]
    loss = r.get("loss")
    grad = r.get("grad_norm")
    if loss is None or grad is None:
        events.append({"step": step, "type": "nan_or_missing", "suggestion": "查 FP16 loss scaling/数据 NaN/梯度爆炸；回滚到最近健康 checkpoint"})
        continue
    if grad > grad_mean * 10:
        events.append({"step": step, "type": "grad_spike", "suggestion": "查 grad clip、学习率与数据批次"})
    if step > 500 and loss > loss_mean + 5 * loss_std:
        events.append({"step": step, "type": "loss_spike", "suggestion": "查该 step 数据批次、checkpoint 一致性、精度设置"})
    if r["tokens_per_s"] < tps_mean * 0.5:
        events.append({"step": step, "type": "throughput_drop", "suggestion": "查数据加载/网络/掉卡/散热"})

# 去重并保持首次出现
seen = set()
summary = []
for ev in events:
    key = ev["type"]
    if key not in seen:
        seen.add(key)
        summary.append(ev)

report = {
    "total_steps": len(rows),
    "baseline": {"loss_mean": round(loss_mean, 3), "grad_mean": round(grad_mean, 3), "tps_mean": round(tps_mean, 1)},
    "events": summary,
}
out_path = root / "logs" / "analysis_report.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(json.dumps(report, ensure_ascii=False, indent=2))
~~~

### Step 3 跑分析，看“黑匣子”怎么报事故（5 分钟）

~~~bash
python scripts/analyze_log.py
~~~

预期输出类似：

~~~json
{
  "total_steps": 3001,
  "baseline": {"loss_mean": 3.9, "grad_mean": 1.0, "tps_mean": 5200.0},
  "events": [
    {"step": 1200, "type": "loss_spike", "suggestion": "查该 step 数据批次、checkpoint 一致性、精度设置"},
    {"step": 1200, "type": "grad_spike", "suggestion": "查 grad clip、学习率与数据批次"},
    {"step": 2500, "type": "nan_or_missing", "suggestion": "查 FP16 loss scaling/数据 NaN/梯度爆炸；回滚到最近健康 checkpoint"}
  ]
}
~~~

对照真实场景的排查顺序：**先看 grad_norm（训练健康）→ 再看 loss 曲线（拟合）→ 最后看吞吐（效率）**。样例日志里 step 1200 的 spike 与 step 2500 的 NaN 都能被自动抓住。

### Step 4 断电恢复演练（10 分钟，核心动作）

1. 用一个真实训练（或第 11 章 CPT）配置每 500 步存一次 checkpoint；
2. 跑到 step 2000 左右，直接 kill 训练进程（模拟断电/掉卡）；
3. 从最近 checkpoint 续训：

~~~text
# Hugging Face Trainer 风格（在训练脚本里指定）：
trainer.train(resume_from_checkpoint="checkpoints/exp001/step-001000")

# Megatron/DeepSpeed 风格：启动参数加
--resume-from-checkpoint checkpoints/exp001/step-001000
~~~

4. **验收续训质量**：续训后第 1 个 step 的 loss 应与中断前日志基本连续（误差 <5% 量级）；global step 从 1000 继续，而不是从 0 重来；数据游标不回退（用日志里的样本序号验证）；
5. 把演练结果写进 docs/training_runbook.md（中断时间、恢复方式、loss 连续性数据），并打版本：

~~~bash
cd llm-demo
git add logs scripts docs
git commit -m "ops: log analyzer v0.1 + resume drill runbook"
git tag training-ops-v0.1
~~~

> 生产要求：把“断电恢复演练”做成每月例行演练，像消防演习一样——**真出事那天，大家才知道流程好不好用**。

## 06 参数详解：稳定性配置速查

| 参数 | 参考取值 | 说明 |
|---|---|---|
| 保存间隔 | 30–60 分钟 或 500–2000 步 | 双触发：时间兜底 + 步数精确 |
| 保留份数 | 最近 5–10 份 + 里程碑份 | 磁盘有限时优先保“最近+里程碑” |
| 异步保存 | 开 | 避免保存阻塞训练（注意磁盘 IO 峰值） |
| 异常检测窗口 | 100–500 步 | 太小误报多，太大发现慢 |
| grad spike 倍数 | >基线 10 倍 | 与 grad clip 阈值配合 |
| loss spike 倍数 | >滚动均值 3–5 倍标准差 | 需排除退火期的正常上升 |
| val loss 早停 | 连续 N 个评估点不降（N=5–20） | 配合评测闸门使用 |
| checkpoint 校验 | 保存后立即读回 | hash + 可加载双检查 |

> 阈值要“按你的实验噪声”标定：先跑 500 步健康训练，统计 loss/grad 的波动分布，再定告警线——不要直接抄别人的数值。

---

## 07 高频踩坑排查

**坑 1：只存模型不存优化器**
症状：续训后 loss 曲线断层、收敛变慢。
解法：checkpoint 必须含模型+优化器+调度器+RNG+数据游标（第 02 节状态表全都要）。

**坑 2：checkpoint 写一半被 kill 损坏**
症状：恢复时报文件不完整/加载失败。
解法：先写临时目录再原子 rename；保存后立即读回校验；latest 用软链指向最近完整版本。

**坑 3：NaN 出现后没有自动停**
症状：NaN 之后又跑了几千步，日志被污染，只能回滚更远。
解法：监控到 NaN/Inf 立即暂停训练并告警，自动回滚到最近健康 checkpoint；不要“再跑跑看”。

**坑 4：loss spike 当成随机波动忽略**
症状：spike 后 loss 自己恢复了，但同批次毒数据还在数据池里，下次还会炸。
解法：spike 要定位到“具体数据批次”，把该批次抽出检查（坏样本、错位、重复）；只重启不查因=治标不治本。

**坑 5：吞吐骤降不查硬件**
症状：训练越来越慢，以为是数据问题，其实是散热降频或某卡掉线。
解法：监控每卡利用率/温度/NCCL 错误；tokens/s 骤降先看硬件面板再看日志。

**坑 6：续训后数据重复读**
症状：恢复后样本序号从头开始，等于悄悄改了数据配比。
解法：保存并恢复 dataloader 状态；或用“确定性分片+step 推导样本位置”的方案。

**坑 7：只在最后存一个 checkpoint**
症状：最后一步恰好是 loss spike 后的坏模型，白训一场。
解法：按保存策略留多份；用评测选最优 checkpoint，而不是默认用最后一个。

---

## 08 进阶优化：训练稳定性的进化方向

**① 异步保存 + 快照**：保存放独立线程/进程，配合文件系统快照，训练几乎无感；超大模型用“分片并行保存”缩短保存时间。

**② 自动恢复（Self-Healing）**：检测到节点故障后自动重新调度、从最近 checkpoint 续训并通知负责人；云平台/编排系统原生支持时优先用平台的。

**③ Loss 预测与偏差告警**：用小规模外推建立“预期 loss 走廊”，实际 loss 连续偏离走廊即告警——比“spike 后才发现”早几小时。

**④ 健康检查四件套**：心跳（进程活着）、吞吐（>阈值）、数值（无 NaN）、进度（step 前进）；四件套任一异常自动触发预案。

**⑤ 事故复盘库**：每次训练事故记录“症状→日志特征→根因→修复→预防”，沉淀成 team runbook；下次同类事故 10 分钟内定位，而不是重新排查一遍。

---

## 09 本章核心总结（TOP3）

**TOP1**：断点续训 = 恢复“模型+优化器+调度器+RNG+数据游标”全套状态；checkpoint 要原子写入、保留多份、保存后校验，验收标准是“续训 loss 曲线连续”。

**TOP2**：收敛判断用四条曲线（train loss/val loss/grad norm/吞吐）+ 评测闸门；loss 负责“在学”，评测负责“学得值不值”，平台期先查数据与梯度再决定是否加步数。

**TOP3**：训练日志要结构化（JSONL）并可自动分析：NaN 自动停、grad/loss spike 定位到批次、吞吐骤降查硬件；每月做一次断电恢复演练，把“翻车”变成可执行流程。

---

## 10 连载衔接

上一章（第 12 章）解决了“怎么算账、怎么选参”；本章补齐了“挂了怎么救、好了怎么停、坏了怎么查”——至此，预训练工程·数据核心层 11 章全部收官，主线工程拥有了从数据到稳定训练的完整地基。

下一章进入全新阶段：【第 14 章】全维度微调技术拆解：全参/LoRA/QLoRA/适配器怎么选。模型能力层的大幕正式拉开。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #断点续训 #训练日志 #Checkpoint #训练稳定性 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 13 断点续训、收敛判断与失败排查（本篇）

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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 14 章 全维度微调技术拆解*