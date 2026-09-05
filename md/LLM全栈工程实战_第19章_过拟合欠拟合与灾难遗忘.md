# 【LLM全栈工程·第19章】过拟合、欠拟合与灾难遗忘：症状、归因与解法

> 本篇为「LLM全栈工程」连载第 19 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① train loss 降、val loss 涨：过拟合的识别与解法；② 模型学了领域忘了通用？灾难遗忘的度量与缓解；③ 用曲线说话：欠拟合/过拟合/遗忘的三张诊断图。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「微调与对齐体系」第 6 篇：建立拟合状态诊断与灾难遗忘防控的完整方法 |
| 适用场景 | 个人学习（小数据微调）；小团队效果调优；企业模型迭代中的回归控制 |
| 核心知识点 | 过拟合/欠拟合的曲线特征；归因顺序；解法清单；灾难遗忘的度量与缓解 |
| 技术选型 | 早停/正则/容量控制/数据增强/回放/EWC 的适用边界 |
| 分步实操 | 生成三份“拟合状态”样例日志 → 自动诊断脚本 → 对真实训练日志诊断 → 遗忘回归检查 |
| 参数详解 | gap 阈值、趋势窗口、dropout/weight decay、回放比例 |
| 踩坑排查 | 用噪声当趋势、忽视 val 集大小、遗忘测错集、先调参后查数据 |
| 进阶优化 | 数据增强与合成、多任务回放、EWC/L2、LoRA 路由隔离、持续学习 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 20 章：微调效果评估体系 |

---

## 01 开篇导语

微调最让人困惑的时刻：训练 loss 一路下降，看着很完美——但一测 held-out 题，效果没涨甚至更差。

这不是模型“坏”了，而是它陷入了三种典型状态之一：**过拟合**（背会了训练题）、**欠拟合**（根本没学会）、**灾难遗忘**（学了新的忘了旧的）。三种状态的症状、成因和解法完全不同，用错药比不用药更糟。

本章给你三样工具：

1. **三张“曲线图”**：train/val loss 的形态对应哪种状态；
2. **一张归因顺序表**：先查数据、再查容量、最后查正则；
3. **一套遗忘防控**：回放、低 LR、隔离 adapter 与评测闸门。

交付物：一个能自动诊断“过拟合/欠拟合/健康”的训练日志分析脚本，以及一套灾难遗忘回归检查流程。

> 一句话记住本章：**train 与 val 的 gap 是照妖镜——gap 变大是过拟合，双双不降是欠拟合，旧任务掉分是新任务带来的遗忘。**

---

## 02 白话原理：三张曲线图

### 2.1 过拟合：背题而不是学会

| 现象 | 曲线特征 |
|---|---|
| train loss 持续下降 | ↓ |
| val loss 先降后升 | U 形 |
| gap（val-train）不断拉大 | ↗ |

原因：模型容量（rank/参数量）大于数据信息量，或同一批数据看得太多，把训练集的噪声与特例也记住了。

**SFT 里的典型表现**：训练题答得一字不差，换个问法就崩；回答里出现训练集特有的错误或风格。

### 2.2 欠拟合：还没学会

| 现象 | 曲线特征 |
|---|---|
| train loss 高且下降慢 | 平缓 ↓ |
| val loss 与 train 接近但都高 | 平行 |
| 评测分没有明显提升 | — |

原因：容量不够、LR 不合适、数据格式问题（模板错/标签错）、训练步数太少。

### 2.3 灾难遗忘：学了新的忘了旧的

| 现象 | 曲线特征 |
|---|---|
| 新任务 loss/评测涨 | ↑ |
| 旧任务（通用/历史领域）评测掉 | ↓ |

原因：继续训练时新分布覆盖旧分布，参数被“冲走”。微调小数据时表现为“通用能力回退”（例如只会按公司格式说话，不会写代码了）。

> 识别口诀：**看 gap 判过拟合，看绝对高度判欠拟合，看旧任务评测判遗忘——三条曲线配合 held-out 评测一起看。**

## 03 解法清单：按归因顺序用药

### 3.1 过拟合的解法（按优先级）

| 优先级 | 手段 | 说明 |
|---|---|---|
| 1 | 查数据质量与重复 | 重复/噪声样本是过拟合第一来源（先做去重） |
| 2 | 加数据（真实/合成） | 多样性是过拟合的天敌 |
| 3 | 降容量 | LoRA rank 减半；缩小 target modules |
| 4 | 早停 | 用 val loss/评测选中间 checkpoint |
| 5 | 加正则 | dropout 0.05→0.1、weight decay |
| 6 | 混入通用数据 | SFT 中加 10%–30% 通用指令保持泛化 |

### 3.2 欠拟合的解法（按优先级）

| 优先级 | 手段 | 说明 |
|---|---|---|
| 1 | 查数据格式/标签 | 模板错、答案错会“学不会” |
| 2 | 调 LR | 太低学不动；扫 1e-4~3e-4 |
| 3 | 加容量 | rank 16→32/64，或换全参 |
| 4 | 加训练量 | epoch 3→5（盯 val） |
| 5 | 换更强底座 | 7B→14B 或领域预训练底座 |

### 3.3 灾难遗忘的解法

| 手段 | 原理 | 代价 |
|---|---|---|
| 通用回放（replay） | 训练时混入旧任务数据 | 需要保留旧数据样本 |
| 低学习率 | 减少对旧参数的冲击 | 新任务学得慢 |
| LoRA 隔离 | 每个任务独立 adapter | 推理要路由 |
| L2/EWC 正则 | 限制参数偏离底座 | 实现复杂、超参敏感 |
| 评测闸门 | 通用评测掉分即回滚 | 需要评测集 |

> 归因顺序铁律：**先查数据（质量/模板/重复）→ 再查容量 → 最后才调正则与 LR**。90% 的“过拟合”其实是数据问题。

---

## 04 技术选型：诊断与防控工具

| 需求 | 推荐 | 说明 |
|---|---|---|
| 曲线可视化 | WandB/TensorBoard/自绘 | 必须有 train 与 val 两条线 |
| 自动诊断 | 本章 diagnose_fit.py 或自写规则 | 规则简单可解释 |
| 遗忘度量 | 固定通用评测集（锁版本） | 每次实验必跑 |
| 回放数据 | 通用语料高分桶 + 旧领域样本 | 与任务数据同批次混合 |
| 防遗忘正则 | EWC/L2（全参时） | LoRA 场景优先级低 |

## 05 分步实操：自动诊断拟合状态（30 分钟）

### Step 1 生成三份“拟合状态”样例日志（5 分钟）

保存 scripts/make_fit_samples.py：

~~~python
# scripts/make_fit_samples.py —— 生成 healthy/overfit/underfit 三份样例日志
import json
import math
import pathlib
import random

root = pathlib.Path(__file__).resolve().parent.parent
log_dir = root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
random.seed(3)

def write_log(name, train_fn, val_fn):
    rows = []
    for step in range(101):
        train_loss = train_fn(step)
        val_loss = val_fn(step) if step % 10 == 0 else None
        rows.append({"step": step, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4) if val_loss else None})
    path = log_dir / name
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("written:", path)

# 健康：train/val 同步下降，gap 稳定
write_log("fit_healthy.jsonl",
          lambda s: 2.5 * math.exp(-s / 50.0) + 0.4,
          lambda s: 2.5 * math.exp(-s / 50.0) + 0.55)

# 过拟合：train 一路降，val 先降后升
write_log("fit_overfit.jsonl",
          lambda s: 2.5 * math.exp(-s / 30.0) + 0.15,
          lambda s: 2.0 * math.exp(-s / 45.0) + 0.5 + max(0, s - 50) * 0.012)

# 欠拟合：train/val 都高且几乎不降
write_log("fit_underfit.jsonl",
          lambda s: 2.2 - s * 0.0025,
          lambda s: 2.3 - s * 0.0020)
~~~

运行：

~~~bash
python scripts/make_fit_samples.py
~~~

### Step 2 写自动诊断脚本（10 分钟）

保存 scripts/diagnose_fit.py：

~~~python
# scripts/diagnose_fit.py —— 根据 train/val loss 曲线判断拟合状态
# 用法：python diagnose_fit.py logs/fit_healthy.jsonl
import json
import pathlib
import statistics
import sys

def slope(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        return 0.0
    return (n * sxy - sx * sy) / denom

path = sys.argv[1] if len(sys.argv) > 1 else "logs/fit_healthy.jsonl"
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
train_pts = [(r["step"], r["train_loss"]) for r in rows if r.get("train_loss") is not None]
val_pts = [(r["step"], r["val_loss"]) for r in rows if r.get("val_loss") is not None]

if len(val_pts) < 3:
    print("数据不足：val 点少于 3 个，无法诊断")
    raise SystemExit

# 首 20% 与末 30% 的均值
def mean_of(points, ratio_start, ratio_end):
    n = len(points)
    seg = points[int(n * ratio_start):int(n * ratio_end)] or points[-1:]
    return statistics.mean(p[1] for p in seg)

train_first = mean_of(train_pts, 0.0, 0.2)
train_last = mean_of(train_pts, 0.7, 1.0)
val_first = mean_of(val_pts, 0.0, 0.2)
val_last = mean_of(val_pts, 0.7, 1.0)
val_slope = slope([p[0] for p in val_pts[-max(3, len(val_pts) // 3):]],
                  [p[1] for p in val_pts[-max(3, len(val_pts) // 3):]])

gap = val_last - train_last
train_drop = train_first - train_last
report = {"file": path, "train_first": round(train_first, 3), "train_last": round(train_last, 3),
          "val_first": round(val_first, 3), "val_last": round(val_last, 3),
          "gap_last": round(gap, 3), "val_slope_last": round(val_slope, 5), "train_drop": round(train_drop, 3)}

if gap > 0.4 and val_slope > 0.0005:
    report["diagnosis"] = "overfit"
    report["advice"] = "先查数据重复/噪声 -> 降 rank -> 早停 -> 加 dropout"
elif train_drop < 0.1 or (train_last > 1.2 and val_last > 1.3):
    report["diagnosis"] = "underfit"
    report["advice"] = "先查模板/数据格式 -> 调 LR -> 加容量 -> 加训练量"
else:
    report["diagnosis"] = "healthy"
    report["advice"] = "曲线健康，继续用评测确认效果"

out = pathlib.Path(path).with_suffix(".diagnosis.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
~~~

### Step 3 跑诊断（5 分钟）

~~~bash
python scripts/diagnose_fit.py logs/fit_healthy.jsonl
python scripts/diagnose_fit.py logs/fit_overfit.jsonl
python scripts/diagnose_fit.py logs/fit_underfit.jsonl
~~~

预期：healthy → healthy；overfit → overfit（gap 大且 val 末尾上行）；underfit → underfit（双高且几乎不降）。

> 真实日志接入：把第 15 章 LlamaFactory 的 trainer_state.json 里 log_history 转成 {step,train_loss,val_loss} 的 JSONL（脚本 10 行），就能用同一套诊断规则。

## 06 灾难遗忘回归检查（10 分钟，必做）

每次 SFT/CPT 实验都要回答：**通用能力掉了吗？**

1. 固定一份“通用冒烟集”（20–50 题：代码、数学、常识、写作各若干，锁版本）；
2. 微调前后各跑一遍（用第 15 章 sft_eval.py 的模式改造）；
3. 记录通过率变化：

| 模型 | 领域题 | 通用题 | 结论 |
|---|---|---|---|
| 底座 | 低 | 高 | 基线 |
| SFT 后 | 高 | ？ | 通用掉 >10% 即触发遗忘防控 |

4. 如果通用掉分：先加通用回放（训练数据混 10%–30% 通用指令）→ 再降 LR → 仍不行换 LoRA 隔离/EWC（第 03 节）。

> 灾难遗忘的“容忍线”按业务定：通用问答助手要求高（掉 2% 都难受），纯领域工具型 Agent 可以放宽。**没有容忍线的回归检查，等于没有检查。**

---

## 07 参数详解：诊断与防控速查

| 参数 | 参考取值 | 说明 |
|---|---|---|
| gap 告警线 | val-train > 0.3–0.5 | 视任务 loss 量级标定 |
| val 趋势窗口 | 末 20%–30% 的 val 点 | 太短被噪声骗 |
| 回放比例 | 10%–30% | 通用保持与领域提升的平衡点 |
| dropout | 0.05–0.1 | 过拟合时加 |
| weight decay | 0.01–0.1 | 全参微调有效，LoRA 作用弱 |
| 早停 patience | 3–10 个评测点 | 防 val 噪声误停 |
| 遗忘容忍线 | 通用分掉 2%–10% | 按业务定并写进验收标准 |

---

## 08 高频踩坑排查

**坑 1：用几个点判断趋势**
症状：val loss 抖动一次就喊“过拟合”。
解法：至少看末 20%–30% 的窗口斜率；先平滑再下结论。

**坑 2：val 集太小**
症状：val 只有 5 条，曲线全是噪声。
解法：val 至少 50–200 条（小数据可留 20% 但别少于 20 条），或做多次随机切分取平均。

**坑 3：把“答案变短”当遗忘**
症状：模型领域化后回答风格变了，误判为能力丢失。
解法：遗忘要用“固定任务集+固定评分”测，风格变化不是能力丢失；两者分开评估。

**坑 4：先调参后查数据**
症状：过拟合调了一周 dropout，最后发现训练集里有 30% 重复。
解法：归因顺序永远是数据→容量→正则。

**坑 5：SFT 一点过拟合都没有**
症状：train/val 都好，但领域题还是不会——可能是没学会（欠拟合）或数据覆盖不足。
解法：看领域 held-out 题本身；曲线健康不等于任务达标。

**坑 6：遗忘检查用“会变的评测”**
症状：每次评测题不一样，无法对比。
解法：评测集锁版本；变更评测集=重跑基线。

**坑 7：只早停不选点**
症状：早停后拿“最后一步”模型上线。
解法：用 val/评测最优的 checkpoint 上线；把 checkpoint 选择写进流程。

---

## 09 进阶优化：拟合与遗忘的进化方向

**① 数据增强与合成**：对过拟合，用同义改写/模板扩展增加多样性（注意校验事实）；比单纯加 dropout 更治本。

**② 多任务回放调度**：把“通用+旧领域+新领域”按比例与课程调度混合，兼顾新任务速度与旧任务保持。

**③ EWC/L2 防遗忘**：全参继续训练时用 EWC 估计“重要参数”，限制其偏移；实现成本高，先在回放无效时再上。

**④ LoRA 隔离与路由**：每个领域/任务独立 adapter，互不覆盖；推理按意图路由或叠加——遗忘问题从结构上消失。

**⑤ 持续学习流水线**：新数据到达→自动回放训练→通用+领域双评测→达标才合并上线；把“防遗忘”变成例行 CI 式检查。

---

## 10 本章核心总结（TOP3）

**TOP1**：三态识别口诀：gap 拉大=过拟合，双双不降=欠拟合，旧任务掉分=灾难遗忘；曲线要与固定评测集配合看，不能只看 train loss。

**TOP2**：解法按归因顺序：过拟合先查数据再降容量加正则；欠拟合先查模板格式再调 LR 加容量；遗忘用回放+低 LR+LoRA 隔离，评测闸门兜底。

**TOP3**：遗忘回归是每次实验的必做项：固定通用冒烟集、定容忍线、SFT/CPT 前后对比；没有回归检查的微调，等于把模型质量交给运气。

---

## 11 连载衔接

上一章（第 18 章）建立了调参纪律；本章让你能读懂曲线、对症下药——主线工程的微调不再“盲调”，而是“诊断驱动”。

下一章把“效果”这件事正式化：【第 20 章】微调效果评估体系：任务集、基准、消融与 LLM 裁判。怎么建评测集、怎么打分、怎么防止“评测过拟合”，是模型迭代的地基。

---

## 12 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #过拟合 #欠拟合 #灾难遗忘 #微调 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 19 过拟合、欠拟合与灾难遗忘（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 20 章 微调效果评估体系*