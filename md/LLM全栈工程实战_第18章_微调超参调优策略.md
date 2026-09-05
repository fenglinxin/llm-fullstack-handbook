# 【LLM全栈工程·第18章】微调超参调优策略：rank/alpha/lr/epoch 与实验管理

> 本篇为「LLM全栈工程」连载第 18 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 同样的数据，为什么别人微调效果更好？超参调优方法论；② LoRA rank/alpha/LR/epoch 到底怎么调：一次只动一个变量；③ 调参不是玄学：实验管理、对比表与搜索策略。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「微调与对齐体系」第 5 篇：把微调超参调优从“玄学”变成“有纪律的实验” |
| 适用场景 | 个人学习（小模型扫参）；小团队微调效果优化；企业模型迭代前的调优 SOP |
| 核心知识点 | 各超参的作用与交互；搜索策略；固定 seed；实验台账与对比表 |
| 技术选型 | 手动单变量 vs 网格/随机搜索 vs Optuna 类贝叶斯；WandB/MLflow/本地 CSV |
| 分步实操 | 基线实验 → 生成调优网格 → 自动批量训练 → 解析结果表 → 选最优并归档 |
| 参数详解 | rank/alpha/dropout、LR、epoch、batch、warmup、target modules |
| 踩坑排查 | 同时动多个变量、用 loss 当唯一指标、seed 不固定、网格爆炸、只看最优不看稳定性 |
| 进阶优化 | 贝叶斯搜索、早停剪枝、分层 rank、多指标融合选优 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 19 章：过拟合、欠拟合与灾难遗忘 |

---

## 01 开篇导语

数据准备好了、模板验证过了——按下训练按钮，loss 开始下降。然后呢？

很多人在这里开始“玄学调参”：rank 改成 32 试试、LR 乘个 0.5 试试、epoch 加到 10 试试……跑了一堆实验，最后说不清哪个参数起作用，更说不清为什么。

超参调优不是碰运气，它有三个隐藏前提：

1. **你知道每个旋钮在改什么**（容量？步长？训练量？）；
2. **你一次只动一个变量**（否则结果无法归因）；
3. **你有实验台账**（配置、seed、指标全记录，能复现能对比）。

本章先讲透每个超参的“手感”，再给一套可执行的调优流程与自动化脚本——把调参变成流水线，而不是赌博。

> 一句话记住本章：**调参的目标不是“loss 最低”，而是“评测指标最好且稳定”；没有台账的调参等于没有实验。**

---

## 02 白话原理：四个旋钮在改什么

微调超参可以归成四类“力度”：

| 类别 | 旋钮 | 在改什么 |
|---|---|---|
| 容量 | lora_rank、target modules | 补丁能记多少东西 |
| 步长 | learning_rate、warmup | 每步改多猛 |
| 训练量 | epoch、batch、max_samples | 数据看几遍、每步看多少 |
| 正则 | dropout、weight decay | 防过拟合 |

**关键交互**：

- rank 小 + LR 大 → 补丁小还猛改，容易震荡；
- rank 大 + 数据少 → 容量过剩，过拟合；
- epoch 多 + 数据少 → 背题；
- LR 大 + 数据脏 → loss spike；
- batch 小 + LR 大 → 梯度噪声大。

所以调参的顺序建议是：**先定容量（rank/target）→ 再扫 LR → 再调 epoch → 最后用 dropout 微调**，每步只动一个维度。

> 记忆锚点：**rank 决定“能记多少”，LR 决定“记得多快”，epoch 决定“复习几遍”，dropout 决定“会不会背串”。**

## 03 每个旋钮的手感与参考范围

| 参数 | 参考范围 | 手感 | 调大/调小的信号 |
|---|---|---|---|
| lora_rank | 8–64（常见 16–32） | 补丁容量 | 领域题学不会→加大；训练 loss 低但通用掉→减小 |
| lora_alpha | 常 = rank×1~2 | 补丁缩放 | 跟随 rank；单独调的效果弱于 rank |
| lora_dropout | 0–0.1 | 防过拟合 | val 掉点→加大到 0.1 |
| learning_rate | LoRA 1e-4~3e-4；全参 1e-5~2e-5 | 步长 | loss 震荡→减小；收敛太慢→加大 |
| num_train_epochs | 1–5（小数据 3–5） | 复习遍数 | val loss 回升→减小/早停 |
| batch（等效） | 16–64 | 梯度质量 | 锯齿严重→加大或降 LR |
| warmup_ratio | 0.03–0.1 | 起步稳定 | 起步 spike→加大 |
| cutoff_len | 512–4096 | 看多长 | 回答被截→加大 |

> 两个高频误判：**loss 不降就加 rank**（其实先查 LR 与数据）；**效果不行就加 epoch**（其实先查数据与模板）。先归因，再动手。

---

## 04 搜索策略与实验管理

### 4.1 四种搜索策略

| 策略 | 做法 | 成本 | 适合 |
|---|---|---|---|
| 手动单变量 | 一次改一个 | 低 | 起步必做 |
| 网格搜索 | 每个参数取几个值全组合 | 中（组合爆炸） | 参数少、预算足 |
| 随机搜索 | 在范围内随机采样 | 低中 | 参数多时比网格高效 |
| 贝叶斯（Optuna 类） | 根据历史结果推荐下一组 | 中 | 预算有限、参数多 |

### 4.2 实验管理五条军规

1. **固定 seed**：同一配置跑两次要能复现（或至少差异远小于参数差异）；
2. **一次一变量**：网格里每个对比只差一个维度；
3. **全量记录**：配置 hash、数据版本、seed、指标、checkpoint 路径；
4. **评测驱动**：对比用 held-out 评测分，不只用 loss；
5. **最优要复跑**：网格最优配置用新 seed 复跑 2–3 次，取均值——防止“踩中好随机”。

> 工具层面：小团队用“CSV 台账 + Git tag”，规模上来用 WandB/MLflow（自动记录配置与曲线）；关键是**记录本身**，不是工具品牌。

## 05 分步实操：LR×Rank 小网格调优（约 1–2 小时）

沿用 llm-demo 的 sft_train 数据与 Qwen2.5-7B-Instruct。

### Step 1 先跑一个基线（10 分钟）

用第 15 章 configs/sft_industrial.yaml（lr=2e-4、rank=16、epochs=5）训练并记录 held-out 评测分，作为后续所有对比的参照。**没有基线的调参，无法判断“变好还是变坏”。**

### Step 2 写自动扫参脚本（10 分钟）

保存 scripts/run_ft_sweep.py：

~~~python
# scripts/run_ft_sweep.py —— 小网格扫参：生成配置->训练->解析结果->CSV
# 用法：python run_ft_sweep.py --dry-run   （只生成配置）
#       python run_ft_sweep.py             （逐个真实训练）
import csv
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parent.parent
sweep_dir = root / "configs" / "sweep"
sweep_dir.mkdir(parents=True, exist_ok=True)

GRID = [
    {"name": "sweep_1", "lr": 1e-4, "rank": 8, "epochs": 3},
    {"name": "sweep_2", "lr": 1e-4, "rank": 16, "epochs": 3},
    {"name": "sweep_3", "lr": 2e-4, "rank": 8, "epochs": 3},
    {"name": "sweep_4", "lr": 2e-4, "rank": 16, "epochs": 3},
]

TPL = """model_name_or_path: models/Qwen2.5-7B-Instruct
template: qwen
stage: sft
finetuning_type: lora
dataset_dir: data
dataset: sft_train
val_size: 0.2
cutoff_len: 1024
per_device_train_batch_size: 2
per_device_eval_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: __LR__
num_train_epochs: __EPOCHS__
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
seed: 42
eval_strategy: steps
eval_steps: 10
logging_steps: 5
save_steps: 20
lora_rank: __RANK__
lora_alpha: __ALPHA__
lora_dropout: 0.05
output_dir: outputs/__NAME__
plot_loss: true
"""

def make_yaml(cfg):
    text = TPL.replace("__LR__", repr(cfg["lr"]))
    text = text.replace("__RANK__", str(cfg["rank"]))
    text = text.replace("__ALPHA__", str(cfg["rank"] * 2))
    text = text.replace("__EPOCHS__", str(cfg["epochs"]))
    text = text.replace("__NAME__", cfg["name"])
    path = sweep_dir / (cfg["name"] + ".yaml")
    path.write_text(text, encoding="utf-8")
    return path

def read_last_losses(name):
    state_path = root / "outputs" / name / "trainer_state.json"
    if not state_path.exists():
        return None, None
    data = json.loads(state_path.read_text(encoding="utf-8"))
    last_loss = None
    last_eval = None
    for h in data.get("log_history", []):
        if "loss" in h and "eval_loss" not in h:
            last_loss = h["loss"]
        if "eval_loss" in h:
            last_eval = h["eval_loss"]
    return last_loss, last_eval

dry = "--dry-run" in sys.argv
rows = []
for cfg in GRID:
    yaml_path = make_yaml(cfg)
    if dry:
        print("[dry-run]", yaml_path)
        continue
    print("train:", cfg["name"])
    subprocess.run(["llamafactory-cli", "train", str(yaml_path)], check=True)
    loss, eval_loss = read_last_losses(cfg["name"])
    rows.append({"name": cfg["name"], "lr": cfg["lr"], "rank": cfg["rank"],
                 "epochs": cfg["epochs"], "last_loss": loss, "last_eval_loss": eval_loss})

if not dry:
    out_csv = root / "logs" / "ft_sweep_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "lr", "rank", "epochs", "last_loss", "last_eval_loss"])
        writer.writeheader()
        writer.writerows(rows)
    print("results ->", out_csv)
~~~

### Step 3 先干跑生成配置（2 分钟）

~~~bash
python scripts/run_ft_sweep.py --dry-run
ls configs/sweep/
cat configs/sweep/sweep_1.yaml
~~~

确认生成的 YAML 与第 15 章配置一致（只差 lr/rank/epochs），再开始真实训练。

### Step 4 跑 4 组实验（40–80 分钟）

~~~bash
python scripts/run_ft_sweep.py
cat logs/ft_sweep_results.csv
~~~

> 如果时间有限，可把 GRID 临时减到 2 组（只对比 lr），跑通流程后再扩网格——流程的价值在于“可重复、可对比”，不在于一次跑多少组。

### Step 5 用评测选最优，而不是只看 loss（10 分钟）

1. 把 4 个输出分别 export 合并（或直接带 adapter 评测）；
2. 用第 15 章 sft_eval.py 对每个模型跑 held-out 领域题+通用题；
3. 把评测分合并进结果表：

| 实验 | lr | rank | last_loss | val_loss | 领域题 | 通用题 | 结论 |
|---|---|---|---|---|---|---|---|
| sweep_1 | 1e-4 | 8 | ? | ? | ? | ? | ? |
| sweep_2 | 1e-4 | 16 | ? | ? | ? | ? | ? |
| sweep_3 | 2e-4 | 8 | ? | ? | ? | ? | ? |
| sweep_4 | 2e-4 | 16 | ? | ? | ? | ? | ? |

4. 选择标准：**领域题最高且通用题不掉**；如果并列，选训练更快的；如果最优配置的 loss 表现与次优差异极小，优先选更简单的（rank 小的）。

### Step 6 复跑确认 + 归档（10 分钟）

对最优配置换 seed（如 43、44）复跑两次，确认结果稳定（波动明显时说明数据太少或 LR 太大）；然后：

~~~bash
cd llm-demo
git add configs outputs logs scripts
git commit -m "ft-sweep: lr x rank grid (see ft_sweep_results.csv), best=sweep_X"
git tag ft-sweep-v1
~~~

> 生产提示：把“基线→网格→评测→复跑确认”沉淀为团队标准流程，任何模型迭代都走同一套——这才是调参经验的真正资产。

## 06 参数详解：微调调优速查

| 参数 | 起步值 | 搜索范围 | 说明 |
|---|---|---|---|
| lora_rank | 16 | 8/16/32/64 | 容量；先 rank 后其他 |
| lora_alpha | 32 | rank×1~2 | 跟随 rank，少单独调 |
| lora_dropout | 0.05 | 0/0.05/0.1 | val 掉点才加 |
| learning_rate | 2e-4 | 1e-4/2e-4/3e-4（LoRA） | 第二个要扫的维度 |
| num_train_epochs | 3 | 2/3/5 | 小数据别超 5 |
| 等效 batch | 16–32 | 16/32/64 | 锯齿大就加大 |
| warmup_ratio | 0.1 | 0.03/0.1 | 起步 spike 才调 |
| seed | 42 | 复跑用 43/44 | 验证稳定性 |

**调参顺序模板**：基线（rank16/lr2e-4/ep3）→ 扫 rank（8/16/32）→ 固定最优 rank 扫 lr → 固定前两者调 epoch → 最后用 dropout 微调。每一步对照 held-out 评测，而不是 loss。

---

## 07 高频踩坑排查

**坑 1：同时动三个参数**
症状：实验 A（lr×2+rank×2+epoch×2）效果变好，但不知道是谁的功劳。
解法：一次只动一个变量；网格里每个对比只差一个维度。

**坑 2：用训练 loss 当唯一指标**
症状：loss 最低的实验上线效果一般。
解法：以 held-out 领域题+通用题为准；loss 只用来判断“是否收敛/过拟合”。

**坑 3：seed 不固定**
症状：同一配置两次结果差很多，无法判断参数作用。
解法：固定 seed；最优配置换 seed 复跑取均值。

**坑 4：网格爆炸**
症状：5 参数×5 值=3125 组，预算根本不够。
解法：先单变量粗扫缩小范围，再小网格精扫；或上随机/贝叶斯搜索。

**坑 5：只盯着“最优一组”**
症状：最优实验其实只比次优高 0.5%，但复杂一倍。
解法：看整张结果表的趋势（lr 升/降方向、rank 影响），选“简单且稳定”的配置。

**坑 6：忽略数据版本**
症状：扫参中途偷偷换了数据集，结果表全部失真。
解法：每个实验记录数据版本+配置 hash；数据变更必须重跑基线。

**坑 7：不设早停就拉满 epoch**
症状：epoch=10 跑完，val loss 第 3 轮就开始回升。
解法：盯 val loss/评测，用中间最优 checkpoint，不必跑满。

---

## 08 进阶优化：调优的进化方向

**① 贝叶斯搜索（Optuna 类）**：把“下一组试什么”交给优化器，按历史结果推荐参数；配合“早停剪枝”在差配置上提前止损，同样的预算能探索 3–5 倍空间。

**② 分层 rank**：不是所有层都需要 rank 16——离输出近的层对格式影响大，可给高 rank；用分层配置在总参数量不变时提升效果。

**③ 多指标融合选优**：领域分、通用分、格式合规、推理速度合成一个“综合分”，避免单指标过拟合（第 20 章评估体系会给权重模板）。

**④ 小步快跑 + 放大**：先在 1B 模型上把参数趋势摸清（哪个方向有效），再在 7B 上精调——大模型上的实验预算留给“确认”，不留给“探索”。

**⑤ 自动实验台账**：训练启动时自动记录 Git commit、数据版本、配置 hash、seed、开始时间，结束时自动写入结果表——人工记录一定会漏，自动化不会。

---

## 09 本章核心总结（TOP3）

**TOP1**：超参分四类力度：容量（rank/target）、步长（LR/warmup）、训练量（epoch/batch）、正则（dropout）；调参顺序 = 定容量 → 扫 LR → 调 epoch → 微调正则。

**TOP2**：实验管理五军规：固定 seed、一次一变量、全量记录、评测驱动、最优复跑——没有台账的调参等于没有实验。

**TOP3**：用“结果表趋势”而不是“最优一组”做决策：看参数方向与稳定性，选简单且稳的配置；数据版本变更必须重跑基线。

---

## 10 连载衔接

上一章（第 17 章）解决了模板问题；本章建立了调参纪律与自动扫参工具——主线工程的 SFT 从“跑一次”进入“可系统优化”阶段。

下一章处理微调最常见的翻车现场：【第 19 章】过拟合、欠拟合与灾难遗忘：症状、归因与解法。loss 曲线会说话，但你要先学会听。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #超参调优 #LoRA #实验管理 #微调 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 18 微调超参调优策略（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 19 章 过拟合、欠拟合与灾难遗忘*