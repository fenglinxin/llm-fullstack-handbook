# 【LLM全栈工程·第12章】预训练超参选型与硬件适配：把每一分算力花在刀刃上

> 本篇为「LLM全栈工程」连载第 12 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 训练不收敛先别怪代码：学习率、Batch 与精度选型；② 7B 全参训练要多少显存？一张表算清硬件账；③ 从单卡到多卡：DP/ZeRO/TP/PP 到底怎么选。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「预训练工程·数据核心层」第 10 篇：把训练超参与硬件规模对应起来，支撑 CPT/预训练方案落地 |
| 适用场景 | 个人学习（单卡 LoRA/小模型）；小团队选型 GPU；企业预训练/CPT 硬件规划与预算 |
| 核心知识点 | 关键超参（batch/LR/精度/优化器）；显存估算公式；并行策略矩阵；MFU 与硬件匹配 |
| 技术选型 | 全参 vs ZeRO/FSDP vs TP/PP/CP；单卡/单机多卡/多节点硬件档位 |
| 分步实操 | 填实验卡 → 显存估算脚本 → 并行与超参决策 → 200 步烟雾测试 → 记录归档 |
| 参数详解 | 峰值 LR、warmup、batch tokens、beta2、weight decay、grad clip、精度、序列长度 |
| 踩坑排查 | LR 过高 spike、显存算漏激活、TP 无 NVLink、FP16 溢出、只加卡不查数据、不跑烟雾测试 |
| 进阶优化 | 小规模超参迁移、激活重算、序列打包、上下文并行、自动并行与调度 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 13 章：断点续训、收敛判断与训练失败排查 |

---

## 01 开篇导语

上一章你跑通了 LoRA CPT，但心里一定有个疑问：**如果我要做真正的全参预训练/CPT，需要多大的机器？学习率设多少？8 张卡怎么分工？**

这些问题没有“标准答案”，但有**标准方法**。预训练超参选型的本质是：在“显存、算力、网络”三条硬约束下，找到一组让 loss 稳定下降的参数；硬件适配的本质则是：**先算清账，再花钱，不买用不上的卡，也不让卡闲着。**

本章给你三样东西：

1. 一套超参选择的“坐标系”（batch、LR、精度、优化器）；
2. 一张硬件与并行策略的决策地图（从单卡到多节点）；
3. 一个 10 分钟能跑的显存估算与实验规划脚本。

> 一句话记住本章：**超参是“油门”，硬件是“车道”，数据管道是“路况”——三者匹配才能又快又稳；先小规模试跑，再放大投产。**

---

## 02 白话原理：训练系统 = 油门 × 车道 × 路况

一次预训练/CPT 的每个 step，可以用三个视角拆开看：

**① 数据视角（每步看多少）**
全局 batch（token 数）= 单卡 micro-batch × 梯度累积 × 卡数 × 序列长度。它决定梯度估计的稳定性和算力利用率。

**② 优化视角（每步改多少）**
学习率决定参数更新幅度；warmup 让起步平稳；cosine 衰减让后期收敛；weight decay 控制参数不过度膨胀；grad clip 防梯度爆炸。它们共同决定 loss 曲线形态。

**③ 硬件视角（装得下、跑得动）**
显存决定模型/优化器/激活能不能放下；算力决定单位时间能处理多少 token；卡间网络决定并行方案的上限。**先算显存，再谈并行**——这是硬件规划的第一性原理。

> 超参与硬件的匹配铁律：**先在小规模（少卡/短序列/少步数）把“参数组合”跑稳，再按比例放大到目标规模**。直接在大集群上调参，一次失误可能烧掉几十万。

## 03 超参选型：七个关键旋钮

### 3.1 七个旋钮一览

| 旋钮 | 作用 | 选型逻辑 |
|---|---|---|
| 全局 batch（tokens） | 每步梯度估计质量 | 太小震荡，太大效率低；与 LR 联动 |
| 峰值学习率 | 更新幅度 | 随模型变大而下降；先小规模扫参 |
| warmup | 起步稳定 | 通常 0.5%–2% 总步数（或数百步） |
| 衰减策略 | 后期收敛 | cosine 常用；预训练末期可接数据退火 |
| 优化器 | 参数更新算法 | 预训练常用 AdamW；beta2 常用 0.95–0.999 |
| weight decay | 防参数膨胀 | 预训练常见 0.1 量级（AdamW） |
| 精度 | 显存/数值 | BF16 主流；FP16 需 loss scaling；FP8 前沿 |

### 3.2 学习率参考区间（按规模）

| 模型规模 | 峰值 LR（参考） | 全局 batch（tokens，参考） |
|---|---|---|
| <1B | 6e-4 ~ 1e-3 | 0.25M – 1M |
| 1B–10B | 2e-4 ~ 4e-4 | 0.5M – 4M |
| 10B–70B | 1e-4 ~ 2e-4 | 2M – 16M |
| 70B+ | 5e-5 ~ 1e-4 | 4M – 60M |

> 这些是“经验参考坐标系”，不是真理：你的数据质量、模型结构、优化器细节都会移动最优值。**正确姿势是用小规模扫参 + 迁移**（见本章进阶优化），而不是照抄某家大厂的数。

### 3.3 Batch 与 LR 的联动

经验规律：batch 翻倍时，LR 可以按平方根或线性比例上调（不同学派有不同建议）。工程上更稳的做法是：**固定 batch，先扫 LR（如 5 个档位各跑 500 步看 loss 曲线），再固定 LR 调 batch**——一次只动一个变量，才能归因。

### 3.4 精度选择的坑位提醒

- BF16：动态范围与 FP32 接近，训练稳定，Ampere 及以上 GPU 的主流选择；
- FP16：需要动态 loss scaling，处理不当会出现 loss spike/NaN；
- FP8：省显存提吞吐，但对数值敏感，通常大厂成熟 pipeline 才用；
- 混合精度下优化器状态仍建议保留 FP32 主权重，否则小学习率下可能不收敛。

> 判断精度是否正常的快捷方式：前 1000 步 grad norm 应保持健康量级（不趋 0、不爆炸）；出现诡异 spike 时先怀疑精度与 grad clip，再怀疑数据。

## 04 硬件适配：先算账，再花钱

### 4.1 显存估算：一张公式表

| 项目 | 估算 | 说明 |
|---|---|---|
| 模型权重（BF16） | 参数量 × 2 字节 | 7B ≈ 14GB |
| 梯度（BF16） | 参数量 × 2 字节 | 与权重同量级 |
| AdamW 优化器状态 | 参数量 × 12 字节左右 | FP32 主权重 + 一阶/二阶矩 |
| 小计（全参 BF16+AdamW） | 参数量 × 16 字节 | 7B ≈ 112GB，单卡必爆 |
| 激活值 | 与 batch×序列×层数相关 | 用激活重算（activation checkpointing）可大幅下降 |
| LoRA/QLoRA | 只训 adapter，底座可冻结/量化 | 7B QLoRA 单卡 24GB 可跑 |

**核心结论**：7B 全参 BF16+AdamW 约需 112GB 以上（未含激活），单张 80GB 卡也放不下——必须 ZeRO/FSDP 分片或 CPU offload；而 LoRA/QLoRA 把“要训练的状态”降到 1% 以内，单卡就能跑。**先明确“全参还是 LoRA”，显存账完全不同。**

### 4.2 硬件档位与并行策略

| 档位 | 典型配置 | 能干的事 | 并行建议 |
|---|---|---|---|
| 入门单卡 | 1×24GB | 7B QLoRA/LoRA CPT、SFT、推理 | 不需要并行 |
| 单机多卡 | 4–8×80GB（NVLink） | 7B–13B 全参 CPT；34B 需 ZeRO+offload 谨慎评估 | ZeRO/FSDP 起步，必要时 TP |
| 多节点 | 16–128×80GB+（IB 网络） | 34B–70B+ 预训练 | ZeRO + TP/PP 组合 |
| 大集群 | 数百卡 H100 级 | 百亿以上旗舰预训练 | TP+PP+CP+DP 全上 |

### 4.3 并行策略选择表

| 策略 | 拆什么 | 需要什么 | 什么时候用 |
|---|---|---|---|
| DP（数据并行） | 每卡一份模型，切数据 | 普通网络即可 | 模型单卡放得下 |
| ZeRO/FSDP | 分片优化器/梯度/参数 | 节点内高速互联更佳 | 模型单卡放不下但可全参训练 |
| TP（张量并行） | 把每层拆到多卡 | NVLink/高速互联 | 单卡放不下单层，卡间带宽高 |
| PP（流水线并行） | 不同层放不同卡 | 带宽要求低于 TP | 多节点、层数多 |
| CP（上下文并行） | 沿序列长度切分 | 高速互联 | 超长序列（128K+） |

> 网络是隐形瓶颈：TP 需要 NVLink 级互联，跨节点硬上 TP 会慢到怀疑人生；单机 8 卡优先 ZeRO/FSDP，多节点再上 PP+TP 组合。

### 4.4 MFU：算力有没有被浪费

MFU（Model FLOPs Utilization）衡量“显卡理论算力被用上多少”。成熟集群 40%–60% 已属优秀量级（随硬件/模型浮动），如果长期只有个位数，先查：数据加载、卡间通信、kernel 效率、batch 太小——**加卡之前先查 MFU**，否则加再多卡也是空转。

## 05 分步实操：10 分钟做出训练方案

### Step 1 填“实验卡”（2 分钟）

训练前先回答六个问题（写进 configs/experiment_card.md）：

| 问题 | 示例答案 |
|---|---|
| 模型规模 | 7B |
| 训练方式 | LoRA CPT（单卡） / 全参 CPT（8 卡） |
| 数据预算 | 50 亿 token（CPT 小规模可 1–5 亿） |
| 硬件 | 1×RTX 4090 24GB 或 8×A100 80GB |
| 目标评测 | 领域评测 + 通用评测 |
| 时间/成本上限 | 3 天 / 预算 X |

### Step 2 跑显存估算脚本（5 分钟）

保存 scripts/plan_pretrain.py：

~~~python
# scripts/plan_pretrain.py —— 显存估算 + 并行建议
# 用法：python plan_pretrain.py <参数量B> <full|lora> <卡数> <单卡显存GB>
import json
import pathlib
import sys

params_b = float(sys.argv[1]) if len(sys.argv) > 1 else 7.0
mode = sys.argv[2] if len(sys.argv) > 2 else "full"
n_gpu = int(sys.argv[3]) if len(sys.argv) > 3 else 1
vram_gb = int(sys.argv[4]) if len(sys.argv) > 4 else 80

GB = 1024 ** 3
params = params_b * 1e9

if mode == "full":
    weights = params * 2      # BF16 权重
    grads = params * 2        # BF16 梯度
    optimizer = params * 12   # AdamW：FP32 主权重+一阶+二阶
    total = weights + grads + optimizer
    note = "全参 BF16 + AdamW（未含激活值；激活用 checkpointing 控制）"
else:
    weights = params * 2      # 冻结底座 BF16
    trainable = params * 0.01 # 假设 LoRA 可训练参数约 1%
    total = weights + trainable * 16
    note = "LoRA：底座冻结（BF16），只训约 1% 参数"

per_gpu = total / n_gpu

print("参数规模: %.1fB  模式: %s  卡数: %d  单卡: %dGB" % (params_b, mode, n_gpu, vram_gb))
print("估算总占用: %.1f GB  ->  均摊每卡: %.1f GB" % (total / GB, per_gpu / GB))
print("说明:", note)
print("激活值未计入，实际需额外预留 10%-40%（可用激活重算压缩）。")

plan = {"model_b": params_b, "mode": mode, "n_gpu": n_gpu, "per_gpu_est_gb": round(per_gpu / GB, 1), "note": note}
if per_gpu < vram_gb * 0.85:
    plan["verdict"] = "feasible"
    if mode == "full" and n_gpu > 1:
        plan["suggestion"] = "用 ZeRO-3/FSDP 分片；激活多时开 activation checkpointing"
    elif mode == "lora":
        plan["suggestion"] = "单卡 LoRA/QLoRA 可跑，QLoRA 显存更省"
elif mode == "full":
    plan["verdict"] = "infeasible_now"
    plan["suggestion"] = "增加卡数 / 开 ZeRO+offload / 改 LoRA(QLoRA) / 换小模型"
else:
    plan["verdict"] = "tight"
    plan["suggestion"] = "开 QLoRA 4bit 或降序列长度"

out_dir = pathlib.Path(__file__).resolve().parent.parent / "configs"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "train_plan.json", "w", encoding="utf-8") as f:
    json.dump(plan, f, ensure_ascii=False, indent=2)
print("plan -> configs/train_plan.json")
~~~

### Step 3 试算三种场景（3 分钟）

~~~bash
python scripts/plan_pretrain.py 7 lora 1 24     # 单卡 24GB LoRA CPT
python scripts/plan_pretrain.py 7 full 8 80     # 8 卡全参（ZeRO-3）
python scripts/plan_pretrain.py 70 full 64 80   # 70B 多节点（TP+PP+ZeRO）
~~~

对照输出理解：7B 全参均摊每卡约 14GB（未含激活），8×80GB 集群可行；70B 全参均摊 17.5GB/卡但**激活与通信开销大**，实际需要 TP/PP+ZeRO 组合与更多预留。

### Step 4 定方案并跑 200 步烟雾测试（10 分钟+）

把决策写进 configs/train_plan.json 对应的训练配置（batch/LR/精度/并行），然后在目标集群上：

1. 只取 1% 数据、200–500 步；
2. 盯四个数：train loss 是否下降、grad norm 是否健康、tokens/s 是否达标、MFU 是否 >15%；
3. 任何一项异常都先回 Step 3 调整，不要直接开全量。

### Step 5 归档并打版本（5 分钟）

~~~bash
cd llm-demo
git add configs scripts
git commit -m "plan: 7B LoRA CPT single-24GB, lr sweep candidates, smoke-test gate"
git tag train-plan-v0.1
~~~

> 烟雾测试通过 ≠ 可以全量：全量前还要做 checkpoint 策略、日志监控与失败预案——那是第 13 章的内容。

## 06 参数详解：预训练/CPT 关键超参速查

| 参数 | 参考区间 | 适配场景 |
|---|---|---|
| 峰值 LR | 按规模表（5e-5~1e-3） | 模型越大越低；LoRA CPT 可用 5e-5~2e-4 |
| warmup 步数 | 总步数 0.5%–2% | 大数据长训取低值，小数据取高值 |
| 全局 batch | 0.25M–60M tokens | 小模型/CPT 取低，大模型取高 |
| micro-batch | 1–16（按显存） | 先定 micro-batch，再用梯度累积凑全局 batch |
| beta2 | 0.95–0.999 | 预训练常用 0.95；微调可用 0.999 |
| weight decay | 0.01–0.1 | 预训练常 0.1（AdamW） |
| grad clip | 0.5–1.0 | 防梯度爆炸；loss spike 时先查它 |
| 精度 | BF16 主流 | FP16 需 loss scaling；FP8 需成熟 pipeline |
| 序列长度 | 2048–8192+ | 与位置编码上限匹配，长文档任务再加 |
| activation checkpointing | 开/关 | 显存紧就开，代价是约 20%–30% 重算开销 |

> 给初学者的“第一个参数组”：7B LoRA CPT = LR 1e-4 + batch 1M tokens + BF16 + grad clip 1.0 + cosine + warmup 1%，跑 500 步看曲线再调。**永远先跑小步数再放大。**

---

## 07 高频踩坑排查

**坑 1：LR 照搬大厂论文**
症状：把 175B 模型的 6e-5 直接用到 1B 模型，loss 半天不动。
解法：按规模表取初值，小规模扫 3–5 个 LR，看 500 步 loss 曲线再定。

**坑 2：显存只算权重，不算优化器与激活**
症状：7B 全参 BF16 以为 14GB 就能跑，结果 OOM。
解法：用本章公式：全参 AdamW ≈ 16 字节/参数 + 激活预留；或用 plan_pretrain.py 先估算。

**坑 3：TP 用在跨节点普通网络上**
症状：张量并行把每层拆到两台机器，通信慢到吞吐暴跌。
解法：TP 只用于 NVLink 级高速互联；跨节点用 PP/DP/ZeRO。

**坑 4：FP16 不开 loss scaling**
症状：训练几千步后 loss 突然 NaN/Inf。
解法：BF16 优先；若用 FP16 必须配动态 loss scaling 与 grad clip。

**坑 5：batch 太小导致 loss 震荡**
症状：loss 曲线锯齿严重，怎么调 LR 都没用。
解法：先凑到参考 token 量级（见 3.2 表），再调 LR；小 batch 请降低 LR。

**坑 6：只看 loss 不看 grad norm**
症状：训练“看起来在降”，其实梯度早已爆炸又恢复。
解法：grad norm 进监控；超过历史 10 倍即告警（第 13 章给排查流程）。

**坑 7：烟雾测试没跑就全量开跑**
症状：配置错误到第 10000 步才发现，浪费整周算力。
解法：任何新配置先 200–500 步烟雾测试，四个指标（loss/grad norm/吞吐/MFU）达标才全量。

---

## 08 进阶优化：超参与硬件的进化玩法

**① 小规模超参迁移**：在 1B 级模型上做 LR/batch 扫参，用“按规模缩放规则”迁移到 7B/70B（LR 随规模降、batch 随规模升）；虽然不完美，但比大集群盲调便宜两个数量级。

**② 激活重算 + 序列打包**：显存不够先开 activation checkpointing；数据侧用序列打包（第 03 章）把短文档拼成长序列，减少 padding 浪费，等效提升吞吐。

**③ 上下文并行（CP）**：序列长度到 128K+ 时，单卡装不下整条序列的激活，用 CP 沿序列切分；它需要高速互联，通常与 TP 配合。

**④ 自动并行/调度器**：Megatron 类框架支持自动切分策略搜索；云上训练平台帮你做节点调度与故障迁移。小团队别自研调度器，用成熟框架+平台。

**⑤ MFU 专项优化**：数据加载用独立进程预取、开启 kernel 融合与通信计算重叠（overlap）；每提一档 MFU，等于免费多买一批卡。

---

## 09 本章核心总结（TOP3）

**TOP1**：预训练超参是“batch–LR–精度–优化器”联动系统：先定全局 batch（tokens），再扫 LR，BF16 起步，grad clip 1.0，最后用小步数烟雾测试验证。

**TOP2**：硬件先算账：全参 BF16+AdamW ≈ 16 字节/参数（7B≈112GB+激活），必须 ZeRO/FSDP/offload；LoRA/QLoRA 把训练状态降到 1% 以内，单卡可跑。TP 要 NVLink，跨节点用 PP/DP。

**TOP3**：任何配置先跑 200–500 步烟雾测试，盯 loss/grad norm/吞吐/MFU 四指标；小规模扫参迁移到大模型，比大集群直接盲调省钱一个量级。

---

## 10 连载衔接

上一章（第 11 章）完成了领域 CPT 的首次实验；本章把超参与硬件的地图补齐——你会算显存、会选并行、会做烟雾测试，训练方案从“试出来的”变成“算出来的”。

下一章处理训练中最让人头大的环节：【第 13 章】断点续训、收敛判断与训练失败排查。checkpoint 怎么设计、loss spike 怎么定位、训练挂了怎么优雅恢复——把“训练翻车”从事故变成流程。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #预训练超参 #GPU #分布式训练 #MFU #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 12 预训练超参选型与硬件适配（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 13 章 断点续训、收敛判断与训练失败排查*