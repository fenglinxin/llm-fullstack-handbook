# 【LLM全栈工程·第14章】全维度微调技术拆解：全参/LoRA/QLoRA/适配器怎么选

> 本篇为「LLM全栈工程」连载第 14 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 微调不是越全越好：全参、LoRA、QLoRA 一次讲透；② 一张 24GB 显卡能微调 7B 吗？QLoRA 原理与选型；③ 微调技术选型决策树：数据量、显存、任务类型三个维度。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「微调与对齐体系」开篇：建立全维度微调技术地图与选型决策方法 |
| 适用场景 | 个人学习（单卡选型）；小团队领域微调立项；企业微调方案评审 |
| 核心知识点 | 全参/LoRA/QLoRA/软提示/Adapter 原理；显存与效果权衡；合并部署 |
| 技术选型 | 按“数据量×显存×任务类型”给决策树；框架选 LlamaFactory/TRL/PEFT |
| 分步实操 | 用同一份数据跑 full/lora/qlora 三组对比实验，填“微调体检表”定胜负 |
| 参数详解 | rank/alpha/dropout、量化位数、target modules、三类方法的学习率 |
| 踩坑排查 | 无脑全参、rank 过小、QLoRA 不回归、adapter 忘合并、只测训练集 |
| 进阶优化 | DoRA/rsLoRA、多 adapter 路由、层间分配、全参+LoRA 混合 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 15 章：SFT 监督微调实战 |

---

## 01 开篇导语

第 11 章我们用 CPT 把领域知识写进了底座。但“懂知识”的模型还不一定能好好回答问题——它可能不会按你的格式输出、不会拒绝编造、不会遵守 system prompt。这一步靠**微调**完成。

打开任何微调教程，你会看到一堆名词：全参微调、LoRA、QLoRA、P-Tuning、Adapter……它们到底有什么区别？是不是“全参效果最好、其他都是将就”？

答案是：**不是**。微调方法的本质是在“改多少参数”和“花多少资源”之间做选择，而“最优解”取决于你的数据量、显存和任务类型——很多场景下，LoRA 的效果与全参差距很小，成本却差一个数量级。

本章给你一张完整的技术地图：

1. 每种方法在改什么、凭什么有效；
2. 一张选型决策树（数据量×显存×任务）；
3. 一个“三方法同数据对比”的实操流程。

> 一句话记住本章：**微调选型 = 先看任务要什么（知识/格式/风格），再看资源有什么（显存/数据），最后选能完成任务的最省方案。**

---

## 02 白话原理：四种主流微调在“改什么”

### 2.1 全参微调（Full Fine-tuning）

把底座模型的所有参数都放开更新。模型容量用得最充分，理论上限最高；但代价是要为每个参数保存梯度与优化器状态（第 12 章公式：≈16 字节/参数），7B 全参 BF16 微调通常要 4×80GB 级别集群。

**适合**：数据量大（十万级以上）、任务与底座差距大、有充足算力。

### 2.2 LoRA：只学“补丁”

LoRA（Low-Rank Adaptation）冻结底座全部参数，只在某些权重旁边插入两个小矩阵 A、B，让更新量变成低秩形式：W_new = W + B×A。训练时只更新 A、B（通常占全部参数 0.1%–1%）。

- 显存需求骤降：不需要为底座存梯度与优化器状态；
- 效果接近全参（数据量不大时差距很小）；
- 训练完可以把 B×A 合并回 W——**推理时零额外开销**。

### 2.3 QLoRA：给 LoRA 再加“压缩”

QLoRA 先把底座量化成 4-bit（NF4 格式）加载，再在其上做 LoRA 训练。显存进一步减半以上，7B 模型单卡 24GB 就能微调；代价是量化带来的轻微精度损失，以及训练速度略慢（需要反量化计算）。

**适合**：消费级显卡、显存紧张、快速验证。

### 2.4 软提示与 Adapter（轻量适配）

- P-Tuning/Prefix Tuning：不更新模型，只学习一段“软提示向量/前缀”，参数量最小，适合极轻量的风格适配，不适合注入知识；
- Adapter：在 Transformer 层间插入小网络，按层适配；类似 LoRA 的另一种“补丁”思路，生态与工具链不如 LoRA 普及。

> 记忆锚点：**全参=整本书重写；LoRA=贴一张小补丁；QLoRA=把书压缩后再贴补丁；软提示=只换个开场白。** 知识注入靠补丁，风格切换有时开场白就够。

## 03 横向对比与选型决策树

### 3.1 六维对比表

| 维度 | 全参 | LoRA | QLoRA | 软提示/Adapter |
|---|---|---|---|---|
| 可训练参数 | 100% | 0.1%–1% | 0.1%–1% | <0.1% |
| 7B 显存量级（参考） | 100GB+（多卡） | 单卡 24–48GB 可行 | 单卡 12–24GB 可行 | 最小 |
| 效果上限 | 最高 | 接近全参 | 接近 LoRA（略低） | 有限 |
| 知识注入能力 | 强 | 中强 | 中强 | 弱 |
| 格式/风格适配 | 强 | 强 | 强 | 中 |
| 训练速度 | 慢 | 快 | 中（4bit 反量化开销） | 最快 |
| 遗忘风险 | 较高 | 较低 | 较低 | 极低 |
| 部署开销 | 无额外 | 可合并=无额外 | 需合并/带量化 | 需带软提示 |

### 3.2 选型决策树（三步走）

**第一步：看显存**
- 单卡 ≤24GB → 直接看 QLoRA（7B 级）或 LoRA（3B 级）；
- 单机 4–8×80GB → LoRA/全参都可，先 LoRA 验证再决定；
- 多节点集群 → 全参与 LoRA 都行，看数据量。

**第二步：看数据量**
- <1 万条：LoRA/QLoRA 优先，全参极易过拟合；
- 1 万–10 万条：LoRA 主力，效果不够再试全参；
- 10 万条以上且任务复杂：全参的上限开始体现。

**第三步：看任务类型**
- 注入领域知识 → CPT（第 11 章）打底，微调只做格式/行为；
- 格式/指令跟随 → SFT + LoRA 足够；
- 风格微调（写诗/客服话术）→ 小数据 LoRA 甚至软提示即可；
- 全能力迁移（通用助手→专用助手且算力充足）→ 全参 SFT。

> 结论模板：**“显存紧→QLoRA，数据少→LoRA，任务复杂且算力足→全参；先用最省方案跑通，效果不够再升级。”**

---

## 04 工程选型：框架与部署要点

| 方案 | 特点 | 适合 |
|---|---|---|
| Hugging Face PEFT + TRL | 底层可控、生态标准 | 想理解原理/自定义流程 |
| LlamaFactory | 配置驱动、多方法一键切 | 快速实验与日常微调 |
| 云微调服务 | 托管、免运维 | 不想管 GPU |

部署要点（容易被坑）：

1. **LoRA 训练完要合并再部署**：合并成完整权重（LlamaFactory export），避免推理时动态加载 adapter 的版本兼容问题；
2. **QLoRA 的 4bit 底座只用于训练**：部署时建议合并回 BF16/量化到 INT4 的推理格式（第 31 章），不要直接把训练用 4bit 权重当部署产物；
3. **adapter 要与底座版本绑定**：换底座版本必须重训/重验 adapter。

## 05 分步实操：同一份数据，三种方法对比（约 1–2 小时）

沿用 llm-demo：数据用第 10 章的 domain/domain_out/domain_qa.jsonl（20 条 Alpaca 问答），底座用第 02 章下载的 Qwen2.5-7B-Instruct。**先跑通对比流程，再换真实数据。**

### Step 1 选型辅助脚本（5 分钟）

保存 scripts/ft_plan.py：

~~~python
# scripts/ft_plan.py —— 微调方法选型助手
# 用法：python ft_plan.py <模型B> <显存GB> <数据量K条> <任务:knowledge|format|full>
import sys

model_b = float(sys.argv[1]) if len(sys.argv) > 1 else 7.0
vram = int(sys.argv[2]) if len(sys.argv) > 2 else 24
data_k = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
task = sys.argv[4] if len(sys.argv) > 4 else "format"

if vram <= 24:
    method = "qlora" if model_b >= 7 else "lora"
    reason = "单卡显存有限：7B 级用 QLoRA，3B 级用 LoRA"
elif data_k < 10:
    method = "lora"
    reason = "数据量小于 1 万条：LoRA 防过拟合且成本低"
elif task == "full":
    method = "full"
    reason = "任务复杂且数据/算力充足：全参上限更高"
else:
    method = "lora"
    reason = "默认路线：LoRA 起步，效果不足再升级全参"

print("推荐方法:", method)
print("理由:", reason)
print("提示: 先用推荐方法跑通，再用同数据对比 LoRA vs 全参（数据>=5万条时）验证差距。")
~~~

试几个场景：

~~~bash
python scripts/ft_plan.py 7 24 1 format     # 单卡 24GB 小数据 -> qlora
python scripts/ft_plan.py 7 320 50 full     # 8x80GB 5万条复杂任务 -> full
python scripts/ft_plan.py 3 24 0.5 style    # 小模型风格微调 -> lora
~~~

### Step 2 准备三份训练配置（10 分钟）

同一份 domain_qa.jsonl，分别注册为三个数据集名（或复用 demo_helpdesk 的注册方式），然后写三份配置。

**全参版 configs/ft_full.yaml（需 4×80GB 级）**：

~~~yaml
model_name_or_path: models/Qwen2.5-7B-Instruct
template: qwen
stage: sft
finetuning_type: full
dataset_dir: data
dataset: domain_qa
cutoff_len: 1024
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
logging_steps: 5
save_steps: 20
output_dir: outputs/ft-full
~~~

**LoRA 版 configs/ft_lora.yaml**：

~~~yaml
model_name_or_path: models/Qwen2.5-7B-Instruct
template: qwen
stage: sft
finetuning_type: lora
dataset_dir: data
dataset: domain_qa
cutoff_len: 1024
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
logging_steps: 5
save_steps: 20
output_dir: outputs/ft-lora
~~~

**QLoRA 版 configs/ft_qlora.yaml**（在 LoRA 版基础上加两行）：

~~~yaml
# 在 LoRA 配置基础上追加：
quantization_bit: 4
quantization_method: bitsandbytes
~~~

> 三份配置的关键差异：全参 LR 用 1e-5（改动全参数必须小步走），LoRA/QLoRA 用 2e-4（只改补丁可以大步走）；QLoRA 因 4bit 反量化计算，batch 建议保持 1–2。

### Step 3 三方法同数据对比（30–60 分钟）

按顺序跑（全参需要多卡/大显存，没有就只对比 LoRA vs QLoRA）：

~~~bash
# 1. 先记录基线：未微调底座对 5 个领域问题的回答
# （用第 02 章 vLLM + 脚本记录为 baseline_answers.txt）

# 2. LoRA
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/ft_lora.yaml

# 3. QLoRA（单卡 24GB 首选）
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/ft_qlora.yaml

# 4. 全参（如有多卡）
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train configs/ft_full.yaml
~~~

每个训练过程中另开终端记录峰值显存：

~~~bash
nvidia-smi --query-gpu=memory.used --format=csv -l 5 | sort -t, -k1 -n | tail -n 3
~~~

### Step 4 合并、部署、填“微调体检表”（10 分钟）

每份 adapter 用第 02 章 export 配置合并到 models/ft-{method}-merged，然后用同一组领域问题+通用问题问三个模型 + 基线，填表：

| 项目 | 基线 | 全参 | LoRA | QLoRA |
|---|---|---|---|---|
| 领域问题得分（5 题对几题） | ？ | ？ | ？ | ？ |
| 通用问题得分 | ？ | ？ | ？ | ？ |
| 峰值显存 | ？ | ？ | ？ | ？ |
| 训练耗时 | - | ？ | ？ | ？ |
| 训练 loss（末值） | - | ？ | ？ | ？ |
| 合并后模型大小 | - | ？ | ？ | ？ |

**决策规则**：

1. 领域分与基线比必须显著提升（20 条演示数据可能不明显，属正常，流程看方法不看分数）；
2. 通用分不能明显下降；
3. 在“效果达标”的方法里选显存/耗时最低的——**够用就好，省下的算力留给数据迭代**。

### Step 5 归档与打版本

~~~bash
cd llm-demo
git add configs outputs scripts
git commit -m "ft: compare full/lora/qlora on domain_qa (see ft_compare.md)"
git tag ft-compare-v0.1
~~~

> 建议把“体检表”沉淀成团队模板 docs/ft_compare.md：以后每个微调项目都按同一张表对比，选型不再靠吵架。

## 06 参数详解：微调方法关键参数速查

| 参数 | 全参 | LoRA/QLoRA | 说明 |
|---|---|---|---|
| 学习率 | 1e-5~2e-5 | 1e-4~3e-4 | 全参小步走，补丁可大步 |
| lora_rank | - | 8–64（常见 16–32） | 数据多/任务难可加大 |
| lora_alpha | - | 常为 rank 的 1–2 倍 | 控制补丁缩放 |
| lora_dropout | - | 0–0.1 | 防过拟合 |
| target modules | - | 默认 q/k/v/o 起步 | 效果不够再扩到 gate/up/down |
| 量化位数 | - | 4（QLoRA） | NF4 常用；8bit 精度更高 |
| epoch | 2–5 | 2–5 | 小数据多看 val 防背题 |
| batch | 按显存 | 1–8 | 等效 batch 16–64 常见 |

> 调参顺序建议：先固定“方法”（按决策树），再调 rank 与 LR（第 18 章展开）；不要同时动方法、rank、LR、数据四个变量。

---

## 07 高频踩坑排查

**坑 1：无脑全参微调**
症状：5000 条数据也上全参，显存爆、过拟合、通用能力掉。
解法：按决策树先 LoRA/QLoRA；数据到 5–10 万级且效果有瓶颈再试全参。

**坑 2：LoRA rank 设太小**
症状：训练 loss 降不下去或领域能力不足。
解法：rank 从 16 起，试 32/64 并配 alpha=2×rank；还不行检查数据而不是继续加 rank。

**坑 3：QLoRA 训完不回归**
症状：部署后发现质量比 LoRA 差一截，但没人知道差在哪。
解法：QLoRA 与 LoRA 跑同数据对比表；领域要求高时用 LoRA，显存实在不够再 QLoRA。

**坑 4：adapter 没合并就部署**
症状：推理服务加载 adapter 报版本错/行为不对。
解法：训练后统一 export 合并成完整模型再部署；adapter 与底座版本绑定记录。

**坑 5：target modules 选错**
症状：只调了 embedding/不调 attention，效果全无。
解法：默认覆盖 attention 的 q/k/v/o；需要更强适配再加 MLP 的 gate/up/down。

**坑 6：用训练集当评测**
症状：loss 很低、训练题全会，换新题就露馅。
解法：训练/评测集分离（第 16/20 章），评测题要与训练数据不同分布并去重。

**坑 7：三种方法用不同数据/不同 epoch 对比**
症状：对比结论全是变量混杂，等于没对比。
解法：同数据、同 epoch、同评测，只允许方法（及配套 LR）不同。

---

## 08 进阶优化：微调技术的进化方向

**① DoRA / rsLoRA**：DoRA 把权重分解为幅度与方向分别适配，常比 LoRA 更稳；rsLoRA 调整 rank 缩放，支持更大 rank——效果不够时的低成本升级路径。

**② 多 Adapter 路由**：每个领域/风格训一个 LoRA，推理时按请求路由到对应 adapter（或叠加多个），实现“一底座多技能”且互不污染。

**③ 层间 rank 分配**：不是每层都需要同样容量——注意力/底层与任务相关时给更高 rank；用“分层 rank 搜索”省显存提效果。

**④ 全参+LoRA 混合**：先 LoRA 快速验证数据，再把 adapter 作为初始化做短程全参精修（full refine），兼顾成本与上限。

**⑤ 量化感知训练**：直接训练时就考虑 4bit/8bit 推理约束（QAT 思想），让微调与最终部署精度对齐，减少“训完再量化掉点”。

---

## 09 本章核心总结（TOP3）

**TOP1**：微调方法 = 改多少参数的选择：全参=整本书重写（强但贵），LoRA=贴补丁（性价比之王），QLoRA=压缩后贴补丁（单卡可行），软提示=换开场白（最轻量）。

**TOP2**：选型决策三步走：显存定范围（≤24GB 用 QLoRA/LoRA）→ 数据量定路线（<1 万 LoRA，10 万+ 才考虑全参）→ 任务类型定目标（知识靠 CPT，格式行为靠 SFT，风格可软提示）。

**TOP3**：任何选型结论都要用“同数据三方法体检表”验证：领域分、通用分、显存、耗时、模型大小五列数据说话；训练后 adapter 合并部署，QLoRA 必须做质量回归。

---

## 10 连载衔接

上一章（第 13 章）给训练系统装上了保险与黑匣子；本章打开了模型能力层的大门——你现在知道全参、LoRA、QLoRA 各自的位置，也知道怎么用对比实验做选择。

下一章真正动手：【第 15 章】SFT 监督微调实战：数据、训练、评估到合并上线。把第 10 章的领域问答数据，变成能部署上线的领域助手。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #微调 #LoRA #QLoRA #全参微调 #PEFT #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 14 全维度微调技术拆解（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 15 章 SFT 监督微调实战*