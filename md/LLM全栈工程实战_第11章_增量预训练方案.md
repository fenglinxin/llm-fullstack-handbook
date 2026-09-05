# 【LLM全栈工程·第11章】增量预训练方案：领域适配、学习率与灾难遗忘

> 本篇为「LLM全栈工程」连载第 11 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 别急着微调：领域知识先进底座，增量预训练实战；② 学了领域忘了通用怎么办？CPT 数据配比与遗忘控制；③ 从开源底座到行业底座：增量预训练的正确打开方式。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「预训练工程·数据核心层」收官篇：把第 10 章领域语料“练进”底座模型（CPT），并守住通用能力 |
| 适用场景 | 个人学习（小模型/LoRA CPT 实验）；小团队做领域底座；企业从通用底座走向行业底座 |
| 核心知识点 | CPT 与预训练/SFT 的边界；数据配比与回放；学习率策略；灾难遗忘的度量与缓解 |
| 技术选型 | 全参 CPT vs LoRA CPT；LlamaFactory/自写脚本；回放数据来源 |
| 分步实操 | 组 CPT 数据（领域+通用回放）→ LlamaFactory stage=pt 训练 → 领域/通用双向评测 → 调配比 |
| 参数详解 | 领域占比、回放比例、学习率、epoch、退火、评测闸门 |
| 踩坑排查 | 拿 Instruct 模型硬练、领域占比过高、不回放通用、不测遗忘、小数据多 epoch 背题 |
| 进阶优化 | 数据退火、EWC/L2 防遗忘、adapter 策略、多轮增量更新 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 12 章：预训练超参选型与硬件适配 |

---

## 01 开篇导语

上一章我们建好了领域弹药库：规则语料、合成问答、质检报告，整整齐齐躺在 domain_out 里。

现在问题来了：**怎么把这些领域知识真正“装进”模型？**

很多人的第一反应是“直接 SFT”。但 SFT 教的是“按指令回答”，数据量小、学习率偏高，很难把大量新知识稳定地写进参数——你问它公司打印机型号，它可能还是只会编。

正确的中间步骤叫**增量预训练（Continued Pre-Training，CPT）**：用预训练同样的“下一个 token 预测”目标，在开源底座上继续训练领域语料（通常混入通用语料做“回放”），让模型先把领域知识“学会”，再做 SFT 教格式与行为。

本章回答四个问题：

1. CPT 和预训练、SFT 到底什么关系；
2. 领域数据和通用数据按什么比例混；
3. 学习率等超参怎么定；
4. 怎么防止“学了领域、忘了通用”。

> 一句话记住本章：**CPT 是给底座“补知识”的慢功夫，SFT 是给模型“教规矩”的快功夫；先补知识，再教规矩。**

---

## 02 白话原理：CPT 到底在做什么

### 2.1 三种训练的分工

| 训练 | 目标 | 数据 | 学习率 | 效果 |
|---|---|---|---|---|
| 预训练 | 从零学语言与世界 | 数万亿 token 通用语料 | 较高（按规模递减） | 底座能力 |
| CPT（本章） | 在底座上补领域知识 | 领域为主 + 通用回放 | 低（1e-5~1e-4 量级） | 领域知识进参数 |
| SFT | 学指令与输出格式 | 万级指令问答 | 中 | 会按指令回答 |

### 2.2 CPT 的“输入输出”与预训练完全一样

CPT 用的还是“看前文、猜下一个 token”的自监督目标——不需要人工标注。领域文档本身就是训练数据。模型在反复“接龙”领域文本时，把其中的术语、事实、行文规律压缩进参数。

### 2.3 为什么会有“灾难遗忘”

继续训练本质是“用新分布覆盖旧分布”。如果只喂领域数据，模型参数会逐渐偏向领域，把通用语料里学到的知识“冲掉”——这就是**灾难遗忘（Catastrophic Forgetting）**。

防遗忘三板斧：

1. **低学习率**：每次改动小一点，别把旧知识冲太狠；
2. **通用回放**：混合 70%–95% 的通用语料，让模型不忘“母语”；
3. **度量与闸门**：训练前后都跑通用评测，掉分超阈值就回滚/降比例。

> 关键认知：CPT 的目标不是“领域 loss 最低”，而是“**领域能力提升的同时，通用能力不掉出容忍线**”。

## 03 数据配比与遗忘控制：CPT 的命门

### 3.1 领域与通用怎么混

| 方案 | 领域:通用 | 特点 | 适用 |
|---|---|---|---|
| 领域为主 | 80:20 及以上 | 领域涨得快，遗忘风险高 | 几乎不用通用能力的专用模型 |
| 平衡混比 | 30:70 | 领域与通用兼顾 | 多数场景起点 |
| 通用为主 | 5:95 | 几乎不遗忘，但领域渗透慢 | 数据极少/领域极窄时 |

工程上推荐从 **10%–30% 领域**起步做消融：固定算力跑 3 组配比（10/20/30），用“领域评测分 + 通用评测分”画曲线选点。

### 3.2 回放数据怎么选

- 通用回放不必重新爬全网——用**高质量通用子集**即可（比如通用语料里 quality_score 前 30% 的样本）；
- 回放数据要与领域数据**同批次混合**，而不是先领域后通用分段练；
- 回放里可以按“易遗忘主题”加权（代码/数学/长文），具体通过消融确认。

### 3.3 数据退火（Data Annealing）

训练后期（最后 5%–10% 的 step）把领域高质量数据占比提到 60%–90%、学习率同步衰减——相当于考前冲刺，让模型把刚学的领域知识“钉牢”。退火阶段数据要选领域里 quality_score 最高的子集。

### 3.4 遗忘度量：双向评测闸门

每次 CPT 实验必须带两组评测：

| 评测 | 内容 | 目标 |
|---|---|---|
| 领域评测 | 第 10 章建的领域评测题（与训练源去重） | 领域分涨 |
| 通用评测 | MMLU 类公共基准或自建通用集 | 通用分不掉出容忍线（如 -2% 内） |

**没有双向评测的 CPT，等于蒙眼开车**——只看领域 loss 下降，可能通用能力已经崩了。

---

## 04 技术选型：全参还是 LoRA，用什么框架

| 方案 | 优点 | 缺点 | 适合 |
|---|---|---|---|
| 全参 CPT | 知识写入最充分 | 显存/算力高，遗忘风险更大 | 有集群、数据量大 |
| LoRA/QLoRA CPT | 单卡可跑、可插拔、易回滚 | 知识容量上限低于全参 | 中小团队首选起点 |
| Adapter 合并策略 | 领域 adapter 与通用底座分离 | 推理要带 adapter | 多领域并存场景 |

框架选择：

| 框架 | 用法 | 适合 |
|---|---|---|
| LlamaFactory（本章） | stage=pt + 数据集配置 | 快速跑通 LoRA CPT |
| TRL / transformers 自写 | 更底层控制 | 需要自定义回放/退火逻辑 |
| Megatron/DeepSpeed | 大规模全参 CPT | 企业大模型 |

结论：**单卡/小集群先用 LoRA CPT 验证数据与配比**，验证有效后再决定是否升级全参 CPT——数据配比的经验可以平移，训练方式只是执行手段。

## 05 分步实操：LoRA 增量预训练 + 双向评测（约 1 小时）

沿用 llm-demo（第 02 章环境：LlamaFactory + 本地模型 + vLLM）。本章用 stage=pt 跑 LoRA CPT。

> 生产注意：CPT 应该用**底座模型（非 Instruct）**。本章为了复用第 02 章下载的 Instruct 模型做演示，仅用于理解流程；正式项目请下载对应 Base 版。

### Step 1 组装 CPT 数据：领域 + 通用回放（10 分钟）

保存 scripts/make_cpt_data.py：

~~~python
# scripts/make_cpt_data.py —— 组装 CPT 数据：领域语料 + 通用回放
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
corpus_path = root / "domain" / "domain_out" / "domain_corpus.jsonl"
out_path = root / "data" / "domain_cpt.jsonl"

general_replay = [
    {"text": "Transformer 是一种基于自注意力机制的神经网络架构，广泛用于自然语言处理任务。", "source": "general"},
    {"text": "Python 是解释型、面向对象的编程语言，语法简洁，生态丰富，适合快速开发。", "source": "general"},
    {"text": "数据库索引用于加速查询，常见结构包括 B+ 树与哈希索引，但会占用额外存储。", "source": "general"},
    {"text": "梯度下降通过反复计算损失函数的梯度来更新模型参数，使损失逐步下降。", "source": "general"},
    {"text": "缓存是把频繁访问的数据放在更快的存储中，以空间换时间。", "source": "general"},
    {"text": "版本控制工具帮助团队记录代码变更历史，支持多人协作与回滚。", "source": "general"},
    {"text": "API 是软件之间交互的约定接口，良好的 API 设计应清晰、稳定、易用。", "source": "general"},
    {"text": "监控系统通过采集指标、日志与链路追踪，帮助运维快速定位线上问题。", "source": "general"},
    {"text": "加密技术用于保护数据传输与存储安全，常见算法分为对称加密与非对称加密。", "source": "general"},
]

rows = []
for rec in [json.loads(line) for line in open(corpus_path, encoding="utf-8")]:
    rows.append({"text": rec["text"], "source": "domain:" + rec["rule_id"]})
rows.extend(general_replay)

with open(out_path, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("cpt rows:", len(rows), "(domain", len(rows) - len(general_replay), "+ general", len(general_replay), ")")
~~~

运行：

~~~bash
python scripts/make_cpt_data.py
~~~

### Step 2 注册数据集 + 写训练配置（10 分钟）

把 data/dataset_info.json 更新为（保留原有 demo_helpdesk，新增 domain_cpt）：

~~~json
{
  "demo_helpdesk": {
    "file_name": "demo_helpdesk.jsonl",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "domain_cpt": {
    "file_name": "domain_cpt.jsonl",
    "columns": {
      "prompt": "text"
    }
  }
}
~~~

保存 configs/cpt_lora.yaml：

~~~yaml
# configs/cpt_lora.yaml
model_name_or_path: models/Qwen2.5-7B-Instruct
template: qwen
stage: pt
finetuning_type: lora
dataset_dir: data
dataset: domain_cpt
cutoff_len: 1024
max_samples: 60

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1

bf16: true
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05

logging_steps: 5
save_steps: 10
output_dir: outputs/domain-cpt-lora
plot_loss: true
~~~

> 说明：LlamaFactory 的 pt 阶段把每行 text 当作连续文本训练，labels 自动等于 input_ids（自回归）。domain_cpt 只注册了 prompt→text 映射，训练时取 text 字段。

### Step 3 跑 LoRA CPT（20–40 分钟）

~~~bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/cpt_lora.yaml
~~~

看到输出目录 outputs/domain-cpt-lora 与 loss 曲线即成功。由于演示数据只有十几条，loss 会快速下降——注意这属于“背熟演示集”，真实项目要用几千到几百万条领域文本 + 更大回放池。

### Step 4 合并导出 + 双向评测（15 分钟）

保存 configs/export_cpt.yaml：

~~~yaml
# configs/export_cpt.yaml
model_name_or_path: models/Qwen2.5-7B-Instruct
adapter_name_or_path: outputs/domain-cpt-lora
template: qwen
finetuning_type: lora
export_dir: models/domain-cpt-merged
export_size: 2
export_legacy_format: false
~~~

合并并起两个服务做“前后对比”（底座 8000 端口、CPT 后 8001 端口）：

~~~bash
llamafactory-cli export configs/export_cpt.yaml

# 终端 A：底座（微调前基线）
vllm serve models/Qwen2.5-7B-Instruct --served-model-name base --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.45

# 终端 B：CPT 后模型（需另一块卡，或依次启动）
vllm serve models/domain-cpt-merged --served-model-name cpt --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.45
~~~

> 单卡用户请依次启动：先起底座问完关掉，再起 CPT 模型问同一组问题，把答案存成 before/after 文本对比。

保存 scripts/eval_cpt.py：

~~~python
# scripts/eval_cpt.py —— 领域+通用双向评测（OpenAI 兼容接口）
import sys
from openai import OpenAI

base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1"
client = OpenAI(base_url=base_url, api_key="EMPTY")

questions = [
    ("domain", "公司打印机连不上，按公司规定应该怎么处理？"),
    ("domain", "VPN 账号被锁定了，公司规定怎么处理？"),
    ("general", "请用一句话解释什么是注意力机制。"),
    ("general", "Python 里的列表和元组有什么区别？"),
]

for kind, q in questions:
    resp = client.chat.completions.create(
        model="base" if "8000" in base_url else "cpt",
        messages=[{"role": "user", "content": q}],
        temperature=0.2,
        max_tokens=300,
    )
    print("[" + kind + "] Q:", q)
    print("A:", resp.choices[0].message.content)
    print("-" * 50)
~~~

运行并对比：

~~~bash
# 先起底座服务后：
python scripts/eval_cpt.py http://localhost:8000/v1 > eval_before.txt
# 再起 CPT 服务后：
python scripts/eval_cpt.py http://localhost:8001/v1 > eval_after.txt
diff eval_before.txt eval_after.txt   # 或人工对比
~~~

**看什么**：领域问题（打印机/VPN）的答案是否出现公司规则（HP LaserJet M405、30 分钟解锁）；通用问题是否仍然正常——如果通用回答明显变差，说明回放不够或学习率偏高。

### Step 5 配比消融 + 打版本（10 分钟）

1. 修改 make_cpt_data.py：把领域:通用从 4:9 调到 8:6、12:3，各训练一版；
2. 每版跑 Step 4 的双向评测，记录“领域分 vs 通用分”；
3. 选“通用不掉出容忍线、领域提升最大”的配比；
4. 打版本：

~~~bash
cd llm-demo
git add data configs outputs scripts
git commit -m "cpt: domain v1 lora (domain:general=4:9, lr=1e-4) + dual eval"
git tag cpt-exp-v0.1
~~~

> 工业提示：正式 CPT 的算力预算里，**评测与消融要占 20%–30%**——训练只花一部分钱，剩下的钱花在搞清楚“该不该这么训”上。

## 06 参数详解：CPT 超参速查

| 参数 | 参考取值 | 说明 |
|---|---|---|
| 领域:通用 | 10:90 ~ 30:70 起步 | 专用模型可 50:50+；用消融定 |
| 学习率（LoRA CPT） | 5e-5 ~ 2e-4 | 比 SFT 略低，比全参 CPT 高 |
| 学习率（全参 CPT） | 1e-5 ~ 5e-5 | 全参改动面大，必须更低 |
| epoch | 大语料 <1；小语料 2–5 | 盯领域与通用 loss/评测 |
| 退火比例 | 最后 5%–10% step | 高质量领域数据 + LR 衰减 |
| 回放来源 | 通用语料高分桶 | 与领域同批次混 |
| LoRA rank | 16–64 | 领域数据多可加大；单卡 16 起步 |
| cutoff_len | 2048–8192 | 领域长文档多就加长 |
| 通用掉分容忍线 | -1%~-3%（按评测集噪声） | 超过即回滚/降领域占比 |

> 真实项目的 CPT 调参顺序建议：先定“评测闸门”（领域题+通用题），再定配比，最后微调学习率——顺序反了会陷入 loss 数字的局部最优。

---

## 07 高频踩坑排查

**坑 1：拿 Instruct 模型做 CPT 当生产方案**
症状：在 Chat 版模型上继续预训练，对话格式被“预训练目标”干扰，越练越不会聊天。
解法：生产 CPT 用 Base 版底座；Instruct 版只适合做演示。CPT 后再走 SFT/对齐恢复指令能力。

**坑 2：只喂领域数据，不回放通用**
症状：领域评测涨了，通用能力暴跌（写代码、数学、常识全废）。
解法：通用回放 70%+；用双向评测盯住通用掉分。

**坑 3：学习率照搬 SFT**
症状：SFT 的 2e-4 直接用于全参 CPT，几个 step 后 loss 飙升。
解法：全参 CPT 用 1e-5~5e-5；LoRA CPT 用 5e-5~2e-4；先跑 200 step 看稳定性再放大。

**坑 4：小数据多 epoch 背题**
症状：演示数据跑 20 epoch，loss 归零，以为模型“学会了”。
解法：小数据只做流程验证；真实效果看“没见过的领域题”，而不是训练 loss。

**坑 5：不建领域评测集，只测 loss**
症状：loss 下降但问领域问题还是不会——因为领域评测题与训练数据重叠少或 loss 被通用 token 稀释。
解法：用第 10 章隔离出的领域评测题做“会不会”的实测；loss 与评测两条曲线一起看。

**坑 6：评测集混进训练数据**
症状：领域评测分虚高，上线被打回原形。
解法：评测集先于训练集构建并锁版；任何新数据入库前用 MinHash 对评测集去重（第 07 章）。

**坑 7：只训一版就上线**
症状：没做配比消融、没做通用回归，领域模型带病上线。
解法：至少对比 2–3 组配比 + 通用回归；把实验记录（数据版本/配置/评测）归档，作为上线评审证据。

---

## 08 进阶优化：CPT 的进化方向

**① 数据退火工程化**：把“退火阶段数据子集”做成独立数据版本（domain_anneal_v1），退火起始 step 由 loss 曲线与评测共同触发，而不是拍脑袋定 90%。

**② 防遗忘正则**：全参 CPT 可加 L2 正则/EWC 类约束，限制参数偏离底座太远；LoRA 本身改动面小，是天然的“轻遗忘”方案。

**③ Adapter 多领域策略**：每个领域训一个 LoRA adapter，推理时按请求路由/叠加 adapter——领域之间互不污染，升级单个领域只需重训该 adapter。

**④ 增量更新节奏**：领域知识每月更新时，用“新知识 + 旧知识抽样 + 通用回放”做滚动 CPT；每次更新都要重跑双向评测与回归集。

**⑤ 先小后大的放大流程**：先在 1B 级模型上把配比、退火、评测闸门跑通（成本低、迭代快），再把同一套数据策略放大到 7B/70B——数据决策在小模型上做，算力花在大模型上。

---

## 09 本章核心总结（TOP3）

**TOP1**：CPT 用“下一个 token 预测”把领域知识写进底座，是预训练与 SFT 之间的知识桥梁；生产用 Base 模型，CPT 后再做 SFT/对齐。

**TOP2**：灾难遗忘的解法是“低学习率 + 通用回放（70%+）+ 双向评测闸门”；领域:通用配比从 10:90~30:70 起步做消融，领域 loss 不是唯一指标。

**TOP3**：CPT 的正确验收是“领域评测涨 + 通用评测不掉出容忍线”；评测集先建、锁版、与训练集去重，实验记录归档——没有评测闸门的 CPT 等于盲训。

---

## 10 连载衔接

上一章（第 10 章）建好了领域语料；本章用 LoRA CPT 把它们练进底座，并用双向评测守住了通用能力——主线工程从“有领域数据”推进到“有领域底座”。

下一章把训练侧的“仪表盘”补齐：【第 12 章】预训练超参选型与硬件适配。学习率、批次、精度、并行策略与硬件怎么匹配，是让 CPT/预训练跑得又稳又省的关键。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #增量预训练 #CPT #灾难遗忘 #领域大模型 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 11 增量预训练方案（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 12 章 预训练超参选型与硬件适配*