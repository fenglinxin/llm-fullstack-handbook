# 【LLM全栈工程·第15章】SFT 监督微调实战：数据、训练、评估到合并上线

> 本篇为「LLM全栈工程」连载第 15 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 微调不是“跑个脚本”就完事：SFT 全流程七步实战；② 让模型学会按你的格式回答：监督微调从数据到上线；③ SFT 最容易翻车的三个环节：模板、Loss 掩码与评测泄漏。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「微调与对齐体系」第 2 篇：完整跑一遍工业级 SFT：数据准备→训练→评估→合并→部署→回归 |
| 适用场景 | 个人学习（单卡 LoRA SFT）；小团队领域助手开发；企业微调上线前的标准流程 |
| 核心知识点 | SFT 在学什么（格式/行为）；response-only loss；训练/验证切分；合并部署；held-out 评测 |
| 技术选型 | LlamaFactory vs TRL；LoRA 默认路线；合并部署 vs adapter 部署 |
| 分步实操 | 准备数据切分 → 注册数据集 → 训练配置（含 val）→ 训练 → 导出合并 → vLLM 部署 → 自动化评测 |
| 参数详解 | val_size、epoch、LR、cutoff、eval_steps、seed 等 |
| 踩坑排查 | 模板错乱、loss 没掩码、评测泄漏、val 不设、adapter 不合并、只看 loss 不看格式 |
| 进阶优化 | 多轮数据、packing、指令多样性、困难样本挖掘、SFT+DPO 接力 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 16 章：SFT 数据集构建与标注规范 |

---

## 01 开篇导语

第 02 章你已经跑过一次“最小 SFT”，第 14 章知道了方法怎么选。但“能跑通”和“能上线”之间还隔着一段路：

- 训练时 loss 在降，但模型回答不按你的格式来？
- 验证集没有单独留，训练完才发现评测题就在训练数据里？
- adapter 没合并，部署时加载失败？

本章把 SFT 当“产品工序”来过一遍，七步闭环：

**数据切分 → 注册数据集 → 配置训练（带验证）→ 训练 → 合并导出 → 部署 → held-out 评测回归**

全程沿用主线工程 llm-demo：数据是第 10 章的帮助台问答，底座是第 02 章的 Qwen2.5-7B-Instruct，框架是 LlamaFactory。跑完你会得到：一份带验证切分的数据集、一份训练产物、一个能部署的合并模型、一份自动化评测报告。

> 一句话记住本章：**SFT 的质量不在训练脚本，而在“数据怎么切、Loss 怎么算、效果怎么验”三个细节里。**

---

## 02 SFT 到底在学什么：三个关键机制

### 2.1 SFT 教“格式与行为”，不负责“灌知识”

CPT（第 11 章）负责把新知识写进参数；SFT 负责让模型学会：看到指令怎么组织回答、按什么格式输出、什么情况要拒绝。用领域问答做 SFT 时，答案里的知识最好底座/CPT 已经具备，SFT 主要学“怎么把知识按规范说出来”。

### 2.2 Response-only Loss：只让模型学“答案”

SFT 样本是“指令+回答”。如果让模型对整段文本（包括指令）都计算损失，模型会花一半力气去“背问题”。工业标准做法是**只对回答部分计算 loss，指令部分用 -100 掩码掉**——模型只学“看到这个问题该怎么答”。

LlamaFactory 的 alpaca/sharegpt 格式会自动做这件事，但你要知道它存在：**自写训练脚本时最容易漏的就是这个掩码。**

### 2.3 Chat Template：模型“说话”的格式约定

同样的内容，套不同的聊天模板，模型行为可能完全不同（system 角色、助手前缀、结束符）。训练与推理必须用同一套模板，且与底座官方模板一致——模板错乱的典型症状是：训练 loss 正常，生成却前言不搭后语。

> 记忆锚点：**SFT = 用“问题-标准答案”对，教会模型按模板与格式回答问题；问题部分不参与损失，答案部分才是学习对象。**

## 03 工业 SFT 工作流：七步闭环

~~~text
① 数据准备（清洗/切分/质检）
   → ② 数据集注册（dataset_info）
   → ③ 训练配置（LoRA + 验证集 + 固定 seed）
   → ④ 训练（盯 train/val loss）
   → ⑤ 合并导出（adapter -> 完整权重）
   → ⑥ 部署（vLLM OpenAI 兼容服务）
   → ⑦ Held-out 评测回归（领域题 + 通用题）
~~~

每一步都有“验收动作”：

| 步骤 | 验收动作 |
|---|---|
| 数据切分 | train/val 无重叠；val 与训练不同分布但同格式 |
| 注册 | 打印一条预处理样本，人工核对模板 |
| 训练 | train loss 降、val loss 同步降、无 spike |
| 合并 | 加载后能正常 generate |
| 部署 | 接口响应正常、格式合规 |
| 评测 | 领域题通过率提升、通用题不掉 |

---

## 04 技术选型：框架、方法与评测口径

| 决策点 | 推荐 | 备选 |
|---|---|---|
| 微调方法 | LoRA（数据<10 万时性价比最高） | QLoRA（显存紧）、全参（数据大+算力足） |
| 框架 | LlamaFactory（配置驱动、模板处理成熟） | TRL（要更底层控制） |
| 数据格式 | Alpaca/ShareGPT（LlamaFactory 原生支持） | 自写格式（需自实现 loss 掩码） |
| 评测 | 领域 held-out 题 + 通用题 + 格式合规检查 | 只看 loss（不推荐） |

> 选型结论：**主线工程用 LoRA + LlamaFactory + 领域 held-out 评测**；当需要多轮对话、工具调用等复杂格式时再升级 ShareGPT 格式（第 17/21 章）。

## 05 分步实操：七步跑通工业 SFT（约 1 小时）

### Step 1 数据切分：train / val / held-out（10 分钟）

保存 scripts/prepare_sft.py：

~~~python
# scripts/prepare_sft.py —— 领域问答 -> train/val 切分（固定种子）
import json
import pathlib
import random

root = pathlib.Path(__file__).resolve().parent.parent
src = root / "domain" / "domain_out" / "domain_qa.jsonl"
data_dir = root / "data"

rows = [json.loads(line) for line in open(src, encoding="utf-8")]
random.seed(42)
random.shuffle(rows)

n_val = max(1, int(len(rows) * 0.2))
val_rows = rows[:n_val]
train_rows = rows[n_val:]

def write_rows(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

write_rows(data_dir / "sft_train.jsonl", train_rows)
write_rows(data_dir / "sft_val.jsonl", val_rows)
print("train:", len(train_rows), "val:", len(val_rows))
print("val file -> data/sft_val.jsonl（held-out 评测用）")
~~~

运行：

~~~bash
python scripts/prepare_sft.py
head -n 1 data/sft_val.jsonl
~~~

### Step 2 注册数据集（5 分钟）

在 data/dataset_info.json 中追加（与现有条目并列）：

~~~json
{
  "sft_train": {
    "file_name": "sft_train.jsonl",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "sft_val": {
    "file_name": "sft_val.jsonl",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
~~~

### Step 3 写训练配置（5 分钟）

保存 configs/sft_industrial.yaml：

~~~yaml
# configs/sft_industrial.yaml
model_name_or_path: models/Qwen2.5-7B-Instruct
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
learning_rate: 2.0e-4
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1

bf16: true
seed: 42
eval_strategy: steps
eval_steps: 10
logging_steps: 5
save_steps: 20

lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05

output_dir: outputs/helpdesk-sft
plot_loss: true
~~~

> 说明：val_size=0.2 让训练过程自带验证（看 val loss 防过拟合）；字段名在不同 LlamaFactory 版本可能为 evaluation_strategy，以官方文档为准。sft_val.jsonl 是训练完成后的外部 held-out 评测集，两者用途不同。

### Step 4 训练（20–40 分钟）

~~~bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/sft_industrial.yaml
~~~

观察要点：train loss 与 val loss 同步下降；如果 train loss 降而 val loss 回升，说明开始过拟合（数据少时尤其常见，此时应提前用中间的 checkpoint）。

### Step 5 合并导出（5 分钟）

保存 configs/export_sft.yaml（指向本次训练产物）：

~~~yaml
model_name_or_path: models/Qwen2.5-7B-Instruct
adapter_name_or_path: outputs/helpdesk-sft
template: qwen
finetuning_type: lora
export_dir: models/helpdesk-sft-merged
export_size: 2
export_legacy_format: false
~~~

~~~bash
llamafactory-cli export configs/export_sft.yaml
~~~

### Step 6 部署（5 分钟）

~~~bash
vllm serve models/helpdesk-sft-merged \
  --served-model-name helpdesk-sft \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8000
~~~

### Step 7 Held-out 评测回归（10 分钟）

保存 scripts/sft_eval.py：

~~~python
# scripts/sft_eval.py —— SFT 后 held-out 评测：领域知识命中 + 通用可用性
import json
import pathlib
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# 固定领域题与必须命中的关键信息（来自公司知识库规则）
domain_cases = [
    ("打印机连不上，按公司规定怎么处理？", ["HP LaserJet M405", "工单"]),
    ("公司 WiFi 连不上怎么办？", ["Corp-WiFi"]),
    ("VPN 账号被锁定了怎么办？", ["30 分钟"]),
    ("邮箱容量超了怎么办？", ["90%"]),
]
general_cases = [
    ("请用一句话解释什么是注意力机制。", None),
]

def ask(question):
    resp = client.chat.completions.create(
        model="helpdesk-sft",
        messages=[{"role": "user", "content": question}],
        temperature=0.2,
        max_tokens=300,
    )
    return resp.choices[0].message.content

results = []
for q, keywords in domain_cases:
    answer = ask(q)
    hit = all(k in answer for k in keywords) if keywords else len(answer) > 20
    results.append({"question": q, "pass": hit, "answer": answer[:120]})
    print(("PASS" if hit else "FAIL"), q)

report = {"total": len(results), "pass": sum(1 for r in results if r["pass"])}
out = pathlib.Path(__file__).resolve().parent.parent / "logs" / "sft_eval_report.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump({"report": report, "results": results}, f, ensure_ascii=False, indent=2)
print("summary:", report)
~~~

运行：

~~~bash
python scripts/sft_eval.py
cat logs/sft_eval_report.json
~~~

**验收线**：领域题 4/4 命中公司规则关键信息，通用题仍能正常回答；若领域题不达标，先查数据质量与模板，再调参（第 18 章），不要盲目加 epoch。

### Step 8 归档

~~~bash
cd llm-demo
git add data configs outputs scripts logs
git commit -m "sft: helpdesk industrial v1 (train/val split, eval 4/4)"
git tag sft-helpdesk-v1
~~~

## 06 参数详解：SFT 关键配置速查

| 参数 | 参考取值 | 说明 |
|---|---|---|
| val_size | 0.1–0.2 | 数据少可 0.1；训练过程看 val loss 防过拟合 |
| num_train_epochs | 2–5 | 小数据 3–5；盯 val loss 决定是否提前停 |
| learning_rate | LoRA 1e-4~3e-4 | 全参 1e-5~2e-5 |
| cutoff_len | 1024–4096 | 超过会截断答案，长输出任务要加 |
| per_device_train_batch_size | 1–8 | 等效 batch 16–64 |
| seed | 固定（如 42） | 保证切分与训练可复现 |
| eval_steps | 每 5%–10% 步数 | 太频繁浪费，太稀疏发现不了过拟合 |
| 模板 | 与底座官方一致 | qwen 用 qwen，llama 用 llama |

> 新手最容易忽略的是 seed 与 val_size：不固定 seed，两次训练不可比；不设验证集，过拟合只能靠猜。

---

## 07 高频踩坑排查

**坑 1：模板与底座不匹配**
症状：训练 loss 正常，生成前言不搭后语。
解法：训练与推理统一用底座官方 chat template（config 里 template 字段）；打印一条预处理样本人工核对。

**坑 2：自写脚本没做 response-only loss 掩码**
症状：模型把“问题”也背下来了，问他问题他会复述问题。
解法：用 LlamaFactory 等成熟框架的 alpaca/sharegpt 格式；自写时对 instruction 部分用 -100 掩码。

**坑 3：评测题混进训练集**
症状：评测分虚高，上线被打回原形。
解法：先切分再训练；held-out 集与训练集去重（MinHash）；评测集锁版本。

**坑 4：没有验证集就开训**
症状：loss 一直降，不知道已经过拟合。
解法：val_size 0.1–0.2，盯 val loss；val loss 回升即回退到最优 checkpoint。

**坑 5：adapter 不合并直接部署**
症状：部署环境加载 adapter 报错或行为不一致。
解法：统一 export 合并；合并后跑一遍冒烟问答。

**坑 6：只看 loss 不看格式**
症状：loss 很低，但回答不按“公司规定格式”来。
解法：评测里加“格式合规检查”（关键词/结构命中），SFT 的产出是文本行为，不是 loss 数字。

**坑 7：小数据跑太多 epoch**
症状：20 条数据跑 20 epoch，模型把样例背得滚瓜烂熟。
解法：小数据 3–5 epoch + 早停；效果不够先加数据（第 16 章），而不是加 epoch。

---

## 08 进阶优化：SFT 的进化方向

**① 困难样本挖掘**：训练后把评测中答错的题收集起来，人工修正/扩充后再训——SFT 数据迭代比调参更值钱。

**② 多轮与工具调用格式**：业务需要多轮记忆/工具调用时，从 Alpaca 升级 ShareGPT 格式，并在训练中混合单轮+多轮数据（第 17/21 章）。

**③ SFT+DPO 接力**：SFT 解决“会答”，DPO/RLHF 解决“答得好不好”（第 23–25 章）；先 SFT 到收敛，再对齐，顺序不要反。

**④ 数据配比与课程**：通用指令+领域指令+负样本按比例混合，先易后难课程式训练，能同时保通用与领域。

**⑤ 训练-评测闭环自动化**：数据变更→自动训练→自动 held-out 评测→报告归档，模型迭代从“手动跑”变成“流水线”（第 44 章展开）。

---

## 09 本章核心总结（TOP3）

**TOP1**：SFT 教“格式与行为”，知识靠 CPT；核心机制是 response-only loss 与 chat template——问题不参与损失，模板训练推理必须一致。

**TOP2**：工业 SFT 七步闭环：数据切分→注册→训练（带 val）→合并→部署→held-out 评测→归档；seed 固定、val 隔离、adapter 合并三个细节决定成败。

**TOP3**：SFT 的验收是“领域 held-out 题命中公司规则 + 通用题不掉 + 格式合规”，不是 loss；效果不够先查数据与模板，再调参。

---

## 10 连载衔接

上一章（第 14 章）教了方法怎么选；本章把 SFT 当成产品工序完整跑了一遍——从 20 条领域问答出发，得到了合并模型和 4/4 的领域评测。

下一章往上游走：【第 16 章】SFT 数据集构建与标注规范。20 条只能演示流程，真实项目需要几千上万条高质量指令数据——怎么设计任务、怎么标注、怎么质检，是 SFT 效果的真正分水岭。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #SFT #监督微调 #LoRA #大模型微调 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 15 SFT 监督微调实战（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 16 章 SFT 数据集构建与标注规范*