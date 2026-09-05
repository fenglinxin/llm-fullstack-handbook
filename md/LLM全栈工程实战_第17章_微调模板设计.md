# 【LLM全栈工程·第17章】微调模板设计：Chat Template、角色设定与多轮拼接

> 本篇为「LLM全栈工程」连载第 17 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 训练 loss 正常但生成乱套？先查 Chat Template；② system/user/assistant 到底怎么排：微调模板设计实战；③ 多轮拼接的隐藏细节：EOS、截断与角色标记。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「微调与对齐体系」第 4 篇：解决“模型怎么组织对话输入”的模板问题 |
| 适用场景 | 个人学习（单轮 SFT）；小团队做领域助手；多轮/工具调用前的基础设施 |
| 核心知识点 | Chat Template 机制；角色与特殊 token；训练/推理一致性；多轮拼接与截断 |
| 技术选型 | 官方模板 vs 自定义模板 vs 无模板；Alpaca vs ShareGPT 数据格式 |
| 分步实操 | 模板体检脚本 → 打印真实渲染 → 单轮数据集校验 → 多轮样例生成与验证 |
| 参数详解 | 特殊 token、system 长度、max_turns、截断方向、模板版本 |
| 踩坑排查 | 模板不一致、双 BOS、丢 EOS、system 只在推理加、截断切断回答 |
| 进阶优化 | 模板版本化、按任务换 system、动态上下文注入、工具调用模板 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 18 章：微调超参调优策略 |

---

## 01 开篇导语

上一章我们造了一批高质量指令数据。但数据进模型前还有最后一道“翻译”：**把 instruction/input/output 翻译成模型认识的对话格式**——这就是 Chat Template（聊天模板）。

很多团队在这里翻车：

- 训练用 A 模板，推理用 B 模板，模型“精分”；
- 自己拼字符串时把 BOS 加了两遍、EOS 丢了；
- system 提示只在推理时加、训练数据里没有——上线就失灵。

模板问题的诡异之处在于：**loss 一切正常，生成完全不对**。因为模板错误不改变“能训练”这个事实，只改变“模型学到的东西是不是你想要的”。

本章讲清三件事：

1. Chat Template 到底是什么、长什么样；
2. 角色（system/user/assistant）与特殊 token 怎么设计；
3. 多轮对话怎么拼接、怎么截断才不伤模型。

交付物：一个“模板体检”脚本 + 单轮/多轮数据校验流程。

> 一句话记住本章：**模板是训练与推理之间的“协议”——协议不一致，模型学得再好也白搭。**

---

## 02 白话原理：模板 = 角色 + 特殊标记 + 拼接规则

### 2.1 一次对话在 tokenizer 眼里长什么样

以 Qwen 系官方 chat 模板为例，一段“用户问、助手答”会被渲染成类似：

~~~text
<|im_start|>system
你是公司 IT 帮助台助手……<|im_end|>
<|im_start|>user
打印机连不上怎么办？<|im_end|>
<|im_start|>assistant
先检查电源与网络指示灯……<|im_end|>
~~~

其中：

- 角色标记（system/user/assistant）告诉模型“现在谁在说话”；
- 特殊 token（<|im_start|>/<|im_end|> 这类）是词表里的专用符号，不是普通文本；
- 整个字符串再被 tokenizer 切成 token id 序列喂给模型。

### 2.2 模板的三个作用

| 作用 | 说明 |
|---|---|
| 区分角色 | 模型学会“看到 assistant 标记就该回答” |
| 界定边界 | 特殊 token 让模型知道一段话在哪结束 |
| 注入人设 | system 段稳定携带角色/规则/输出格式 |

### 2.3 为什么必须用官方模板

每个底座模型在预训练/指令微调时都用了自己的模板（Qwen 有 Qwen 的、Llama 有 Llama 的）。**换模板 = 让模型面对一种它没见过的输入格式**，能力会明显下降。所以第一原则：训练与推理都用该模型官方 chat template，不自己发明格式。

> 记忆锚点：**模板是模型的“母语语法”，你微调是在教它用母语写新内容，而不是教它一门新语言。**

## 03 角色设计与数据格式

### 3.1 角色设计原则

| 角色 | 放什么 | 注意 |
|---|---|---|
| system | 人设、总规则、输出格式、安全边界 | 保持简短稳定；不要放会变的临时信息 |
| user | 用户问题/任务 | 训练时指令放这里（参与 loss 掩码） |
| assistant | 标准答案 | 唯一参与 loss 的部分 |
| tool（如支持） | 工具结果 | 工具调用场景专用（第 21 章展开） |

设计注意：

1. **system 要“训练里有、推理也有”**：只在推理时加的 system，模型没学过，等于没有；
2. **system 不是垃圾桶**：把整本知识库塞进 system，既浪费长度又稀释注意力；
3. **人设要与数据一致**：数据里助手自称“IT 助手”，system 就别写“你是诗人”。

### 3.2 两种主流数据格式

| 格式 | 结构 | 适合 | LlamaFactory 模板名 |
|---|---|---|---|
| Alpaca | instruction/input/output | 单轮指令 | qwen/llama 等按底座 |
| ShareGPT | conversations[from/value] | 多轮/系统角色 | 同上（formatting=sharegpt） |

- 单轮起步用 Alpaca，简单；
- 涉及 system 人设与多轮时，**建议直接切 ShareGPT**：角色显式、不易拼错、支持多轮。

### 3.3 LlamaFactory 的 template 字段

config 里 template 与底座对应：Qwen 系填 qwen，Llama 系填 llama，ChatML 风格模型可能填 chatml——**以底座官方文档与 LlamaFactory 支持列表为准**。选错模板的典型后果：训练能跑，但生成格式错乱。

## 05 分步实操：模板体检与数据校验（30 分钟）

### Step 1 写“模板体检”脚本（10 分钟）

保存 scripts/template_doctor.py：

~~~python
# scripts/template_doctor.py —— 打印官方模板渲染，检查关键标记
import json
import pathlib

from transformers import AutoTokenizer

root = pathlib.Path(__file__).resolve().parent.parent
model_dir = str(root / "models" / "Qwen2.5-7B-Instruct")
tok = AutoTokenizer.from_pretrained(model_dir)

conv = [
    {"role": "system", "content": "你是公司 IT 帮助台助手，回答要简洁、按公司规定。"},
    {"role": "user", "content": "打印机连不上怎么办？"},
    {"role": "assistant", "content": "先检查电源与网络指示灯，重启打印机，仍失败请提交工单。"},
]

rendered = tok.apply_chat_template(conv, tokenize=False)
print("===== 渲染结果 =====")
print(rendered)
print("===== 特殊 token =====")
print(tok.special_tokens_map)

ids = tok.apply_chat_template(conv, tokenize=True)
print("token 数:", len(ids))

# 检查 1：生成模式下结尾必须是 assistant 引导
gen_ids = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=True)
print("生成模式结尾 3 个 token id:", gen_ids[-3:])

# 检查 2：数据集逐条渲染是否正常
src = root / "data" / "sft_dataset_v1.jsonl"
errors = []
count = 0
for idx, line in enumerate(open(src, encoding="utf-8")):
    row = json.loads(line)
    user_content = row["instruction"] + ("\n" + row["input"] if row.get("input") else "")
    msg = [
        {"role": "system", "content": "你是公司 IT 帮助台助手。"},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": row["output"]},
    ]
    try:
        text = tok.apply_chat_template(msg, tokenize=False)
        if "<|im_start|>assistant" not in text:
            errors.append({"row": idx, "type": "no_assistant_marker"})
    except Exception as exc:
        errors.append({"row": idx, "type": "render_error", "detail": str(exc)})
    count += 1

print("数据集渲染检查:", count, "条, 异常:", len(errors))
if errors:
    print(errors[:3])
~~~

运行（需在 llm-demo 虚拟环境中且已下载模型）：

~~~bash
python scripts/template_doctor.py
~~~

**看什么**：渲染文本里 system/user/assistant 标记齐全；生成模式结尾是 assistant 引导；数据集所有行都能正常渲染且含 assistant 标记。

### Step 2 生成多轮样例并验证拼接（15 分钟）

保存 scripts/make_multiturn_data.py：

~~~python
# scripts/make_multiturn_data.py —— ShareGPT 多轮样例（含 system 与追问）
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
out_path = root / "data" / "multiturn_demo.jsonl"

samples = [
    {
        "conversations": [
            {"from": "system", "value": "你是公司 IT 帮助台助手，按公司规定回答。"},
            {"from": "human", "value": "你好，在吗？"},
            {"from": "gpt", "value": "你好，我是 IT 帮助台助手，有什么可以帮你？"},
            {"from": "human", "value": "打印机连不上怎么办？"},
            {"from": "gpt", "value": "先检查电源与网络指示灯；重启打印机；删除并重新添加 HP LaserJet M405；仍失败请提交工单，注明打印机 IP 与错误码。"},
            {"from": "human", "value": "按你说的重启了还是不行。"},
            {"from": "gpt", "value": "请直接提交 IT 工单并注明打印机 IP 与错误码，工程师会远程排查。"},
        ]
    },
    {
        "conversations": [
            {"from": "system", "value": "你是公司 IT 帮助台助手，按公司规定回答。"},
            {"from": "human", "value": "VPN 账号被锁了，能马上解锁吗？"},
            {"from": "gpt", "value": "账号连续输错 5 次会锁定 30 分钟；紧急解锁请联系 IT 热线。"},
            {"from": "human", "value": "好的，那我 30 分钟后再试。"},
            {"from": "gpt", "value": "可以。如果仍无法登录，请提交 VPN 权限申请或联系 IT 热线。"},
        ]
    },
]

with open(out_path, "w", encoding="utf-8") as f:
    for item in samples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print("multiturn samples:", len(samples), "->", out_path)
~~~

运行：

~~~bash
python scripts/make_multiturn_data.py
~~~

在 data/dataset_info.json 中注册（ShareGPT 格式，字段名以 LlamaFactory 版本为准）：

~~~json
{
  "multiturn_demo": {
    "file_name": "multiturn_demo.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations"},
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt",
      "system_tag": "system"
    }
  }
}
~~~

**手工验证拼接**：把第一条 conversation 手动按官方模板展开，确认每轮 assistant 回答后都有结束标记、human 下一轮前有正确的 user 起始标记——这一步肉眼看过一次，就再也不会犯“拼接错位”的错。

### Step 3 打版本

~~~bash
cd llm-demo
git add data scripts
git commit -m "template: doctor v1 + multiturn sharegpt demo (2 samples)"
git tag template-v1
~~~

> 第 21 章会展开多轮数据规模化与长上下文处理；本章先确保“拼得对、验得过”。

## 06 参数详解：模板设计速查

| 参数/项 | 参考取值 | 说明 |
|---|---|---|
| template 名 | 与底座官方一致（qwen/llama/chatml…） | 训练与推理同源 |
| system 长度 | 50–500 token | 放稳定人设/规则，别塞动态知识 |
| 单轮最大轮数 | 训练 2–8 轮（按业务） | 推理上下文更长时可逐步加 |
| 截断方向 | 从左截断（保留最近对话） | 多轮超长时优先丢最早轮次 |
| EOS 完整性 | 每个 assistant 回答后必有 | 缺失会导致模型不会“闭嘴” |
| 数据集格式 | 单轮 Alpaca / 多轮 ShareGPT | 有 system 与多轮就上 ShareGPT |
| 模板版本 | 进 Git + 数据 manifest | 换模板=换数据分布，必须重测 |

> 模板改动是“高影响低频率”操作：每次改完必须重跑 template_doctor + 训练前后对比，不要顺手改。

---

## 07 高频踩坑排查

**坑 1：训练与推理模板不一致**
症状：训练 loss 正常，上线生成格式错乱。
解法：训练与推理共用同一 tokenizer.apply_chat_template；把渲染结果打印出来人工对比。

**坑 2：自己手拼字符串，BOS/EOS 出错**
症状：文本开头多一个奇怪符号，或回答结尾没有结束符，模型一直说不停。
解法：不要手拼特殊 token，用官方模板 API；特殊 token 以 tokenizer 输出为准。

**坑 3：system 只在推理时加**
症状：离线评测挺好，上线加了 system 后行为突变。
解法：训练数据里也要有 system（或至少 30%–50% 样本带 system），与线上完全一致。

**坑 4：system 内容过长**
症状：模型被长 system 干扰，答非所问。
解法：system 只放人设/规则/格式；动态内容放 user 轮或走检索注入。

**坑 5：多轮截断切在回答中间**
症状：长对话训练时，某些样本的 assistant 回答被截掉一半，模型学到半截话。
解法：按“轮次边界”截断，宁可丢整轮，不切半句；记录截断策略进 manifest。

**坑 6：模板名填错**
症状：Qwen 模型填了 llama 模板，生成全是 <s>[INST] 风格乱码。
解法：以模型官方与框架文档为准；用 template_doctor 打印渲染结果核对。

**坑 7：多轮拼接错位**
症状：assistant 的话跑到 user 位置，模型角色混乱。
解法：用 ShareGPT 结构化格式而不是手工字符串拼接；每条样本渲染后自动校验角色顺序。

---

## 08 进阶优化：模板设计的进化方向

**① 模板版本化**：模板代码+特殊 token+system 文案一起版本化，训练 manifest 记录模板版本；线上模板升级要走灰度。

**② 按任务切换 system**：不同场景（客服/代码/合规）用不同 system，训练数据按场景分组各带各的 system——而不是一个万能 system 打天下。

**③ 动态上下文注入位设计**：给“知识库检索结果”预留固定插入位（如 user 轮前缀），训练时模拟插入，让模型习惯检索增强输入（与 RAG 章节衔接）。

**④ 工具调用模板**：训练工具调用时用专门 role（tool/function），保证模型只会在该角色出现时输出工具调用格式（第 21 章）。

**⑤ Token 效率优化**：统计模板 overhead（system+角色标记占比），长上下文场景压缩 system 模板，能省下可观的 KV Cache 与成本。

---

## 09 本章核心总结（TOP3）

**TOP1**：Chat Template 是训练与推理之间的协议：角色标记 + 特殊 token + 拼接规则，必须用底座官方模板并保持训练/推理一致，禁止手拼特殊 token。

**TOP2**：system 要“训练有、推理也有”，内容稳定简短；单轮用 Alpaca，带 system/多轮用 ShareGPT；每次模板改动都要打印渲染结果人工核对。

**TOP3**：多轮拼接的命门是“轮次边界完整”：assistant 回答后必有 EOS，超长从左截断不切半句；模板版本与数据 manifest 绑定，改动必须重测。

---

## 10 连载衔接

上一章（第 16 章）造好了多任务指令数据；本章给数据配上了正确的“语法”——模板体检通过、多轮拼接验证通过，SFT 的数据侧准备全部就绪。

下一章进入训练侧调优：【第 18 章】微调超参调优策略：rank/alpha/lr/epoch 与实验管理。同样的数据，为什么别人训出来效果好？差距往往在超参与实验纪律上。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #ChatTemplate #微调模板 #多轮对话 #ShareGPT #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 17 微调模板设计（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 18 章 微调超参调优策略*