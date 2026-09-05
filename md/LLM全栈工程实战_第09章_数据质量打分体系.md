# 【LLM全栈工程·第09章】数据质量打分体系：规则、分类器与多维度融合

> 本篇为「LLM全栈工程」连载第 09 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 清洗只回答“脏不脏”，质量分回答“值不值得学”；② 给每一条训练数据打分：质量信号的选取与融合；③ 高分语料怎么来？从人工金样到自动打分的闭环。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「预训练工程·数据核心层」第 7 篇：为通过前几道闸门的数据建立 0–1 质量分，支撑配比、采样与训练退火 |
| 适用场景 | 个人学习（本地打分实验）；小团队语料筛选；企业预训练/增量预训练数据分级 |
| 核心知识点 | 打分与过滤的边界；质量信号；多信号融合；人工金样校准；按分采样与分层 |
| 技术选型 | 启发式特征 / FastText 类分类器 / 困惑度 / embedding 多样性 / LLM 打分 |
| 分步实操 | 在 safe 层后加 scored 层：注入高/中/低样例 → 多维特征打分 → 分桶输出 → 人工抽检 → 打版本 |
| 参数详解 | 信号权重、分桶阈值、长度/标点/多样性归一化参数、perplexity 分位 |
| 踩坑排查 | 单信号误判、权重拍脑袋、LLM 打全量、分数被“刷”、分布漂移无监控 |
| 进阶优化 | 人工金样回归、分类器打分、按分配比、训练退火、分数监控 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 10 章：领域专属数据构建 |

---

## 01 开篇导语

到上一章为止，你的语料已经过了五关：清洗、过滤、去重、脱敏——能活下来的都是“合格数据”。

但“合格”和“优质”是两回事。同一批合格数据里：

- 有的是 3000 字的结构化技术手册（信息密度极高）；
- 有的是 30 字的“好的收到明天见”（没信息量但不违规）；
- 有的是机器翻译腔的说明书（能读但别扭）。

如果训练时一视同仁，模型会浪费大量学习预算在低信息文本上。**工业级做法是给每条数据打一个 0–1 的质量分**，然后让分数参与三个决策：

1. **采样**：高分数据多采样、低分数据少采样；
2. **配比**：不同来源/类型按质量分布调整混合比例；
3. **训练退火**：训练后期把高质量数据占比提高，做“精修”。

本章讲清楚：质量分由哪些信号构成、怎么融合、怎么用人工标注校准，并给你一套跑得起来的打分脚本。

> 一句话记住本章：**过滤是“要不要”，打分是“给多少权重”；没有质量分的语料，等于没有导航的舰队。**

---

## 02 白话原理：质量分 = 多维信号的加权投票

### 2.1 为什么不能用“一个指标”定质量

“质量”是个复合概念：一段文本可能长度合格但重复啰嗦，可能语言通顺但全是空话，可能信息密集但格式混乱。单一指标（比如长度）很容易被“注水长文”骗过。所以质量分要由多个信号投票。

### 2.2 打分与前面章节的关系

| 环节 | 决策 | 输出 |
|---|---|---|
| 清洗（05） | 文本干不干净 | 保留/删除/改写 |
| 过滤（06） | 该不该留 | 保留/删除 |
| 去重（07） | 是不是重复 | 保留/删除 |
| 脱敏（08） | 有没有 PII | 保留/掩码/删除 |
| **打分（本章）** | **值多少权重** | **0–1 分 + 分桶** |

前四关是“闸门”，本章是“天平”——同一个 doc_id 从 sources 到 scored，一路带着 manifest 血缘，每个环节的决策都可回查。

### 2.3 分数怎么用：三种典型策略

- **硬阈值**：score < 0.3 直接弃用（相当于补一道柔性过滤）；
- **按分采样**：score 0.9 的文档以 1.0 概率采，0.5 的以 0.3 概率采；
- **分层配比**：把语料按分数分桶，训练时控制每桶的 token 占比，防止低质桶淹没高质桶。

> 生产上推荐“分层 + 采样”而不是“一刀切”：低分桶里偶尔也有稀缺语料（比如某方言、某冷门格式），直接删光可能误伤多样性。

## 03 质量信号：从五个维度看一条文本

| 维度 | 信号 | 说明 |
|---|---|---|
| 结构完整 | 长度、段落数、结尾标点率 | 截断/碎片通常分低 |
| 语言自然 | 困惑度（KenLM/小模型）、标点分布 | 机器翻译、拼接文通常异常 |
| 信息密度 | 唯一 n-gram 比例、去停用词密度 | 复读机/空话比例高则分低 |
| 风格规范 | 感叹号密度、全大写比例、emoji 占比 | 营销/喊话风格降权 |
| 语义质量 | 分类器“优质 vs 低质”概率 | 需要标注训练（见第 04 节） |

### 3.1 两个关键信号的白话解释

**困惑度（Perplexity，PPL）**：用一个语言模型给文本打分，模型“越意外”分数越高。人话文本的 PPL 通常低，机器翻译/乱序文本 PPL 高。它适合做“像不像自然语言”的度量，但要注意：领域术语会让 PPL 虚高，所以要按语料类型分别建基线。

**唯一 n-gram 比例（多样性）**：把文本切成 2-gram/3-gram，统计“不重复的比例”。全是“好的好的收到收到”的文本，重复比例极高；信息密集的文档，重复比例低。它是零成本、很有效的“复读机探测器”。

### 3.2 多信号融合：加权和 + 校准

最简单的融合是加权和：

~~~text
score = w1×结构分 + w2×语言分 + w3×信息密度分 + w4×风格分 + w5×语义分
~~~

但权重不能拍脑袋，正确流程是：

1. **抽 300–1000 条**样本，人工标 0–1（或分 5 档）；
2. 对每个信号看它与人工分的相关性（哪些信号有用）；
3. 用线性回归/简单网格搜权重，使“自动分与人工分”的排序一致性最高；
4. 固定权重与分桶阈值，版本化。

> 校准的验收指标：自动分与人工分的 Spearman 相关系数（排序一致性），以及分桶后人工抽检的“高桶确实比低桶好”的比例。没有金样校准的权重都是玄学。

---

## 04 技术选型：打分手段怎么组合

| 手段 | 成本 | 能力 | 适合阶段 |
|---|---|---|---|
| 启发式特征（本章代码） | 极低 | 长度/多样性/风格 | 第一步，先跑起来 |
| FastText 类分类器 | 低 | “低质 vs 优质”语义判断 | 有几百条标注后 |
| KenLM/小模型困惑度 | 低中 | 语言自然度 | 中大规模 |
| Embedding 多样性/相似度 | 中 | 查重与多样性 | 与去重/采样配合 |
| LLM 打分 | 高 | 综合质量判断最接近人 | 只用于金样标注与抽检，不用于全量 |

生产推荐分层：**规则特征全量跑（便宜）→ 困惑度/分类器跑中危样本 → LLM 只抽几百条做金样校准**。把 LLM 当“老师”给规则特征打标签，是性价比最高的路线。

## 05 分步实操：给管道加上 scored 层（30 分钟）

继续沿用 llm-demo/data-pipeline。scored 层读 safe/safe.jsonl，为每条文本产出 quality_score。

### Step 1 注入不同质量的样本（5 分钟）

保存 scripts/make_quality_sources.py：

~~~python
# data-pipeline/scripts/make_quality_sources.py —— 注入高/中/低质量样本
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
sources = root / "sources"

high = "公司打印机 HP LaserJet M405 安装与故障处理完整指南：第一步，从公司软件中心下载对应驱动；第二步，安装时选择 USB 或网络连接，网络连接请确认打印机 IP 在 10.20.x.x 网段；第三步，安装完成后打印测试页验证。常见故障：卡纸时先关机断电，打开前盖取出卡纸碎片，合盖重启；打印模糊时检查墨粉余量与打印浓度设置；无法发现打印机时检查网络指示灯并重新添加设备。以上步骤仍不能解决时，请提交 IT 工单并注明打印机 IP 与错误码。"

rows = [
    {"text": high, "meta": {"channel": "quality_high"}},
    {"text": "问题：WiFi 连不上怎么办？答案：先确认连接 Corp-WiFi，再忘记网络重连，仍失败请提交工单。", "meta": {"channel": "quality_mid"}},
    {"text": "好的好的 收到收到 明天再说 拜拜 辛苦啦", "meta": {"channel": "quality_low_chat"}},
    {"text": "该打印机驱动程序的安装过程非常简单容易，首先你需要下载安装包，然后你需要运行安装包，接下来你需要点击下一步按钮，最后你就可以完成安装了。", "meta": {"channel": "quality_machine"}},
]

with open(sources / "helpdesk_quality.jsonl", "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("quality rows:", len(rows))
~~~

### Step 2 全量重跑前序管道

~~~bash
python data-pipeline/scripts/make_quality_sources.py
python data-pipeline/scripts/normalize_sources.py
python data-pipeline/scripts/clean_sources.py
python data-pipeline/scripts/filter_sources.py
python data-pipeline/scripts/dedup_sources.py
python data-pipeline/scripts/mask_pii.py
wc -l data-pipeline/safe/safe.jsonl
~~~

这些样本没有 PII、不算脏、不重复，会全部进入 safe 层——但它们的信息密度天差地别，正是打分环节要区分的事。

### Step 3 写多维打分脚本（15 分钟）

保存 scripts/score_sources.py：

~~~python
# data-pipeline/scripts/score_sources.py —— 多维特征融合打分（0-1）
import json
import pathlib
import re

root = pathlib.Path(__file__).resolve().parent.parent
safe_dir = root / "safe"
scored_dir = root / "scored"
manifests = root / "manifests"
scored_dir.mkdir(parents=True, exist_ok=True)

# 权重（先用启发式初值，再用人工金样校准，不要直接当最终答案）
W = {
    "diversity": 0.30,
    "length": 0.25,
    "end_punct": 0.20,
    "punct": 0.15,
    "no_exclaim": 0.10,
}
BUCKETS = {"high": 0.6, "mid": 0.4}

def cjk_ratio(text):
    total = 0
    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            total += 1
    return total / max(1, len(text))

def diversity_score(text):
    if len(text) < 2:
        return 0.0
    grams = [text[i:i + 2] for i in range(len(text) - 1)]
    return len(set(grams)) / max(1, len(grams))

def length_score(text):
    return min(1.0, len(text) / 500.0)

def punct_score(text):
    count = sum(1 for ch in text if ch in "，。！？；：、,.;:!?")
    return min(1.0, count / max(1, len(text)) * 12.0)

def end_punct_score(text):
    count = sum(1 for ch in text if ch in "。！？!?")
    return min(1.0, count / max(1, len(text)) * 80.0)

def no_exclaim_score(text):
    count = sum(1 for ch in text if ch in "！!")
    return 1.0 - min(1.0, count / max(1, len(text)) * 20.0)

rows = [json.loads(line) for line in open(safe_dir / "safe.jsonl", encoding="utf-8")]

out_rows = []
bucket_count = {"high": 0, "mid": 0, "low": 0}
for obj in rows:
    text = str(obj.get("text", ""))
    signals = {
        "chars": len(text),
        "cjk_ratio": round(cjk_ratio(text), 4),
        "diversity": round(diversity_score(text), 4),
        "end_punct": round(end_punct_score(text), 4),
    }
    score = (
        W["diversity"] * diversity_score(text)
        + W["length"] * length_score(text)
        + W["end_punct"] * end_punct_score(text)
        + W["punct"] * punct_score(text)
        + W["no_exclaim"] * no_exclaim_score(text)
    )
    score = round(max(0.0, min(1.0, score)), 4)
    if score >= BUCKETS["high"]:
        bucket = "high"
    elif score >= BUCKETS["mid"]:
        bucket = "mid"
    else:
        bucket = "low"
    bucket_count[bucket] += 1
    obj["quality_score"] = score
    obj["quality_bucket"] = bucket
    obj["quality_signals"] = signals
    out_rows.append(obj)

with open(scored_dir / "scored.jsonl", "w", encoding="utf-8") as f:
    for obj in out_rows:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

with open(scored_dir / "score_distribution.json", "w", encoding="utf-8") as f:
    json.dump({"buckets": bucket_count, "weights": W}, f, ensure_ascii=False, indent=2)

print("bucket_count:", bucket_count)
print("total:", len(out_rows))
~~~

### Step 4 跑打分并检查分桶（5 分钟）

~~~bash
python data-pipeline/scripts/score_sources.py
cat data-pipeline/scored/score_distribution.json
python -c "import json; rows=[json.loads(l) for l in open('data-pipeline/scored/scored.jsonl',encoding='utf-8')]; rows.sort(key=lambda r:-r['quality_score']); [print(round(r['quality_score'],2), r['quality_bucket'], r['text'][:40]) for r in rows[:6]]"
~~~

预期：完整技术指南落在 high；正常问答落在 high/mid；机器翻译腔落在 mid；闲聊“好的好的收到收到”落在 low。

### Step 5 人工金样校准（5 分钟，核心动作）

1. 从 scored.jsonl 里高/中/低桶各抽 10–20 条；
2. 不看自动分，人工打 0–1 分；
3. 对比自动分与人工分：如果“低桶里混着明显高质量文档”，说明权重或信号要调（比如加困惑度信号）；
4. 把人工分存成 data-pipeline/manifests/golden_scores.jsonl，作为下一版权重回归的训练数据。

### Step 6 打版本

~~~bash
cd llm-demo
git add data-pipeline
git commit -m "score: multi-signal v0.1 (diversity/length/punct) + buckets + golden starter"
git tag data-score-v0.1
~~~

> 生产提示：golden_scores 每轮扩充 100–300 条，攒到 1000 条后就可以训练 FastText 分类器，把“规则打分”升级成“规则召回+分类器精排”（见本章进阶优化）。

## 06 参数详解：打分配置速查

| 参数 | 参考取值 | 说明 |
|---|---|---|
| 分桶阈值 | high≥0.6，mid≥0.4 | 按语料分布调整：优质语料多时整体上移 |
| 长度归一化上限 | 300–2000 字符 | 短问答语料用 300，长文语料用 2000 |
| 标点归一化系数 | 8–15 | 让正常文本落在 0.5–0.9 区间 |
| 困惑度分位线 | 85–95 分位 | 高于分位线的判低质（需先建语料基线） |
| 分类器阈值 | 0.5–0.7 | 由金样集 precision/recall 决定 |
| 金样规模 | 起步 300，稳定 1000+ | 太少权重不可靠，太多标注成本高 |
| 抽样比例 | 高桶 1.0 / 中桶 0.5 / 低桶 0.1 | 按训练预算调节，别把低桶清零 |

> 权重与阈值必须跟“语料版本”一起管理：换一个数据源后分布会变，先重测金样再决定要不要调参。

---

## 07 高频踩坑排查

**坑 1：用单一信号当质量分**
症状：按“长度>500”筛数据，注水长文全进来了，高质量短问答被扔掉。
解法：多信号融合（结构/语言/密度/风格/语义），并给每个信号看与人工分的相关性。

**坑 2：权重拍脑袋，从不校准**
症状：0.3/0.25/0.2 全靠感觉，分桶结果和人工直觉对不上。
解法：抽 300+ 金样做线性回归/网格搜索，用排序相关性验收；金样进 Git 版本管理。

**坑 3：拿 LLM 给 TB 级语料全量打分**
症状：贵且慢，还引入 LLM 自身偏好（喜欢长句、喜欢特定风格）。
解法：LLM 只用于金样标注与抽检；全量用规则+小模型。

**坑 4：打分只看“像不像好文本”，不看领域价值**
症状：一段领域黑话密集但格式乱的专家笔记被打了低分。
解法：打分维度里加入“领域信号”（术语密度、领域词典命中）；不同领域各自建金样，不要一套权重走天下。

**坑 5：分数被“刷”**
症状：内容农场发现高分特征后批量生产“高分模板文”。
解法：定期把高分桶做人工抽检与 MinHash 复查；分数特征要防游戏化（比如多样性特征配长度惩罚）。

**坑 6：只打分不监控分布漂移**
症状：新数据源接入后分数分布悄悄变了，没人发现。
解法：每批次记录 score 分布（均值/分位/桶占比），与基线对比，漂移超阈值触发告警与人工复查。

---

## 08 进阶优化：质量打分的进化路径

**① 金样驱动的权重回归**：把人工分当作 y，信号当作 x，用逻辑回归/排序模型学权重；样本攒到 1000+ 后，升级成 FastText 分类器直接预测“优质概率”。

**② 按分配比与退火**：训练时按分数分桶控制 token 占比；预训练后期做“高质量数据退火”——把高分段比例逐步提高、学习率同步下降，相当于给模型做“考前冲刺”（第 11 章增量预训练会用到）。

**③ 多样性约束采样**：打分之外叠加“多样性预算”：同一领域/同一来源的高分文档太多时按 embedding 相似度降采样，避免高分桶变成同质桶。

**④ LLM 蒸馏打分器**：用 LLM 给几千条样本打分作为软标签，训练一个小分类器逼近 LLM 的判断——把 LLM 的质量观“蒸馏”进便宜模型，全量可用。

**⑤ 分数反哺数据发现**：低分桶不等于垃圾：定期人工翻查低分桶里的“异常高分特征缺失样本”，往往能发现新语料类型（方言、新格式），再为它们建专属信号。

---

## 09 本章核心总结（TOP3）

**TOP1**：质量分 = 多维度信号（结构/语言/信息密度/风格/语义）的加权融合；过滤是“要不要”，打分是“给多少权重”，两者不能互相替代。

**TOP2**：权重必须用人工金样校准：300 条起步、1000+ 条稳定，用排序相关性验收；LLM 只当金样老师，不直接给 TB 级数据打分。

**TOP3**：分数的用途是采样、配比与训练退火；要防“刷分”、要监控分布漂移，并保留低分桶的多样性——高分不是唯一目标，均衡才是。

---

## 10 连载衔接

上一章（第 08 章）让语料不再携带个人信息；本章给每条数据贴上 quality_score 与分桶标签——至此，通用数据管道“清洗→过滤→去重→脱敏→打分”全部就位。

下一章把目光转向“专精”：【第 10 章】领域专属数据构建：采集、清洗与合成数据。通用语料解决“会说话”，领域数据决定“懂行”——它是主线工程从通用底座走向领域模型的关键一跃。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #数据质量 #质量打分 #大模型训练数据 #数据工程 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 09 数据质量打分体系（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 10 章 领域专属数据构建*