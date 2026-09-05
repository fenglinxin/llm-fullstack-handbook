# 【LLM全栈工程·第07章】数据去重算法：从精确去重到 MinHash 近似去重

> 本篇为「LLM全栈工程」连载第 07 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 语料里一半是“复制粘贴”？预训练数据去重实战；② MinHash 到底在算什么：近似去重白话原理与工程落地；③ 去重不只是删重复文件：文档、句子、评测集三级防污染。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「预训练工程·数据核心层」第 5 篇：建立精确去重 + 近似去重的完整方案 |
| 适用场景 | 个人学习（本地小语料）；小团队多源语料治理；企业预训练/增量预训练与评测防泄漏 |
| 核心知识点 | 重复数据的危害；去重分层（精确/归一化/近似/包含）；MinHash + LSH 原理；评测集防污染 |
| 技术选型 | 精确哈希、归一化哈希、MinHash-LSH（datasketch）、后缀数组/段落去重、Bloom Filter |
| 分步实操 | 在第 06 章 filtered 层后加 dedup 层：注入重复样本 → 精确去重 → MinHash 近似去重 → 报告打版本 |
| 参数详解 | shingle 大小、num_perm、LSH threshold、归一化规则、去重粒度 |
| 踩坑排查 | 阈值过低误杀、shingle 过大漏重、未归一化、短文档失效、评测集污染、跨源重复漏网 |
| 进阶优化 | 句子/段落级去重、包含关系去重、分布式去重、评测泄漏检测 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 08 章：敏感数据脱敏 |

---

## 01 开篇导语

问你一个反直觉的问题：**一份语料重复 10 遍，和它完全不存在，哪个更糟？**

答案是“重复 10 遍”更糟——因为它不仅浪费 10 倍算力，还会让模型**过度学习这一段**：训练时反复见到同一段文本，模型会把它背下来，挤压其他知识的容量；更隐蔽的是，如果重复文本恰好出现在测试集里，评测分数会被污染，给你一个虚假的“模型很强”。

这正是预训练数据流水线里必须有“去重闸门”的原因。上一章过滤漏斗留了第四道闸门的位置，本章把它补上：

1. **精确去重**：内容一模一样（或只有空白/全半角差异）；
2. **近似去重**：同一篇文章被不同网站改了几个字、换了几段；
3. **包含/段落去重**：A 文档整段嵌进了 B 文档；
4. **评测防污染**：训练集不得与评测集“长得像”。

本章会讲清 MinHash 这个“用指纹找相似”的算法原理，并给你一套跑得起来的精确 + 近似去重代码。去重做完，你的数据才真正“干净且不重复”。

> 一句话记住本章：**去重不是删文件，是保护模型的泛化能力与评测可信度。**

---

## 02 白话原理：去重 = 给文档发“指纹”

### 2.1 为什么重复会伤害模型

| 危害 | 机制 | 表现 |
|---|---|---|
| 算力浪费 | 重复 token 白训 | 同样算力下有效语料变少 |
| 过拟合 | 高频重复文本被背诵 | 模型对特定段落“复读” |
| 能力失衡 | 重复内容挤占学习预算 | 少见的真知识学不到 |
| 评测污染 | 测试题在训练里见过 | 分数虚高、上线露馅 |

### 2.2 三种“指纹”思路

**① 精确指纹（哈希）**
把整篇文档做 sha256，一样就是重复。快、准，但“改一个标点”就失效。

**② 归一化指纹**
先把文本统一格式（去空白、全角转半角、小写化），再哈希。能抓“格式不同、内容相同”的重复。

**③ 相似指纹（MinHash）**
把文档拆成很多个小片（shingle），每个小片做哈希，取“一组随机哈希里的最小值”作为签名；两篇文档的签名重合度 ≈ 它们的相似度。**它回答的是“这两篇像不像”，而不是“是不是同一篇”。**

### 2.3 MinHash 的白话类比

想象给每篇文档做 128 次“随机抽题”：每次从文档的所有 5 字小片里抽“哈希值最小的那个小片”作为这一轮的答案。两篇相似的文档，抽到的“最小答案”大概率相同；完全不相关的文档，128 轮里碰巧相同的极少。

于是每篇文档变成 128 个数字的“指纹”。比对指纹时不用两两全文比对，而是用 **LSH（局部敏感哈希）** 把“某些轮答案相同”的文档扔进同一个桶——只在同桶里做精细确认。这就是“先粗筛、后精查”。

> 一句话：**MinHash 把“文档相似度”变成“指纹重合度”，LSH 把“全库两两比对”变成“只在候选桶里比对”。**

## 03 去重体系：四层防线怎么排

工业级去重不是“一个算法跑到底”，而是按“越来越贵”的顺序层层拦截：

| 层 | 手段 | 抓什么 | 成本 |
|---|---|---|---|
| L0 归一化 | 统一编码/空白/大小写 | 为后续哈希打基础 | 极低 |
| L1 精确去重 | sha256（归一化文本） | 完全相同文档 | 极低 |
| L2 近似去重 | MinHash + LSH | 转载/改写/换序文档 | 中 |
| L3 包含与段落去重 | 句子哈希、后缀数组、段落指纹 | A 整段在 B 中、模板套壳 | 高 |

**执行顺序建议**：清洗 → 过滤（第 05/06 章）→ L1 精确去重 → L2 近似去重 → L3 段落级去重 → 质量打分（第 09 章）。先去重再打分，可以避免同一内容的多个副本各自拿到高分，把打分预算浪费在重复上。

### 3.1 关键细节：去重粒度

| 粒度 | 抓的重复 | 误伤风险 |
|---|---|---|
| 整篇文档 | 整篇转载 | 低 |
| 段落/句子 | 拼接文、模板文 | 中（引用、名言、固定话术会误伤） |
| 定长窗口（如 50 token） | 局部复制 | 需要“重复窗口占比”阈值 |

正文引用、法律条文、代码样板天然重复，去重时要加“允许重复清单”（如开源 License 文本、固定声明），或把重复率阈值与文档长度挂钩。

### 3.2 评测集防污染（最容易忽略的一层）

**训练去重和评测去重是两件事**：训练去重是为了泛化，评测去重是为了“分数可信”。正确姿势是把评测集（或评测集的问题/答案）也纳入去重：凡与评测样本近似的训练文档，一律剔除——否则你测的不是模型能力，是模型的记忆力。

---

## 04 技术选型：什么时候用什么

| 方案 | 抓什么 | 优点 | 缺点 | 适用 |
|---|---|---|---|---|
| 精确哈希（sha256） | 完全一致 | 零误杀、极快 | 改一字即失效 | 必做第一层 |
| 归一化哈希 | 格式差异 | 同上+抓格式变体 | 同义词改写失效 | 第一层升级 |
| MinHash + LSH | 近似重复 | 可设相似度阈值、可扩展 | 有误杀/漏网，需调参 | 第二层主力 |
| Bloom Filter | 集合成员判断 | 极省内存 | 有假阳性 | 超大库预筛 |
| 后缀数组类 | 包含/最长公共子串 | 精确抓包含 | 工程重、贵 | 大厂级第三层 |
| 段落指纹 | 模板套壳/拼接文 | 直观 | 需按文档类型调 | 新闻/网页语料 |

工具落地参考：Python 生态用 datasketch（MinHash/LSH）；分布式场景在 Spark/Ray 上实现同样的 shingle+MinHash 流程；Data-Juicer 类 LLM 数据处理框架内置多种去重算子，可直接接入管道。

结论：**起步标配 = 归一化 + sha256 精确去重 + MinHash-LSH 近似去重**；段落级与包含级去重等语料规模与重复形态明确后再上。

## 05 分步实操：给管道加上 dedup 层（30 分钟）

继续沿用 llm-demo/data-pipeline。本章演示需要 datasketch 库：

~~~bash
pip install datasketch
~~~

### Step 1 注入重复样本（5 分钟）

保存 scripts/make_dup_sources.py：

~~~python
# data-pipeline/scripts/make_dup_sources.py —— 注入精确重复与近似重复样本
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
sources = root / "sources"

dup_rows = [
    # 与第 04 章 KB 完全一致（格式加了空格差异，归一化后应命中精确去重）
    {"text": "规则 ： 公司打印机型号为 HP LaserJet M405 ；报修入口 http://it.example.local/printer", "meta": {"channel": "dup_exact_kb"}},
    # 与第 04 章 QA 近似重复（插入少量字词）
    {"text": "问题：打印机连不上怎么办？\n答案：检查电源与网络指示灯；重启打印机；删除并重新添加打印机（型号为 HP LaserJet M405）；仍然失败请提交IT工单。", "meta": {"channel": "dup_near_qa"}},
    # WiFi 规则的近似改写
    {"text": "员工使用的 WiFi 名称是 Corp-WiFi，要用域账号登录，忘记密码就去自助平台重置。", "meta": {"channel": "dup_near_wifi"}},
]

with open(sources / "helpdesk_dup.jsonl", "w", encoding="utf-8") as f:
    for row in dup_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("dup rows:", len(dup_rows))
~~~

### Step 2 重跑“归一化 → 清洗 → 过滤”，让重复样本进入 filtered 层

~~~bash
python data-pipeline/scripts/make_dup_sources.py
python data-pipeline/scripts/normalize_sources.py
python data-pipeline/scripts/clean_sources.py
python data-pipeline/scripts/filter_sources.py
wc -l data-pipeline/filtered/keep.jsonl
~~~

> 注意：重复样本本身是“干净且无害”的，所以会顺利通过前两道闸门——它们必须由专门的去重闸门来拦。

### Step 3 写去重脚本（15 分钟）

保存 scripts/dedup_sources.py：

~~~python
# data-pipeline/scripts/dedup_sources.py —— L1 精确去重 + L2 MinHash 近似去重
import hashlib
import json
import pathlib
import re
import unicodedata

from datasketch import MinHash, MinHashLSH

root = pathlib.Path(__file__).resolve().parent.parent
filtered_dir = root / "filtered"
dedup_dir = root / "dedup"
manifests = root / "manifests"
dedup_dir.mkdir(parents=True, exist_ok=True)

SHINGLE = 5        # 字符 shingle 长度（中文语料常用 4-6）
PERM = 128         # MinHash 签名长度
THRESHOLD = 0.6    # 相似度阈值（LSH 粗筛）

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def shingles(text, k=SHINGLE):
    if len(text) <= k:
        return [text]
    return [text[i:i + k] for i in range(len(text) - k + 1)]

def minhash_of(text):
    m = MinHash(num_perm=PERM)
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m

rows = [json.loads(line) for line in open(filtered_dir / "keep.jsonl", encoding="utf-8")]

# ---------- L1：精确去重（doc_id 相同 或 归一化后 sha256 相同） ----------
seen_ids = set()
seen_norm = {}
exact_dups = []
candidates = []
for obj in rows:
    doc_id = obj["doc_id"]
    if doc_id in seen_ids:
        # 相同 doc_id 说明正文完全相同（doc_id 是内容哈希），保留首条
        exact_dups.append({"doc_id": doc_id, "duplicate_of": doc_id, "method": "exact_same_doc"})
        continue
    norm = normalize_text(obj["text"])
    key = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    if key in seen_norm:
        # 内容仅格式不同（空白/全半角），归一化后相同
        exact_dups.append({"doc_id": doc_id, "duplicate_of": seen_norm[key], "method": "exact_norm"})
        continue
    seen_ids.add(doc_id)
    seen_norm[key] = doc_id
    candidates.append(obj)

# ---------- L2：MinHash + LSH 近似去重 ----------
lsh = MinHashLSH(threshold=THRESHOLD, num_perm=PERM)
mh = {}
for obj in candidates:
    m = minhash_of(obj["text"])
    mh[obj["doc_id"]] = m
    lsh.insert(obj["doc_id"], m)

keep = []
near_dups = []
dropped = set()
for obj in sorted(candidates, key=lambda o: o["doc_id"]):
    if obj["doc_id"] in dropped:
        continue
    keep.append(obj)
    for cand in lsh.query(mh[obj["doc_id"]]):
        if cand != obj["doc_id"] and cand not in dropped:
            dropped.add(cand)
            near_dups.append({"doc_id": cand, "duplicate_of": obj["doc_id"], "method": "minhash"})

# ---------- 输出 ----------
with open(dedup_dir / "unique.jsonl", "w", encoding="utf-8") as f:
    for obj in keep:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
with open(dedup_dir / "dedup_report.jsonl", "w", encoding="utf-8") as f:
    for item in exact_dups + near_dups:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("input:", len(rows), "unique:", len(keep))
print("exact_dups:", len(exact_dups), "near_dups:", len(near_dups))
~~~

### Step 4 跑去重并核对报告（5 分钟）

~~~bash
python data-pipeline/scripts/dedup_sources.py
wc -l data-pipeline/filtered/keep.jsonl data-pipeline/dedup/unique.jsonl
cat data-pipeline/dedup/dedup_report.jsonl
~~~

预期看到：精确重复的 KB 文档被 exact 命中；改写版打印机 QA 与 WiFi 规则被 minhash 命中；正常唯一文档全部保留。

### Step 5 调参实验（5 分钟）

把 THRESHOLD 从 0.6 改成 0.4、0.8 各跑一次，观察 near_dups 数量变化：

- 阈值越低，抓得越多，误杀越多；
- 阈值越高，漏网越多，误杀越少。

**正确调法**：准备 50 条人工标注的“真重复/真不重复”金样，画一条“阈值→误杀率/漏网率”曲线，选交叉点；不要凭感觉定 0.6。

### Step 6 打版本（5 分钟）

~~~bash
cd llm-demo
git add data-pipeline
git commit -m "dedup: exact sha256 + minhash lsh (shingle=5, perm=128, thr=0.6)"
git tag data-dedup-v0.1
~~~

> 生产提示：本脚本是单机教学版。分布式版把“精确哈希”换成分布式去重任务，把 MinHash 按“桶号”分区后并行比对；核心逻辑（归一化 → shingle → 签名 → 同桶精查）完全一致。

## 06 参数详解：去重参数怎么定

| 参数 | 参考区间 | 说明 |
|---|---|---|
| shingle 大小 | 字符级 4–6；词级 8–13 | 中文常用字符 5-gram；太短误杀高，太长漏网多 |
| num_perm | 128–256 | 签名越长越准，内存与计算越高；128 是常见起步 |
| LSH threshold | 0.5–0.8 | 抓转载用 0.6–0.7；只抓整篇盗用可 0.8+ |
| 归一化范围 | 空白/大小写/全半角/NFKC | 不要过度归一化（去掉所有标点会把不同文档变一样） |
| 文档最短长度 | 200–1000 字符 | 短文档 shingle 太少，MinHash 不稳定，先用精确去重 |
| 重复窗口占比 | 20%–50% | 段落级去重：重复片段占全文比例超过阈值才判重 |
| 桶数/band 数 | 由 threshold 推导（datasketch 自动） | 理解原理后可用 band/rows 手动控制 |

> 调参金标准：**用 50–100 条人工标注的“真重复/真不重复”样本做金样**，跑不同参数组合，选“误杀率与漏网率平衡点”。所有参数进 Git，配一次报告。

---

## 07 高频踩坑排查

**坑 1：没做归一化就精确哈希**
症状：同一篇文章的“空格/全角”版本没被去重。
解法：先 NFKC + 去空白 + 大小写归一，再做 sha256；但别把标点全删（会把不同文章误判相同）。

**坑 2：阈值设太低，正常文档被误杀**
症状：两篇只是共用常见句式（如“综上所述”）的文章被删。
解法：提高 threshold、加大 shingle、加“允许重复片段清单”；用金样集看误杀率。

**坑 3：shingle 太大，改写文漏网**
症状：转载文每句改几个字就躲过去。
解法：中文用字符 5-gram 起步，按漏网样本调小到 4；对“插入广告段”的转载，配段落级去重。

**坑 4：对短文档跑 MinHash**
症状：几十字的工单、问答，签名全是同一个值或极不稳定。
解法：短文档（<200 字符）只走精确去重；近似去重只用于长文档。

**坑 5：评测集污染没查**
症状：训练集与测试集来自同一批抓取，模型“高分低能”。
解法：评测集文档先入指纹库，训练文档与评测集相似度超阈值即剔除；每次新增评测题都要重跑一次防泄漏检查。

**坑 6：跨源重复漏网**
症状：每个来源内部去了重，但 A 源和 B 源互相转载没查。
解法：去重必须**跨全量语料**做（合并所有源后再跑 L1/L2），不能按来源分别去重。

**坑 7：只做文档级去重，拼接文漏网**
症状：内容农场把 10 篇文章各取一段拼成“新文”，文档级相似度不高。
解法：加段落/句子级去重：句子哈希去重 + 段落指纹统计重复占比。

---

## 08 进阶优化：去重体系的进化方向

**① 三级去重流水线**：文档级（sha256+MinHash）→ 段落级（段落指纹）→ 句子级（句子哈希），每级都输出“重复率报告”，用重复占比决定是否整篇丢弃。

**② 包含关系去重**：用“文档 A 的 n-gram 是否大多出现在文档 B”检测 A⊂B 的包含关系；短文档嵌长文档是最隐蔽的重复形态之一（工程实现常用后缀数组/SimHash 变体）。

**③ 分布式 MinHash**：TB 级数据把 shingle 哈希后按“桶号”分区，Spark/Ray 上并行插入与查询；先按 band 粗筛再做精细 Jaccard 校验。

**④ 评测防泄漏自动化**：把“训练集 vs 评测集”的相似度检查做成管道常驻任务，任何评测集更新自动触发；并记录泄漏样本供审计。

**⑤ 重复样本的二次利用**：别把去重删掉的数据直接扔掉——高频重复本身是“流行度”信号，可统计进第 09 章质量打分的特征；被评测泄漏剔除的样本单独归档，供分析模型“背题”行为。

---

## 09 本章核心总结（TOP3）

**TOP1**：去重 = 归一化精确去重（sha256）打底 + MinHash-LSH 近似去重拦截转载改写 + 段落/句子级去重抓拼接文；顺序不能反，粒度要分层。

**TOP2**：MinHash 把“文档相似度”变成“指纹重合度”，LSH 把“全库两两比对”变成“同桶精查”；核心参数是 shingle 大小、num_perm、threshold，用金样集调参而不是拍脑袋。

**TOP3**：评测防污染是与训练去重同等重要的一层：训练集必须与评测集做相似度筛查，否则你得到的“高分”只是模型的记忆力。

---

## 10 连载衔接

上一章（第 06 章）立起了过滤闸门；本章补上第四道闸门并给出可运行代码——filtered 层现在升级成了 dedup 后的 unique 层，语料“干净、该留、不重复”。

下一章处理合规红线：【第 08 章】敏感数据脱敏：PII 识别、脱敏策略与合规落地。数据里藏着的身份证号、手机号、内部地址，是训练数据里最不能带进模型的东西。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #数据去重 #MinHash #LSH #大模型训练数据 #工程实战**（Voice前沿 出品，欢迎收藏追更）

**系列目录（46 章，随连载持续更新）**

第 0 阶段·开篇引路
- 01 一条大模型生产线全程发生了什么
- 02 2 小时跑通最小闭环
- 03 预训练到底在练什么

第 1 阶段·预训练工程·数据核心层
- 04 数据管道从零构建
- 05 数据清洗规范与脏数据剔除
- 06 文本过滤实战
- 07 数据去重算法（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 08 章 敏感数据脱敏*