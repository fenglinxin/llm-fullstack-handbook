# 【LLM全栈工程·第05章】数据清洗规范与脏数据剔除：先认识“脏”长什么样

> 本篇为「LLM全栈工程」连载第 05 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 大模型语料里 60% 是垃圾？数据清洗到底在洗什么；② 从网页到训练文本：一套能上线的清洗规则库怎么建；③ 别等训完才发现数据脏：清洗规则、阈值与前后对比一次讲清。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「预训练工程·数据核心层」第 3 篇：建立清洗规范，把 staging 层原始数据变成干净文本 |
| 适用场景 | 个人学习（本地规则清洗）；小团队领域语料；企业预训练/增量预训练数据生产线 |
| 核心知识点 | 脏数据四分类；规则库设计（名称/版本/动作/理由）；清洗前后对比；规则版本化管理 |
| 技术选型 | 规则清洗 vs 分类器清洗 vs LLM 清洗；HTML 抽取工具（trafilatura 类）与文本清洗自研的边界 |
| 分步实操 | 在第 04 章管道上新增 clean 层：造脏样例 → 清洗规则包 → 跑清洗 → 对比审计 → 打版本 |
| 参数详解 | 行长度、符号占比、重复字符、乱码标记、样板文本等阈值与适配场景 |
| 踩坑排查 | 正则误删、编码乱码、清洗顺序、过度清洗、无对比就上线、规则不可复现 |
| 进阶优化 | 清洗报告、人工抽检 golden set、分类器承接规则、LLM 清洗适用边界 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 06 章：文本过滤实战（低质/重复/有害内容过滤体系） |

---

## 01 开篇导语

上一章我们把多源数据搬进了 staging 层——它们是“原样”的：网页 HTML、工单导出、聊天记录……**离能训练还差一次大规模“洗澡”。**

真实语料有多脏？一个网页快照里，正文可能只占 30%，剩下是导航菜单、广告、Cookie 弹窗、评论区的“666”、还有乱码和截断；一份爬来的文档可能带着 Markdown 语法残留；一份内部聊天记录可能一半是“收到”“好的”。

如果把这些直接喂给模型，会发生两件事：一是模型花大量参数学习“下一页”“登录注册”这些垃圾规律；二是真正有价值的知识被噪声淹没，数据上限被拉低。

本章回答三个问题：

1. 脏数据到底分几类、长什么样；
2. 清洗规则怎么设计才不会“杀敌一千自损八百”；
3. 怎么用代码跑一次可审计、可复现的清洗。

交付物：接在上一章管道上的 clean 层——一个带版本号的清洗规则包、一份清洗前后对比报告、一个 Git 标签。

---

## 02 白话原理：清洗不是“删东西”，是“分类处理”

清洗的本质是**对每一条文本做分类决策**，常见动作只有四种：保留、删除、截断、改写。所有清洗规则，最终都落在这四个动作上。

先认识脏数据的四个家族：

**① 结构噪音（网页壳子）**
导航菜单、页头页脚、广告、Cookie 横幅、上一篇/下一篇、评论区“加载更多”……特征：和正文无关、跨页面高度重复。
处理：HTML 解析后抽取正文（用 trafilatura 类工具），再按“样板文本”规则删除。

**② 格式与编码噪音**
乱码（锟斤拷、â€œ 这类 mojibake）、全角半角混乱、零宽字符、控制字符、多余空白、HTML 实体残留（&amp;）。
处理：统一转码、字符归一、去控制字符；ftfy 类库修 mojibake。

**③ 语义垃圾**
纯符号行、无意义重复（哈哈哈哈哈哈、666666）、广告话术（“点击领取”“加微信”）、模板占位（“此处插入图片”“TODO”）、内容农场拼接文。
处理：规则 + 分类器，按比例阈值与模式库识别。

**④ 内容残缺**
截断文本（文章在句子中间断掉）、只有标题没有正文、只有 URL 没有描述、重复段落拼接。
处理：长度/结尾完整性检查、与源文件校验、启发式规则。

> 一句话：**清洗 = 对每条文本回答“它是不是一段完整、干净、有用的自然语言”**。回答不了就交给规则/分类器打分，本章先解决“明显脏”，第 09 章再解决“质量高低”。

## 03 脏数据图鉴与规则库设计

### 3.1 十种最高频脏数据（工业界见闻）

| # | 类型 | 典型样例 | 判断特征 |
|---|---|---|---|
| 1 | 网页导航样板 | “首页 产品 关于我们 联系我们” | 与正文无关、跨页重复 |
| 2 | 广告/引流 | “加微信 xxx，领取免费资料” | 高营销词密度 |
| 3 | 乱码 | “锟斤拷锟斤拷”“â€™” | mojibake 特征序列 |
| 4 | 纯符号/表情行 | “！！！！！！！” “😀😀😀” | 符号占比极高 |
| 5 | 无意义重复 | “哈哈哈哈哈哈哈哈” | 单字重复率极高 |
| 6 | 模板占位 | “【图片】”“此处插入视频”“TODO” | 模板词表命中 |
| 7 | 截断文本 | “本文介绍了大模型的……”（断在句中） | 结尾无标点、长度异常 |
| 8 | HTML/代码残留 | “div 标签”“&nbsp;”“python 代码围栏” | 标签/转义/围栏残留 |
| 9 | URL 裸堆 | “www.xxx.com http://yyy.cn …” | URL 密度高、无语义 |
| 10 | 版权/免责声明墙 | “版权所有，未经许可不得转载……” | 高重复声明模板 |

### 3.2 规则库设计：让每条规则“有名有姓有理由”

清洗规则最忌写成“一串神秘正则”。每条规则至少包含六个字段：

| 字段 | 含义 | 示例 |
|---|---|---|
| rule_id | 规则唯一 ID | clean_001_symbol_line |
| version | 规则版本 | v1 |
| category | 属于四类中的哪类 | 格式噪音 |
| description | 人类可读说明 | 删除符号占比>80% 的行 |
| action | 保留/删除/截断/改写 | delete_line |
| params | 参数 | symbol_ratio=0.8 |

规则输出必须写成**可审计日志**：某条 doc 被哪条规则命中、命中了什么。这样清洗过程可复现、可调参，而不是黑盒。

### 3.3 规则执行顺序（先粗后细）

推荐顺序：**解码归一 → 结构抽取 → 行级规则 → 文档级规则 → 完整性检查**。顺序错了会互相干扰（例如先按标点删行，会把 HTML 标签拆散后的正文也误删）。

~~~text
1. 统一 UTF-8，去控制字符与零宽字符
2. HTML/富文本 -> 正文抽取（trafilatura 类工具）
3. 行级清洗：删导航/广告/纯符号/乱码/重复行
4. 文档级清洗：去样板头尾、去重复段落、去版权墙
5. 完整性检查：截断检测、最小长度、语言一致性
6. 输出 clean 层 + 清洗日志
~~~

---

## 04 技术选型：规则、分类器还是 LLM？

| 方案 | 优点 | 缺点 | 适用 |
|---|---|---|---|
| 规则清洗（本章主推） | 快、可控、可解释、零成本 | 只能处理“已知的脏” | 第一步必做，处理 80% 明显脏数据 |
| 小分类器（FastText 类） | 能泛化到没见过的话术 | 需要标注数据与训练维护 | 规则覆盖不了的“语义垃圾”（第 06 章展开） |
| LLM 清洗 | 理解力最强 | 慢、贵、可能改写原文 | 只用于小批量精修/特殊语料，不用于 TB 级流水线 |
| trafilatura 等抽取工具 | 网页正文抽取开箱即用 | 只解决 HTML 抽取，不解决语义垃圾 | 网页类语料必经步骤 |

结论：**清洗的顺序永远是“规则打底 → 分类器补漏 → LLM 精修”**。一上来就用 LLM 洗 TB 级数据，既贵又慢，还会引入改写风险。

## 05 分步实操：给管道装上 clean 层（30 分钟）

继续沿用 llm-demo/data-pipeline。第 04 章的 staging/unified.jsonl 是“干净源的归一化结果”；本节再注入一批“脏源”，然后写清洗规则包把它们拦下来。

### Step 1 造一批脏样例（5 分钟）

保存 scripts/make_dirty_sources.py：

~~~python
# data-pipeline/scripts/make_dirty_sources.py —— 生成脏数据源（模拟网页/聊天垃圾）
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
sources = root / "sources"

dirty_rows = [
    {"text": "首页 关于我们 产品中心 联系我们 登录 注册", "meta": {"channel": "web_nav"}},
    {"text": "加微信 ithelper888 领取免费打印机驱动包！！！", "meta": {"channel": "web_ad"}},
    {"text": "锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷", "meta": {"channel": "mojibake"}},
    {"text": "哈哈哈哈哈哈哈哈哈哈哈哈哈哈", "meta": {"channel": "chat_noise"}},
    {"text": "😀😀😀😀😀😀😀😀😀😀😀", "meta": {"channel": "chat_emoji"}},
    {"text": "http://spam.example.com 点我 http://spam2.example.com 有惊喜 http://spam3.example.com", "meta": {"channel": "url_spam"}},
    {"text": "收到收到收到收到收到收到收到收到收到收到收到收到", "meta": {"channel": "chat_ack"}},
    {"text": "规则：员工 WiFi 名为 Corp-WiFi，使用域账号登录，忘记密码请访问自助平台重置。", "meta": {"channel": "kb_good"}},
]

with open(sources / "helpdesk_dirty.jsonl", "w", encoding="utf-8") as f:
    for row in dirty_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("dirty source rows:", len(dirty_rows))
~~~

### Step 2 重新归一化（把脏数据并入 staging）

~~~bash
python data-pipeline/scripts/make_dirty_sources.py
python data-pipeline/scripts/normalize_sources.py
python data-pipeline/scripts/audit_manifest.py
~~~

此时 staging/unified.jsonl 里既有第 04 章的好数据，也有刚注入的 8 条脏数据。下一步让清洗规则把它们挑出来。

### Step 3 写清洗规则包（10 分钟）

保存 scripts/clean_sources.py：

~~~python
# data-pipeline/scripts/clean_sources.py —— 规则清洗：staging -> clean
import json
import pathlib
import re

root = pathlib.Path(__file__).resolve().parent.parent
staging = root / "staging"
clean_dir = root / "clean"
manifests = root / "manifests"
clean_dir.mkdir(parents=True, exist_ok=True)

RULES = {
    "min_chars": 6,
    "max_repeat_ratio": 0.5,
    "max_urls": 2,
    "nav_keywords": ["首页", "登录", "注册", "联系我们", "版权所有", "未经许可"],
    "ad_keywords": ["加微信", "免费领取", "点击领取"],
}

def hit_repeat_ratio(text):
    prev = ""
    run = 0
    best = 0
    for ch in text:
        if ch == prev:
            run += 1
        else:
            run = 1
            prev = ch
        if run > best:
            best = run
    return best / max(1, len(text))

def is_symbol_line(text):
    symbol_chars = "！!？?😀😂～~。."
    return len(text) > 0 and all(ch in symbol_chars or ch.isspace() for ch in text)

report = {}
kept = []
dropped = []

for line in open(staging / "unified.jsonl", encoding="utf-8"):
    obj = json.loads(line)
    text = str(obj.get("text", ""))
    reasons = []

    if len(text.strip()) < RULES["min_chars"]:
        reasons.append("too_short")
    if "锟斤拷" in text or "â€" in text or "Ã" in text:
        reasons.append("mojibake")
    if is_symbol_line(text.strip()):
        reasons.append("symbol_line")
    if len(text) >= 6 and hit_repeat_ratio(text) >= RULES["max_repeat_ratio"]:
        reasons.append("repeat_noise")
    url_count = len(re.findall(r"https?://\S+", text))
    if url_count >= RULES["max_urls"]:
        reasons.append("url_dump")
    kw = [k for k in RULES["nav_keywords"] + RULES["ad_keywords"] if k in text]
    if kw and len(text) <= 60:
        reasons.append("nav_ad:" + ",".join(kw))

    if reasons:
        dropped.append({"doc_id": obj["doc_id"], "source": obj["source"], "reasons": reasons})
        for r in reasons:
            key = r.split(":")[0]
            report[key] = report.get(key, 0) + 1
    else:
        kept.append(obj)

with open(clean_dir / "clean.jsonl", "w", encoding="utf-8") as f:
    for obj in kept:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
with open(manifests / "clean_report.jsonl", "w", encoding="utf-8") as f:
    for item in dropped:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("kept:", len(kept), "dropped:", len(dropped))
print("report:", json.dumps(report, ensure_ascii=False))
~~~

### Step 4 跑清洗并对比（5 分钟）

~~~bash
python data-pipeline/scripts/clean_sources.py
wc -l data-pipeline/staging/unified.jsonl data-pipeline/clean/clean.jsonl
head -n 3 data-pipeline/manifests/clean_report.jsonl
~~~

预期看到：staging 行数 > clean 行数，report 里有 mojibake、symbol_line、repeat_noise、url_dump、nav_ad 等命中的计数，且第 04 章的问答/知识库数据全部保留。

### Step 5 把规则版本记进账本并打标签（5 分钟）

~~~bash
cd llm-demo
git add data-pipeline
git commit -m "clean: rules v0.1 (min_chars/repeat/url/nav_ad/mojibake)"
git tag data-clean-v0.1
~~~

> 生产环境里，clean_report.jsonl 就是“清洗日志”：谁被删了、为什么被删，全量可查。规则调参时，只改 RULES 参数和规则函数，然后重跑并对比前后 report——这就是“规则版本化”的最小实现。

## 06 参数详解：清洗阈值怎么定

| 参数 | 参考区间 | 说明 |
|---|---|---|
| min_chars | 20–200（按语料类型） | 网页正文可设 50+；工单/问答可设 10–20；设太高会误杀短问答 |
| symbol_ratio | 0.5–0.8 | 行内符号占比高于阈值视为纯符号行；表情包多的聊天语料需单独调 |
| max_repeat_ratio | 0.3–0.6 | 单字符最大连续重复占比；方言/口语语料要放宽 |
| max_urls_per_doc | 1–3 | URL 密度过高视为链接农场；代码语料例外（代码里 URL 正常） |
| nav/ad 词命中数 | 1–3 个词 + 文本长度上限 | 短文本命中即删，长文本还要看位置（头尾样板） |
| min_doc_len（文档级） | 200–2000 字符 | 低于阈值多为碎片/标题页；也要结合语料类型 |
| 截断检测 | 结尾标点率 <30% 且长度异常 | 中文语料结尾常见句号/感叹号/问号，缺失率高要警惕 |

调参铁律：**阈值必须配“抽样人工看 50–100 条”再定**——只看统计不看样例，一定会出现“误杀率”和“漏网率”的失衡。把每次调参的 20 条误杀样例存成回归集，防止改一个规则破坏另一个。

---

## 07 高频踩坑排查

**坑 1：正则误删正文**
症状：清洗后领域术语、代码、专有名词大量消失。
原因：规则太宽（比如“含‘注册’就删”），或没看抽样直接全量跑。
解法：每条规则先在小样本上跑，输出命中样例人工看；关键词规则要叠加“文本长度/位置”条件，并保留误杀回归集。

**坑 2：乱码没在入口解决，越传越脏**
症状：staging 里已经乱码，清洗时靠正则救不回来。
解法：在归一化入口统一 UTF-8 + mojibake 检测（第 04 章 normalize 阶段就该做），乱码率超过阈值直接拒收该源。

**坑 3：清洗顺序错误互相干扰**
症状：先删了“短行”，把 HTML 标签拆散后的正文片段全误删了。
解法：严格按“解码归一 → 结构抽取 → 行级 → 文档级 → 完整性”顺序执行，规则依赖关系写进规则库文档。

**坑 4：对代码语料用文本清洗规则**
症状：代码里的 URL、短行、符号被当垃圾删掉。
解法：按语料类型分流——代码语料走“只清注释噪音/密钥”，文本清洗规则不直接套用；管道里给每条数据打 source_type 标签。

**坑 5：清洗不可复现**
症状：两次清洗结果不一样，或没人知道规则改了什么。
解法：规则全部代码化+版本号，清洗日志落盘（clean_report.jsonl），规则版本写进 manifest；禁止在 Notebook 里手工“再删一遍”。

**坑 6：只删不查，误杀率没人知道**
症状：数据量从 10 亿降到 3 亿，团队说“洗得很干净”，但不知道误杀了多少好数据。
解法：每次清洗输出三层报告——删除统计、命中样例、人工抽检误杀率；误杀率超过 1%–5%（按场景）就要回滚规则。

---

## 08 进阶优化：从规则清洗走向工程化清洗

**① 清洗报告自动化**：把 clean_report 汇总成 HTML/看板（按规则、来源、时间三个维度），谁改规则都要先看报告再上线。

**② 建立误杀回归集**：每次人工发现的“被误删的好数据”存成回归集，规则升级后自动重放，防止旧问题复发——这是清洗规则的质量测试。

**③ 规则 → 分类器接力**：当“广告话术”“内容农场”的变体多到规则写不完时，用已标注的清洗结果训练一个小分类器（FastText 类），把规则的命中样本当正样本、保留样本当负样本（第 06 章详解）。

**④ LLM 清洗的正确位置**：只在“规则+分类器都判不准”的小批量特殊语料上用 LLM（如古文、方言、专业表格转文本），并强制“只输出清洗后的原文，禁止改写”，用 diff 审计。

**⑤ 与质量打分衔接**：清洗负责“删明显脏”，第 09 章的质量打分负责“给活下来的数据排座次”——两者共用同一份 doc_id 与 manifest，数据血缘不断。

---

## 09 本章核心总结（TOP3）

**TOP1**：脏数据分四类——结构噪音、格式/编码噪音、语义垃圾、内容残缺；清洗动作只有保留/删除/截断/改写四种，所有规则都要落在这四个动作上。

**TOP2**：清洗规则必须“有名有姓有版本”：rule_id、category、action、params、version 六件套，输出清洗日志，规则版本写进 manifest——不可复现的清洗等于没洗。

**TOP3**：技术路线是“规则打底 → 分类器补漏 → LLM 精修”；每次清洗都要出“删除统计 + 命中样例 + 人工抽检误杀率”三层报告，阈值靠抽样定，不靠拍脑袋。

---

## 10 连载衔接

上一章（第 04 章）建好了分层管道；本章给它装上了 clean 层和第一版清洗规则包——明显脏的数据已经被记录在案、挡在训练之外。

下一章把“过滤”单独拎出来放大：【第 06 章】文本过滤实战：低质、重复、有害内容的过滤体系。清洗解决“脏”，过滤解决“该不该留”，两者配合才是完整的第一道数据闸门。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #数据清洗 #脏数据 #数据管道 #大模型训练数据 #工程实战**（Voice前沿 出品，欢迎收藏追更）

**系列目录（46 章，随连载持续更新）**

第 0 阶段·开篇引路
- 01 一条大模型生产线全程发生了什么
- 02 2 小时跑通最小闭环
- 03 预训练到底在练什么

第 1 阶段·预训练工程·数据核心层
- 04 数据管道从零构建
- 05 数据清洗规范与脏数据剔除（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 06 章 文本过滤实战*