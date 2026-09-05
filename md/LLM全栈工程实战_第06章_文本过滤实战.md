# 【LLM全栈工程·第06章】文本过滤实战：低质、重复、有害内容的过滤体系

> 本篇为「LLM全栈工程」连载第 06 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 语料不是“洗”完就能用：四道过滤闸门一次讲清；② 低质、有害、语言混杂怎么拦？一套可调阈值的过滤体系；③ 从规则到分类器：文本过滤的工程化落地。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「预训练工程·数据核心层」第 4 篇：在清洗之后建立“语言/质量/安全/重复”四道过滤闸门 |
| 适用场景 | 个人学习（本地过滤实验）；小团队领域语料治理；企业预训练与增量预训练数据安全 |
| 核心知识点 | 过滤与清洗的边界；四道闸门；阈值调优；安全过滤的合规红线；过滤报告与审计 |
| 技术选型 | 启发式规则 vs 小分类器 vs 专用安全审核 vs 大模型过滤；语言识别与困惑度工具 |
| 分步实操 | 在第 05 章 clean 层后加 filter 层：注入边缘样本 → 多维打分过滤 → 报告审计 → 打版本 |
| 参数详解 | 语言占比、感叹号密度、有害词命中、低质信号、困惑度参考阈值 |
| 踩坑排查 | 关键词安全过滤漏 paraphrase、阈值误杀问答、语言过滤伤代码、只过滤不审计等 |
| 进阶优化 | 规则标注→分类器训练、安全红队测试、策略版本化、人工抽检闭环 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 07 章：数据去重算法 |

---

## 01 开篇导语

上一章我们把“明显脏”的数据洗掉了：乱码没了、广告删了、重复的“收到”清走了。但清洗只回答一个问题——**文本本身干不干净**。

这一章要回答另一个问题：**这段文本该不该留在训练集里？**

同样是干净文本，差别可以很大：
- “今天天气不错哈哈哈” vs “变压器故障诊断的五个步骤”；
- 一篇机器翻译得语无伦次的“伪中文” vs 一段地道的中文技术文档；
- 一篇正常科普 vs 一篇含赌博导流话术的“软文”。

如果只做清洗不过滤，模型会学到大量低质表达、有害内容与语言噪音——预训练数据质量的上限，就是这么被拉低的。

本章给你一套“四道闸门”的过滤体系：**语言过滤 → 质量过滤 → 安全过滤 → 重复过滤**，并用手把手的代码演示怎么落地第一版。去重算法因为内容足够深，单独放到第 07 章展开。

> 一句话记住本章：**清洗处理“脏”，过滤决定“留不留”；过滤的每一步都要可审计、可调参、可回滚。**

---

## 02 白话原理：过滤 = 一道一道的“闸门漏斗”

把语料想象成一条河，过滤就是河道上的四道闸门，每道闸门只做一件事：

| 闸门 | 拦什么 | 决策依据 | 误伤风险 |
|---|---|---|---|
| 第一道：语言过滤 | 目标语种之外的文本 | 语言识别（字符分布/语言模型） | 代码、专名、混合语种被误杀 |
| 第二道：质量过滤 | 低质、机器味、空泛文本 | 长度/符号/困惑度/分类器 | 短问答、口语化文本被误杀 |
| 第三道：安全过滤 | 违法有害内容 | 关键词+分类器+人工审核 | 正常讨论被误判（如医学、历史话题） |
| 第四道：重复过滤 | 近似重复文档 | 哈希/MinHash（第 07 章） | 合理引用、模板文档被误杀 |

关键设计原则：

1. **先便宜后昂贵**：规则/哈希先跑，分类器后跑，人工只处理少量疑难样本——TB 级数据经不起每一条都过重模型。
2. **宁可漏不可误杀（对质量闸门）**：质量过滤阈值宁松勿紧，漏掉低质数据只浪费一点算力，误杀好数据则永久损失信息；**安全闸门反过来**：宁可多拦再人工复核，也不能放行。
3. **每条决策留痕**：谁在哪个闸门被拦、命中什么信号，全部落盘——过滤策略也是要版本化、可回滚的代码。

> 管道形态：clean/（第 05 章产物）→ filter/（本章产物）→ dedup/（第 07 章）→ safe/（脱敏）→ final/。

## 03 四道闸门详解：信号与阈值

### 3.1 语言过滤（目标语种识别）

中文语料的第一道闸门是“是不是中文为主”。工程上最便宜的信号是 Unicode 字符分布：

| 信号 | 计算 | 用途 |
|---|---|---|
| CJK 字符占比 | 中文字符数 / 总字符数 | 判断是否中文为主 |
| 拉丁字符占比 | 英文/拼音占比 | 判断是否英文或中英混合 |
| 日文假名/韩文谚文占比 | 对应 Unicode 区间计数 | 过滤近邻语种 |
| 语言模型困惑度 | 用目标语言模型打分 | 精细判断（成本高，放后面） |

参考口径：目标中文语料，CJK 占比低于 30%–50% 的文档一般判为“非中文主文档”；但**代码、数学公式、产品名、引用文献**会让合法文档的 CJK 占比偏低——所以要加“代码/专名豁免”或改用整句级语言识别。

### 3.2 质量过滤（低质信号）

质量过滤不是“一票否决”，而是**多信号打分后按阈值分流**：

| 信号 | 直觉 | 参考区间 |
|---|---|---|
| 文本长度 | 太短通常是碎片 | <50–200 字（按语料类型） |
| 感叹号/问号密度 | 标题党、营销文偏高 | 感叹号占比 >1%–3% 警惕 |
| 大写占比（英文） | 喊话式文本 | ALL CAPS 行 >30% 警惕 |
| 标点符号密度 | 机器翻译常异常 | 标点率过高/过低都查 |
| 困惑度（Perplexity） | 语言模型打分，越低越“像人话” | 按语料分布取 90 分位为界（需自建基线） |
| 分类器置信度 | 训练过的“低质 vs 优质”分类器 | 阈值由 golden set 定 |

### 3.3 安全过滤（红线闸门）

安全过滤的范畴按内容政策划分，常见类别：

- 违法类：赌博导流、诈骗话术、毒品、违禁品交易；
- 有害类：暴力、色情、仇恨言论、自残引导；
- 隐私类：个人身份信息（第 08 章专门做脱敏，这里先标记）；
- 侵权类：明显盗版资源站文本（与 License 体系配合）。

**工程红线（重要）**：关键词命中只能作为“候选”，不能作为最终结论——有害内容大量使用变体、谐音、emoji 与间接表达。生产级方案是“关键词召回 → 分类器精排 → 高风险样本人工复核”，并保留拦截日志以备合规审计。训练语料里的有害内容还会反向教坏模型，这一道闸门不是“可选项”而是“必选项”。

### 3.4 重复过滤（预告）

质量过滤之后，还有大量“近似重复”：同一新闻的多家转载、同一文档的多次备份、拼接重复段。手段从精确哈希到 MinHash 近似去重，粒度从整篇到段落。**本章先留闸门位置，第 07 章给完整算法与代码。**

---

## 04 技术选型：过滤手段怎么组合

| 手段 | 成本 | 能力边界 | 用在哪个闸门 |
|---|---|---|---|
| 启发式规则/阈值 | 极低 | 只能抓“明显信号” | 语言、质量初筛 |
| 小分类器（FastText/线性模型） | 低 | 能泛化常见变体 | 质量、有害内容精排 |
| 专用安全审核服务/模型 | 中高 | 覆盖广、更新快 | 有害内容（生产推荐接入） |
| 大模型审核 | 高 | 理解强但慢且贵 | 小批量人工复核辅助 |
| 语言识别工具（fastText LID 类） | 低 | 成熟可靠 | 语言闸门 |
| 困惑度工具（KenLM 类） | 低中 | 衡量“像不像自然语言” | 质量闸门 |

组合结论（生产推荐）：

1. **语言闸门**：fastText LID 类工具或字符分布规则，先粗筛后精筛；
2. **质量闸门**：规则打分 + 小分类器 + 困惑度，三个信号加权；
3. **安全闸门**：关键词召回 + 专用分类器 + 人工抽检，高风险类别必须人工复核；
4. **全部闸门输出结构化日志**，阈值版本与规则版本一起进 manifest。

> 别在生产环境用“一个正则列表”当安全过滤的全部——那是把合规风险外包给一个 if 语句。

## 05 分步实操：给管道加上 filter 层（30 分钟）

继续沿用 llm-demo/data-pipeline。第 05 章的 clean/clean.jsonl 里都是“干净但未必该留”的文本；本节注入一批“边缘样本”，然后让过滤闸门做决定。

### Step 1 注入边缘样本（5 分钟）

保存 scripts/make_edge_sources.py：

~~~python
# data-pipeline/scripts/make_edge_sources.py —— 注入低质/有害/非目标语言边缘样本
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
sources = root / "sources"

edge_rows = [
    {"text": "注册送彩金 58 元，充 100 送 100，提现秒到账！快來玩！", "meta": {"channel": "spam_gambling"}},
    {"text": "无抵押秒批贷款，加 QQ 123456 立即办理，不看征信。", "meta": {"channel": "spam_loan"}},
    {"text": "有人欠钱不还，我要把他的身份证号和家庭住址发到网上让网友人肉他！", "meta": {"channel": "harmful_doxxing"}},
    {"text": "How to reset your password on the corporate portal. Follow these steps: open the portal, click forgot password, then check your email.", "meta": {"channel": "en_doc"}},
    {"text": "这是一个关于打印机驱动安装说明的说明的说明，可以帮助你帮助你帮助你完成驱动安装的安装的安装。", "meta": {"channel": "zh_low_quality"}},
    {"text": "打印机卡纸怎么办？先关机断电，打开前盖取出卡纸，检查无碎片后合盖开机测试。", "meta": {"channel": "qa_good"}},
]

with open(sources / "helpdesk_edge.jsonl", "w", encoding="utf-8") as f:
    for row in edge_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("edge rows:", len(edge_rows))
~~~

### Step 2 重新走“归一化 + 清洗”，让边缘样本进入 clean 层

~~~bash
python data-pipeline/scripts/make_edge_sources.py
python data-pipeline/scripts/normalize_sources.py
python data-pipeline/scripts/clean_sources.py
wc -l data-pipeline/clean/clean.jsonl
~~~

说明：这些边缘样本大多能通过第 05 章的“脏数据清洗”（它们不是乱码、不是纯符号），所以会留在 clean 层——这正是过滤闸门存在的意义。

### Step 3 写过滤规则包（10 分钟）

保存 scripts/filter_sources.py：

~~~python
# data-pipeline/scripts/filter_sources.py —— 四道闸门：语言/质量/安全（重复过滤见第 07 章）
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
clean_dir = root / "clean"
filtered_dir = root / "filtered"
manifests = root / "manifests"
filtered_dir.mkdir(parents=True, exist_ok=True)

PARAMS = {
    "min_zh_ratio": 0.5,      # 中文语料：CJK 占比低于该值判非中文
    "max_exclaim_ratio": 0.08, # 感叹号占比过高 = 标题党/营销信号
    "max_ngram_repeat": 3,     # 3-gram 重复超过该次数 = 复读机文本
    "min_chars": 6,
}

HARMFUL = {
    "gambling": ["赌博", "赌场", "彩金", "充值返利", "秒到账", "百家乐"],
    "loan_spam": ["贷款", "无抵押", "秒批", "不看征信"],
    "doxxing": ["人肉", "身份证号", "家庭住址", "曝光个人信息"],
}

def cjk_ratio(text):
    total = 0
    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            total += 1
    return total / max(1, len(text))

def exclaim_ratio(text):
    count = sum(1 for ch in text if ch in "！!？?")
    return count / max(1, len(text))

def max_ngram_repeat(text):
    grams = {}
    for i in range(len(text) - 2):
        gram = text[i:i + 3]
        grams[gram] = grams.get(gram, 0) + 1
    return max(grams.values()) if grams else 0

def harmful_hits(text):
    hits = []
    for category, words in HARMFUL.items():
        found = [w for w in words if w in text]
        if found:
            hits.append(category + ":" + ",".join(found))
    return hits

kept = []
dropped = []
report = {}

for line in open(clean_dir / "clean.jsonl", encoding="utf-8"):
    obj = json.loads(line)
    text = str(obj.get("text", ""))
    reasons = []

    if len(text.strip()) < PARAMS["min_chars"]:
        reasons.append("quality:too_short")

    hits = harmful_hits(text)
    if hits:
        reasons.append("harmful:" + "|".join(hits))

    if cjk_ratio(text) < PARAMS["min_zh_ratio"]:
        reasons.append("lang:low_zh_ratio")

    if exclaim_ratio(text) > PARAMS["max_exclaim_ratio"]:
        reasons.append("quality:high_exclaim")

    if len(text) < 300 and max_ngram_repeat(text) >= PARAMS["max_ngram_repeat"]:
        reasons.append("quality:ngram_repeat")

    if reasons:
        dropped.append({"doc_id": obj["doc_id"], "source": obj["source"], "reasons": reasons})
        for r in reasons:
            key = r.split(":")[0]
            report[key] = report.get(key, 0) + 1
    else:
        kept.append(obj)

with open(filtered_dir / "keep.jsonl", "w", encoding="utf-8") as f:
    for obj in kept:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
with open(filtered_dir / "drop_report.jsonl", "w", encoding="utf-8") as f:
    for item in dropped:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("kept:", len(kept), "dropped:", len(dropped))
print("report:", json.dumps(report, ensure_ascii=False))
~~~

### Step 4 跑过滤并检查误杀（5 分钟）

~~~bash
python data-pipeline/scripts/filter_sources.py
wc -l data-pipeline/clean/clean.jsonl data-pipeline/filtered/keep.jsonl
cat data-pipeline/filtered/drop_report.jsonl
~~~

预期结果：

- 赌博/贷款/人肉三条：harmful 命中，删除；
- 英文文档：lang:low_zh_ratio，删除；
- “的说明的说明/帮助你帮助你”复读文本：quality:ngram_repeat，删除；
- 正常问答、知识库规则、短 QA：全部保留。

**关键动作——误杀检查**：把 drop_report 与保留样本各抽 10 条人工看一遍，重点回答：有没有正常文档被误杀？如果有，调阈值而不是加黑名单（例如英文代码文档混入时，应该加“代码豁免”，而不是把 min_zh_ratio 调低到伤及中文）。

### Step 5 打版本（5 分钟）

~~~bash
cd llm-demo
git add data-pipeline
git commit -m "filter: gates v0.1 (lang/quality/harmful) + drop report"
git tag data-filter-v0.1
~~~

> 生产进阶：把 report 里的 dropped 样本按“harmful 类别”单独归档，交给安全团队复核；复核结果回流成安全分类器的训练数据（第 09 章打分体系会讲如何组织这些标注回流）。

## 06 参数详解：过滤阈值速查

| 参数 | 参考区间 | 适配场景 |
|---|---|---|
| min_zh_ratio | 0.3–0.7 | 纯中文语料 0.5–0.7；含代码/公式的语料降到 0.3–0.4 并加代码豁免 |
| max_exclaim_ratio | 0.02–0.10 | 营销/标题党多的源收紧；技术文档正常感叹号少，可更严 |
| max_ngram_repeat | 2–5 次 | 复读机/拼接文多的源收紧；口语语料（“对对对”）要放宽 |
| min_chars | 20–200 | 问答类 10–20；网页正文 50+；代码语料按行处理不走文本规则 |
| 困惑度分位阈值 | 85–95 分位 | 先跑 1% 样本看分布再定，不拍脑袋 |
| 安全词命中数 | 1 个即“候选拦截” | 关键词只召回，分类器/人工复核后终判 |
| 语言识别置信度 | 0.7–0.9 | fastText LID 类工具的置信度门槛 |

调参方法：先抽 1% 样本跑一遍，统计每个信号的分布，找出“好数据 95% 都满足、坏数据 50% 以上违反”的分界点，再人工验证 50–100 条。**所有阈值存进 FILTER_PARAMS 并进 Git，禁止散落在脚本魔法数字里。**

---

## 07 高频踩坑排查

**坑 1：安全过滤只用关键词列表**
症状：加了 500 个敏感词，变体话术照样穿透（“戒赌上岸交流群”这类谐音/隐语）。
解法：关键词只做召回，接分类器精排；高危类别人工复核；把拦截样本持续回流训练。**把安全过滤当成产品做，不是写个列表。**

**坑 2：质量过滤误杀短问答**
症状：语料里大量高质量短问答（“发票抬头怎么改？”），被 min_chars=200 全删。
解法：按 source_type 分桶设阈值；问答/工单类文档用 10–20 字下限，网页正文才用 200 字下限。

**坑 3：语言过滤杀死代码与中英混合文档**
症状：代码仓库文档 CJK 占比 30%，被当“非中文”删掉。
解法：先识别文档类型（代码/文档），代码与混合文档走豁免通道；或改用行级语言判断而非整篇占比。

**坑 4：过滤不可审计**
症状：数据少了一半，但没人说得清是哪条规则删的、删了多少。
解法：drop_report 记录 doc_id/source/reasons 全字段，规则版本进 manifest；任何过滤变更先出对比报告。

**坑 5：只调阈值不加豁免，按下葫芦浮起瓢**
症状：为保住 A 类好数据放宽阈值，B 类垃圾跟着漏进来。
解法：正确姿势是加“豁免规则”（如代码块豁免、引用豁免），而不是全局放宽阈值。

**坑 6：把过滤当一次性任务**
症状：新源接入后不过滤直接混入，数据质量悄悄下滑。
解法：过滤是管道常驻闸门，新源必须先过“体检+过滤”才能进 final；用每批次质量报告盯趋势。

---

## 08 进阶优化：过滤体系的进化路径

**① 规则 → 分类器自助进化**：把 drop_report 与人工复核结果整理成标注集（keep=0/drop=1 + 类别），训练 FastText 类分类器；规则负责召回，分类器负责泛化，人工只审边界样本。

**② 安全红队测试**：每个季度用“绕过测试集”（谐音、隐语、emoji 替换、多语混杂）攻击自己的安全过滤，漏检率纳入质量指标——安全过滤需要对抗性思维，不是静态规则能覆盖的。

**③ 质量校准与分层**：不过滤“一刀切”，而是给文档打质量分并分层：高质层直接进训练、中质层进候选、低质层丢弃或降权（第 09 章数据质量打分体系正式展开）。

**④ 阈值版本化 + 自动回归**：每个过滤参数组合打一个版本号，配上“金样集”（人工标注的 1000 条），每次调参自动跑金样集算误杀率/漏网率，防止越调越歪。

**⑤ 与合规体系联动**：安全过滤的拦截日志、人工复核记录、策略版本，都是合规审计的证据链；建议按“策略即代码 + 日志即证据”来建设。

---

## 09 本章核心总结（TOP3）

**TOP1**：过滤解决“留不留”，清洗解决“脏不脏”；标准顺序是语言 → 质量 → 安全 → 重复四道闸门，先便宜后昂贵、先召回后精排。

**TOP2**：质量闸门宁松勿紧（误杀好数据不可逆），安全闸门宁紧勿松（关键词召回 + 分类器精排 + 人工复核），两类闸门的调参哲学完全相反。

**TOP3**：过滤必须可审计：drop_report 记录每条删除决策，阈值与规则版本进 Git/manifest；每次调参配“误杀率/漏网率”金样回归，把过滤从“玄学删数据”变成工程。

---

## 10 连载衔接

上一章（第 05 章）清掉了“脏”；本章立起四道过滤闸门，把“低质、有害、非目标语言”挡在训练集之外——clean 层现在升级成了 filtered 层。

下一章处理过滤漏斗里最“重”的一环：【第 07 章】数据去重算法：从精确去重到 MinHash 近似去重。你会发现，语料里最大的隐性浪费，是那些你以为不重复的“重复”。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #文本过滤 #数据安全 #内容审核 #大模型训练数据 #工程实战**（Voice前沿 出品，欢迎收藏追更）

**系列目录（46 章，随连载持续更新）**

第 0 阶段·开篇引路
- 01 一条大模型生产线全程发生了什么
- 02 2 小时跑通最小闭环
- 03 预训练到底在练什么

第 1 阶段·预训练工程·数据核心层
- 04 数据管道从零构建
- 05 数据清洗规范与脏数据剔除
- 06 文本过滤实战（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 07 章 数据去重算法*