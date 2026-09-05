# 【LLM全栈工程·第04章】数据管道从零构建：原始采集、多源整合与目录设计

> 本篇为「LLM全栈工程」连载第 04 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 大模型的第一桶金是数据：从 0 搭一条能审计、能复现的数据管道；② 原始语料怎么变成训练集？一张目录结构图讲清数据管道；③ 别再用“下载完就开训”了：数据管道的分层、清单与血缘设计。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「预训练工程·数据核心层」第 2 篇：解决“原始语料从哪来、怎么整合、怎么管版本”的地基问题 |
| 适用场景 | 个人学习（本地小管道）；小团队做领域语料；企业搭建可审计的数据生产线 |
| 核心知识点 | 管道五层模型；多源整合与格式归一；目录分层规范；manifest 清单与血缘；版本化与可复现 |
| 技术选型 | 数据源接入方式、单机/分布式处理引擎、编排框架（Airflow/Dagster/Prefect/自研）对比 |
| 分步实操 | 建目录 → 造样例源 → 归一化脚本 → manifest 审计 → Git 打版本，跑通最小数据管道 |
| 参数详解 | 分片大小、压缩格式、目录分区、manifest 必填字段、快照与保留策略 |
| 踩坑排查 | 无版本无血缘、格式混乱、来源与 License 不清、内存爆炸、目录过深、清洗结果不可复现 |
| 进阶优化 | 增量采集、流批一体、质量闸门前置、预算抽样、数据目录与血缘系统 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 05 章：数据清洗规范与脏数据剔除 |

---

## 01 开篇导语

上一章讲清了预训练的原理——模型在超大语料上玩“文字接龙”。那这些语料到底从哪来？怎么从“互联网上的一堆网页”变成“可以开训的一堆文件”？

很多团队死在这一步：数据下载了十几个 TB，放在一个叫 data_final 的文件夹里，三个月后没人知道哪个版本训了哪个模型、哪些文件重复、哪些来自没授权来源。**模型没训出来，先欠了一屁股数据债。**

这一章不聊清洗算法（那是第 05–10 章的事），只解决三个地基问题：

1. **原始数据从哪采、怎么采**；
2. **多源数据怎么整合成统一格式**；
3. **目录和版本怎么设计，才能可审计、可复现**。

交付物：一个 30 分钟能跑通的最小数据管道——带分层目录、清单文件、归一化脚本和版本标签。它是主线工程“领域大模型生产流水线”的数据起点，也是后面每一章清洗/去重/打分代码的“挂载点”。

---

## 02 白话原理：数据管道 = 搬运 + 质检 + 记账

把数据管道想成一条工厂流水线，原料是各种来源的原始文本，成品是“可以开训的训练集文件”。流水线上站着三类角色：

**① 搬运工（采集与整合）**
把网页、代码、文档、问答、日志从各个源头搬回来，统一成同一种包装（JSONL），并记下每件货的“进货单”（来源、时间、许可）。

**② 质检员（清洗与过滤）**
把脏的、重复的、敏感的、低质的货挑出来（本章只留质检位，具体算法后面章节逐个上）。

**③ 记账员（清单与版本）**
每批货进出都记账：这一版数据由哪些源文件、什么清洗规则、哪个 commit 产生。**没有记账的数据，等于没做过实验。**

管道设计的核心原则只有一条：**每个环节输入输出都必须是“文件 + 清单”，而不是“内存里的变量”**——因为只有落盘的东西才能断点续跑、并行处理、回溯版本。

> 一句话记住本章：**先让数据“有名字、有来源、有版本、可重跑”，再谈清洗算法。**

## 03 原始数据从哪来：常见来源与接入方式

### 3.1 常见语料来源（预训练/增量预训练视角）

| 来源类型 | 代表 | 内容特点 | 接入注意 |
|---|---|---|---|
| 网页快照 | Common Crawl、各语言网页抓取 | 量大、噪声大、多语言 | 模板/导航/广告多，需第 05–06 章清洗过滤 |
| 代码 | GitHub 公开仓库快照（注意许可筛选） | 代码+注释+README | 只保留合规 License，排除密钥与依赖噪音 |
| 学术与书籍 | arXiv、开放书籍项目、出版社授权语料 | 高质量长文 | 许可与版权是红线 |
| 社区问答 | 开放问答/论坛数据 | 问答结构好 | 质量参差，需过滤与去重 |
| 百科与结构化 | Wikipedia 类开放百科 | 结构规范 | 抽取正文时别把模板/编辑注释带进来 |
| 企业内部数据 | 工单、客服记录、文档库、日志 | 领域价值最高 | 脱敏先行（第 08 章），注意合规审批 |

> 真实性红线：**一切来源先查 License 与使用条款**，网页抓取要遵守 robots 与当地法规。训练数据侵权是真实的法律风险，不是“技术问题”。

### 3.2 三种接入模式

| 模式 | 做法 | 适合 | 缺点 |
|---|---|---|---|
| 全量快照 | 周期性把源头整个拉下来（如每月网页快照） | 网页、百科等公开大源 | 存储大、周期长 |
| 增量同步 | 只拉上次以来的新增/变更 | 代码仓库、内部文档 | 需要记录游标/断点 |
| 订阅/推送 | 源头主动推送（API webhook、消息队列） | 内部业务数据 | 需要消费端稳定 |

工业实践通常是“**全量打底 + 增量更新**”：先拉一次全量快照建基线，之后每日增量同步，按天分区落盘。

---

## 04 管道架构与目录设计：先定规矩再写代码

### 4.1 数据分层（五层模型）

| 层 | 目录 | 内容 | 谁消费 |
|---|---|---|---|
| 原始层 | sources/ | 源头原样落盘，只读、不改 | 只有采集任务能写 |
| 标准层 | staging/ | 统一成 JSONL、补 doc_id/来源字段 | 清洗任务 |
| 清洗层 | clean/ | 清洗/过滤/去重/脱敏后 | 打分任务 |
| 成品层 | final/ | 按配比混合、分好 train/val 的训练集 | 训练任务 |
| 元数据层 | manifests/ | 每层输出的清单、统计、规则版本 | 人/审计/血缘 |

**每层只依赖上一层，禁止跳层读写**——这是管道不乱的第一条规矩。

### 4.2 目录分区规范

按“数据集/来源/日期”三层分区，加 _SUCCESS 完成标记：

~~~text
final/helpdesk-v1/
├── source=helpdesk_qa/
│   ├── date=2026-09-05/
│   │   ├── part-00000.jsonl.gz
│   │   ├── part-00001.jsonl.gz
│   │   └── _SUCCESS
│   └── date=2026-09-06/
├── source=helpdesk_kb/
└── manifest.json
~~~

好处：按分区增量处理、坏分区单独重跑、人眼一看就懂数据血缘。

### 4.3 清单（Manifest）：数据管的“账本”

每个成品数据集必须带一个 manifest.json，至少包含：

~~~json
{
  "dataset": "helpdesk-v1",
  "version": "2026-09-05.1",
  "created_by": "pipeline/commit=abc1234",
  "inputs": [
    {"source": "helpdesk_qa", "path": "sources/helpdesk_qa.jsonl", "sha256": "…"},
    {"source": "helpdesk_kb", "path": "sources/helpdesk_kb.jsonl", "sha256": "…"}
  ],
  "rules": {"clean_rules_version": "v0.1", "filter_rules_version": "v0.1"},
  "stats": {"rows": 120, "chars": 88421, "dupes_removed": 3},
  "schema_version": "1"
}
~~~

有了 manifest，任何一次训练都能回答三个问题：**这版数据用了哪些源？跑了哪些规则？是谁在什么代码版本下生成的？** 这就是“可复现训练”的起点。

### 4.4 文件格式与大小

- 行格式：统一 JSONL（每行一个 JSON 对象），不要混 CSV/XML/嵌套 JSON；
- 压缩：大文件用 gzip 或 zstd；小数据集可不压缩；
- 分片：单文件建议控制在 128MB–1GB（压缩后），太小文件数爆炸、太大并行度差；
- 字符编码：全部 UTF-8，入库即转码，禁止“中文变乱码”的隐形事故。

## 05 技术选型：处理引擎与编排框架怎么选

### 5.1 处理引擎

| 规模 | 推荐 | 理由 | 别选 |
|---|---|---|---|
| 本地学习/小团队（GB 级） | Python + polars/pandas + DuckDB | 上手快、够用、易调试 | 为 10GB 数据上 Spark |
| 百 GB–TB 级 | Spark / Ray 分布式处理，或直接使用 Data-Juicer 类 LLM 数据工具 | 并行吞吐、生态成熟 | 单机 pandas 硬扛 |
| PB 级 | 云数据湖 + 云托管处理 | 存储计算分离、弹性 | 自研分布式框架 |

### 5.2 编排框架（谁按顺序跑任务、谁负责重试）

| 方案 | 特点 | 适合 |
|---|---|---|
| 脚本 + cron / CI | 零依赖 | 管道 <10 个任务、单人维护 |
| Apache Airflow | 生态最大、调度成熟 | 中大型团队、复杂依赖 |
| Dagster / Prefect | 数据感知更强、开发体验好 | 数据团队自建管道 |
| 云工作流 | 托管、免运维 | 云上数据湖 |

结论：**起步阶段不要上编排框架**——先用“目录 + 脚本 + Git 标签”把数据规整清楚，任务超过 10 个、需要定时与重试时再上 Airflow/Dagster。工具不是越多越好，数据可复现才是目的。

---

## 06 分步实操：30 分钟搭出最小数据管道

我们把管道建在第 02 章的 llm-demo 工程里，命名为 data-pipeline，后续章节的清洗/去重代码都挂在这里。

### Step 1 建目录（2 分钟）

~~~bash
cd llm-demo
mkdir -p data-pipeline/{sources,staging,clean,manifests,scripts}
~~~

### Step 2 造三个“多源”样例（5 分钟）

保存 scripts/make_sources.py：

~~~python
# data-pipeline/scripts/make_sources.py —— 生成三个模拟数据源
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
sources = root / "sources"
sources.mkdir(parents=True, exist_ok=True)

qa = [
    {"text": "问题：打印机连不上怎么办？\n答案：检查电源与网络指示灯；重启打印机；删除并重新添加 HP LaserJet M405；仍失败请提交工单。", "meta": {"channel": "qa"}},
    {"text": "问题：VPN 账号被锁定怎么办？\n答案：等待 30 分钟自动解锁，或联系 IT 热线。", "meta": {"channel": "qa"}},
]
kb = [
    {"text": "规则：公司打印机型号为 HP LaserJet M405；报修入口 http://it.example.local/printer", "meta": {"channel": "kb"}},
    {"text": "规则：员工 WiFi 名为 Corp-WiFi，使用域账号登录。", "meta": {"channel": "kb"}},
]
chat = [
    {"text": "客服记录：用户反馈 WiFi 连不上，协助重置网络后恢复正常。", "meta": {"channel": "chat"}},
    {"text": "客服记录：用户询问邮箱容量，引导清理大附件后解决。", "meta": {"channel": "chat"}},
]

for name, rows in [("helpdesk_qa.jsonl", qa), ("helpdesk_kb.jsonl", kb), ("helpdesk_chat.jsonl", chat)]:
    with open(sources / name, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("sources ready:", len(qa) + len(kb) + len(chat), "rows in 3 files")
~~~

运行：

~~~bash
python data-pipeline/scripts/make_sources.py
ls -la data-pipeline/sources/
~~~

> 真实场景里，这三个文件分别来自“历史工单导出”“知识库文档”“客服聊天记录”三个系统——它们格式不同、字段不同，这就是“多源”的缩影。

### Step 3 归一化：多源变一源（5 分钟）

保存 scripts/normalize_sources.py：

~~~python
# data-pipeline/scripts/normalize_sources.py —— 多源 -> 统一 JSONL + doc_id
import hashlib
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
sources = root / "sources"
staging = root / "staging"
staging.mkdir(parents=True, exist_ok=True)

rows = []
for path in sorted(sources.glob("*.jsonl")):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get("text", "")).strip()
            if not text:
                continue
            doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            rows.append({
                "doc_id": doc_id,
                "source": path.stem,
                "text": text,
                "chars": len(text),
                "meta": obj.get("meta", {}),
            })

out_path = staging / "unified.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("staging rows:", len(rows), "->", out_path)
~~~

运行并抽查：

~~~bash
python data-pipeline/scripts/normalize_sources.py
head -n 1 data-pipeline/staging/unified.jsonl
~~~

### Step 4 记账：生成 manifest（5 分钟）

保存 scripts/audit_manifest.py：

~~~python
# data-pipeline/scripts/audit_manifest.py —— 统计并写 manifest.json
import hashlib
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
staging = root / "staging"
manifests = root / "manifests"
manifests.mkdir(parents=True, exist_ok=True)

rows = 0
chars = 0
doc_ids = set()
for line in open(staging / "unified.jsonl", encoding="utf-8"):
    obj = json.loads(line)
    rows += 1
    chars += obj["chars"]
    doc_ids.add(obj["doc_id"])

manifest = {
    "dataset": "helpdesk-raw-v1",
    "version": "2026-09-05.1",
    "created_by": "local demo pipeline",
    "inputs": [
        {
            "path": str(p),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
        }
        for p in sorted((root / "sources").glob("*.jsonl"))
    ],
    "rules": {"stage": "normalize-v0.1"},
    "stats": {"rows": rows, "chars": chars, "unique_doc_ids": len(doc_ids)},
    "schema_version": "1",
}

out_path = manifests / "manifest.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(json.dumps(manifest["stats"], ensure_ascii=False))
~~~

运行：

~~~bash
python data-pipeline/scripts/audit_manifest.py
cat data-pipeline/manifests/manifest.json
~~~

### Step 5 版本化与可复现验证（5 分钟）

~~~bash
cd llm-demo
git add data-pipeline
git commit -m "data-pipeline: helpdesk raw v1 (normalize v0.1)"
git tag data-pipeline-v0.1
~~~

**可复现验证（黄金动作）**：把 data-pipeline 整个删掉重跑一遍：

~~~bash
rm -rf data-pipeline/sources data-pipeline/staging data-pipeline/manifests
python data-pipeline/scripts/make_sources.py
python data-pipeline/scripts/normalize_sources.py
python data-pipeline/scripts/audit_manifest.py
~~~

如果第二次 manifest 的 stats 与 sha256 和第一次完全一致，说明管道是确定性的（doc_id 由内容哈希生成，天然稳定）——**这就是“同一代码版本+同一输入=同一输出”的可复现性**。第 05 章开始，我们就在 staging/unified.jsonl 上做清洗。

> 本实操是“单机最小版”。企业版只是把脚本换成 Spark/Ray 任务、把目录换成对象存储、把 Git 标签换成数据目录系统，骨架完全一样：**分层、清单、版本、可重跑**。

## 07 参数详解：管道设计的关键配置项

| 配置项 | 建议 | 说明 |
|---|---|---|
| 单文件大小 | 128MB–1GB（压缩后） | 太大并行度差，太小文件数爆炸 |
| 压缩格式 | gzip（通用）/ zstd（更快） | 大数据集必压，省 70% 以上存储 |
| 分区字段 | dataset/source/date | 增量处理与局部重跑的基本单位 |
| 每行 schema | doc_id/source/text/chars/meta | 训练前再加 quality_score 等字段 |
| doc_id | sha256(text) 前 16–32 位 | 稳定、可去重、可溯源 |
| manifest 必填 | dataset/version/inputs/rules/stats/schema_version | 缺一项都不算正式数据集 |
| 完成标记 | 每分区写 _SUCCESS 空文件 | 防止读到写了一半的数据 |
| 保留策略 | 原始层至少保留到模型退役 | 复现训练永远需要原始输入 |
| 抽样预算 | 训练前按预算抽样子集试跑 | 全量清洗前先用 1% 数据验证规则 |

> 参数原则：**目录与清单的规范要“一开始就定”，清洗阈值可以“后面再调”**——前者返工成本极高，后者只是重跑一次。

---

## 08 高频踩坑排查

**坑 1：只有一个 data_final 文件夹，没有版本**
症状：三个月后分不清哪份数据训了哪个模型。
解法：按本章分层目录 + manifest + Git tag；每个训练任务在配置里记录数据集 version。

**坑 2：多源格式硬塞一个脚本里解析**
症状：CSV、XML、JSON、TXT 混着读，脚本全是 if 分支。
解法：每个源一个“适配器”，统一输出 JSONL 标准行；新增源只写新适配器，不动主流程。

**坑 3：来源与 License 不记录**
症状：数据训完要上线才发现来源不明，只能全部下架。
解法：manifest.inputs 里记录 source/url/license/fetched_at；无 License 的数据不进管道。

**坑 4：内存爆炸**
症状：一个 read_csv/read_json 把 100GB 读进内存，直接 OOM。
解法：按分区读、流式逐行处理、必要时上 Spark/Ray；单机先用 polars/duckdb 这类列式引擎。

**坑 5：清洗结果不可复现**
症状：同一份输入两次跑出不同行数。
解法：修掉所有随机性与“当前时间”写入；doc_id 用内容哈希；规则版本写进 manifest（第 05 章再展开规则版本管理）。

**坑 6：目录层级无限嵌套**
症状：sources/2026/09/05/raw/html/part1/xxx.jsonl……路径比内容还长。
解法：最多三层分区（dataset/source/date），其余信息放 manifest 字段，别塞路径。

---

## 09 进阶优化：从小管道到工业管道

**① 质量闸门前置**：每个源接入时先跑 5 分钟快速体检（行数、空文本率、编码、重复率），不合格直接拒收并告警——垃圾进管道后再清洗，成本是源头拦截的十倍。

**② 增量与全量分离**：原始层“全量快照 + 每日增量”，清洗层“只重跑受影响分区”，用分区级依赖避免天天全量重算。

**③ 预算抽样试跑**：任何新清洗规则先抽 1% 数据看分布与样例，确认规则合理再全量跑，能省下大量无效算力。

**④ 数据目录与血缘**：规模上来后把 manifest 汇总进数据目录系统（如 DataHub 类方案或自研表），让“某模型用了哪版数据”可一键查询——这是大模型可审计性的基础设施。

**⑤ 规则代码化、配置化**：清洗规则不要散落在 Notebook 里，全部收进“规则包”（函数+版本号），manifest.rules 记录规则包版本。第 05–09 章会持续往这个规则包里加东西。

---

## 10 本章核心总结（TOP3）

**TOP1**：数据管道 = 搬运 + 质检 + 记账；核心不是算法而是规范——分层目录、统一 JSONL、manifest 清单、版本标签，四件套缺一不可。

**TOP2**：原始数据采集要先解决“来源与 License”再谈规模；接入用“全量打底 + 增量更新”，每个来源一个适配器，统一输出标准行。

**TOP3**：可复现是数据管道的及格线——同一代码版本 + 同一输入必须产出同一输出；doc_id 用内容哈希、规则版本进 manifest、输出落盘加 _SUCCESS 标记，都是为“可重跑”服务。

---

## 11 连载衔接

上一章（第 03 章）我们看懂了预训练的原理，也知道了“数据决定上限”；本章把数据管道的骨架立了起来——你的第一批多源原始数据已经带着 doc_id 和 manifest 躺在 staging 层。

下一章开始处理“脏”：【第 05 章】数据清洗规范与脏数据剔除。我们会回答：网页模板、乱码、截断文本、广告噪音这些脏数据长什么样，怎么批量识别与剔除。

---

## 12 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #数据管道 #数据工程 #大模型训练数据 #数据血缘 #工程实战**（Voice前沿 出品，欢迎收藏追更）

**系列目录（46 章，随连载持续更新）**

第 0 阶段·开篇引路
- 01 一条大模型生产线全程发生了什么
- 02 2 小时跑通最小闭环
- 03 预训练到底在练什么

第 1 阶段·预训练工程·数据核心层
- 04 数据管道从零构建（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 05 章 数据清洗规范与脏数据剔除*