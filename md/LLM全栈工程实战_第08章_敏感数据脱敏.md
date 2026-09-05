# 【LLM全栈工程·第08章】敏感数据脱敏：PII 识别、脱敏策略与合规落地

> 本篇为「LLM全栈工程」连载第 08 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① 训练数据里的身份证号、手机号，一个都不能进模型；② PII 识别与脱敏实战：正则、校验位、掩码与泛化；③ 脱敏不是打马赛克：识别率、误伤率与再识别攻击测试。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「预训练工程·数据核心层」第 6 篇：在数据进训练集之前完成敏感信息识别与脱敏 |
| 适用场景 | 个人学习（本地 PII 规则）；小团队处理工单/日志/客服语料；企业合规数据生产线 |
| 核心知识点 | PII 类型；识别三件套（正则+校验位+Ner）；掩码/泛化/删除/假名化策略；脱敏质检与再识别测试 |
| 技术选型 | 正则+校验位 vs 开源 PII 框架（Presidio 类） vs LLM 识别 vs 商业 DLP |
| 分步实操 | 在第 07 章 dedup 层后加 safe 层：注入含 PII 样本 → 识别 → 按策略脱敏 → 再识别测试 → 打版本 |
| 参数详解 | 各类型掩码规则、识别置信度、删除阈值、策略表结构 |
| 踩坑排查 | 只靠正则漏变体、校验位缺失误伤、掩码破坏格式、假名化映射丢失、脱敏后仍可拼回 |
| 进阶优化 | 假名化保险库、PII 分类器、LLM 兜底识别、红队再识别测试、合规审计链 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 09 章：数据质量打分体系 |

> 合规声明：本文只讲工程方法，不构成法律意见；具体数据处理请以适用法律法规与公司法务意见为准（如中国《个人信息保护法》等）。

---

## 01 开篇导语

想象一个场景：你从公司工单系统导出 10 万条客服语料准备做领域模型，里面全是真实对话——**“张先生您好，您的手机号 138**** 需要更新吗？”** 如果这份语料不脱敏直接训练，模型可能学会在回答中“复述”用户的手机号、身份证号、家庭住址。

这不是危言耸听：大模型的记忆能力会让它把训练数据里的个人信息“背”出来。**训练语料里的 PII（Personal Identifiable Information，个人可识别信息），是数据合规与用户隐私的双重红线。**

本章解决三个问题：

1. 哪些信息算 PII，藏在什么形态里；
2. 怎么高召回、低误伤地把它们找出来；
3. 找到之后用掩码、泛化、删除还是假名化，以及怎么证明“脱干净了”。

交付物：接在去重层之后的 safe 层——一套带策略表的 PII 识别/脱敏脚本 + 再识别测试 + 审计报告。

> 一句话记住本章：**脱敏不是“把几个数字换成星号”，而是“识别—脱敏—质检—审计”的闭环，且每一步都要能证明。**

---

## 02 白话原理：为什么 PII 必须“出不了训练集”

### 2.1 PII 是什么

PII 是能直接或间接识别到具体个人的信息，常见类型：

| 类别 | 示例 | 识别难度 |
|---|---|---|
| 直接标识符 | 身份证号、手机号、护照号、银行卡号 | 低（格式+校验位） |
| 联系方式 | 邮箱、电话、家庭地址 | 中 |
| 间接标识符 | 姓名+生日+公司组合、工号、车牌 | 高（需组合判断） |
| 敏感个人数据 | 健康记录、财务、生物特征、行踪 | 高（合规要求更严） |

### 2.2 为什么模型会把 PII “背”出来

预训练/微调的本质是让模型记住语料规律。高频出现的完整手机号、身份证号，会被当成“正常文本规律”学进参数。即使训练时没被直接问，用户也可能用提示词诱导模型回忆——这就是“记忆泄漏”（memorization）风险。

### 2.3 脱敏的本质：在“可用性”与“隐私风险”之间找平衡

| 策略 | 隐私保护 | 数据可用性 | 适用 |
|---|---|---|---|
| 删除整条 | 最高 | 丢失整条样本 | 高危文档 |
| 掩码 | 高 | 保留结构 | 手机号/身份证/卡号 |
| 泛化 | 中高 | 保留统计意义 | 地址/年龄/收入区间 |
| 假名化（Tokenization） | 高（配合保险库） | 保留关联分析能力 | 需要跨字段关联的场景 |

> 工程原则：**先识别、后分级、再选策略**；不同 PII 类型用不同策略，全部写进策略表并版本化。

## 03 PII 识别：三件套组合拳

### 3.1 识别方法对比

| 方法 | 抓什么 | 优点 | 缺点 |
|---|---|---|---|
| 正则 | 手机号、邮箱、卡号等强格式 | 快、零成本、可解释 | 漏变体、误伤普通数字串 |
| 校验位 | 身份证、银行卡（Luhn） | 大幅降误伤 | 只对带校验位的类型有效 |
| 上下文规则 | “姓名：张三”“电话 138…” | 抓无格式 PII | 依赖上下文模板 |
| NER 模型 | 人名、地名、机构、地址 | 泛化好 | 需训练/调优，CPU 成本高 |
| LLM 识别 | 模糊/嵌套/跨句 PII | 理解最强 | 慢、贵、有漏检，只做兜底 |

**生产组合**：正则+校验位做第一层召回（便宜、快），NER/上下文规则做第二层补漏，LLM 只抽检小批量疑难样本，人工复核高风险类型。

### 3.2 关键技巧：识别要“带上下文、带校验”

- 手机号：1[3-9] 开头 + 11 位数字 + 前后不能是数字；
- 身份证：18 位 + 末位校验位算法校验，**校验不过的 18 位数字串不是身份证**（避免把订单号当身份证）；
- 银行卡：13–19 位数字 + Luhn 校验；
- 邮箱：标准邮箱正则 + 排除 example.com 类测试域名要按业务决定；
- 姓名/地址：正则很难覆盖，用“称谓词/字段名上下文 + NER”识别。

> 误伤比漏检更隐蔽：把普通 18 位数字全当身份证掩掉，会毁掉大量正常业务文本。**凡是带校验位的类型必须做校验位。**

---

## 04 脱敏策略与工具选型

### 4.1 策略表设计

每条 PII 类型一个策略条目：

~~~json
{
  "phone": {"action": "mask", "keep_head": 3, "keep_tail": 4},
  "email": {"action": "mask", "keep_head": 1},
  "id_card": {"action": "mask", "keep_head": 6, "keep_tail": 4},
  "ip": {"action": "mask", "keep_head": 0},
  "address": {"action": "generalize", "level": "district"},
  "high_risk_doc": {"action": "delete", "pii_count_threshold": 5}
}
~~~

- 掩码（mask）：保留头尾、中间打星，保留格式可读性；
- 泛化（generalize）：把“北京市朝阳区建国路 88 号”泛化成“北京市朝阳区”；
- 删除（delete）：PII 密度过高或高危类型直接弃档；
- 假名化（tokenize）：真值替换成随机 token，映射表放加密保险库（进阶方案）。

### 4.2 工具选型

| 方案 | 定位 | 适合 |
|---|---|---|
| 自研正则+校验位（本章主推起步） | 快、可控 | 手机/邮箱/证件等强格式为主 |
| Presidio 类开源 PII 框架 | 规则+NER 开箱 | 多语言、多类型，中大型管道 |
| 商用 DLP/数据安全平台 | 全托管 | 企业级合规审计需求 |
| LLM 识别 | 兜底精查 | 小批量、复杂上下文 |

结论：**第一步先上“正则+校验位+上下文规则”**，覆盖 80% 的强格式 PII；识别率不够再引入 Presidio 类框架与 NER 模型。

## 05 分步实操：给管道加上 safe 层（30 分钟）

继续沿用 llm-demo/data-pipeline。safe 层读 dedup/unique.jsonl，输出“无 PII”版本。

### Step 1 注入含 PII 的样本（5 分钟）

保存 scripts/make_pii_sources.py：

~~~python
# data-pipeline/scripts/make_pii_sources.py —— 注入含 PII 的演示样本
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
sources = root / "sources"

pii_rows = [
    {"text": "客户联系手机 13812345678，邮箱 zhangsan@example.com，请今天内处理。", "meta": {"channel": "pii_contact"}},
    {"text": "身份证号码：110101199003077512，开户行信息见附件。", "meta": {"channel": "pii_id"}},
    {"text": "我的工号是 9527，电话 13900001111，有急事请直接打这个号码。", "meta": {"channel": "pii_phone"}},
    {"text": "请把周报发送至 lisi@example.com，并抄送 wangwu@corp.example.cn。", "meta": {"channel": "pii_email"}},
    {"text": "打印机卡纸处理：关机断电，打开前盖取出卡纸，合盖后重新开机测试。", "meta": {"channel": "qa_normal"}},
]

with open(sources / "helpdesk_pii.jsonl", "w", encoding="utf-8") as f:
    for row in pii_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("pii rows:", len(pii_rows))
~~~

> 说明：示例中的手机号/邮箱/身份证均为演示构造（身份证校验位按国标算法计算，不代表真实个人）。生产环境请使用真实脱敏测试集与合规审批流程。

### Step 2 把 PII 样本走完“归一化 → 清洗 → 过滤 → 去重”

~~~bash
python data-pipeline/scripts/make_pii_sources.py
python data-pipeline/scripts/normalize_sources.py
python data-pipeline/scripts/clean_sources.py
python data-pipeline/scripts/filter_sources.py
python data-pipeline/scripts/dedup_sources.py
wc -l data-pipeline/dedup/unique.jsonl
~~~

含 PII 的样本内容本身“干净、无害、不重复”，所以会一路通过前四层——这正是 safe 层存在的意义。

### Step 3 写识别 + 脱敏脚本（15 分钟）

保存 scripts/mask_pii.py：

~~~python
# data-pipeline/scripts/mask_pii.py —— PII 识别（正则+校验位）与掩码脱敏
import json
import pathlib
import re

root = pathlib.Path(__file__).resolve().parent.parent
dedup_dir = root / "dedup"
safe_dir = root / "safe"
manifests = root / "manifests"
safe_dir.mkdir(parents=True, exist_ok=True)

# 策略表：不同类型用不同动作（mask 保留头尾）
POLICY = {
    "phone": {"keep_head": 3, "keep_tail": 4},
    "id_card": {"keep_head": 6, "keep_tail": 4},
    "email": {"keep_head": 1},
}
MAX_PII_PER_DOC = 5   # 超过该数量视为高危文档，整条删除

ID_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
ID_MAP = "10X98765432"

def valid_id_card(value):
    if len(value) != 18:
        return False
    total = 0
    for i in range(17):
        ch = value[i]
        if not ch.isdigit():
            return False
        total += int(ch) * ID_WEIGHTS[i]
    return ID_MAP[total % 11] == value[17].upper()

def mask_phone(match):
    v = match.group(0)
    return v[:3] + "*" * 4 + v[7:]

def mask_id_card(match):
    v = match.group(0)
    if not valid_id_card(v):
        return v  # 校验不过：不是身份证，不掩码
    return v[:6] + "*" * 8 + v[14:]

def mask_email(match):
    v = match.group(0)
    at = v.find("@")
    return v[0] + "***" + v[at:]

PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_RE = re.compile(r"(?<![0-9A-Za-z])\d{17}[0-9Xx](?![0-9A-Za-z])")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def find_pii(text):
    found = []
    for m in PHONE_RE.finditer(text):
        if m.group(0)[0] == "1":
            found.append(("phone", m.start(), m.end()))
    for m in ID_RE.finditer(text):
        if valid_id_card(m.group(0)):
            found.append(("id_card", m.start(), m.end()))
    for m in EMAIL_RE.finditer(text):
        found.append(("email", m.start(), m.end()))
    return found

def mask_text(text, pii_list):
    # 从后往前替换，避免位移
    parts = []
    prev = 0
    result = text
    for ptype, start, end in sorted(pii_list, key=lambda x: -x[1]):
        seg = text[start:end]
        if ptype == "phone":
            rep = mask_phone(re.match(PHONE_RE.pattern, seg))
        elif ptype == "id_card":
            rep = mask_id_card(re.match(ID_RE.pattern, seg))
        else:
            rep = mask_email(re.match(EMAIL_RE.pattern, seg))
        result = result[:start] + rep + result[end:]
    return result

report = []
kept = []
deleted = []

for line in open(dedup_dir / "unique.jsonl", encoding="utf-8"):
    obj = json.loads(line)
    text = str(obj.get("text", ""))
    pii = find_pii(text)
    if len(pii) >= MAX_PII_PER_DOC:
        deleted.append({"doc_id": obj["doc_id"], "reason": "high_pii_density", "count": len(pii)})
        continue
    masked = mask_text(text, pii)
    for ptype, start, end in pii:
        report.append({"doc_id": obj["doc_id"], "type": ptype, "raw": text[start:end], "masked_at": start})
    obj["text"] = masked
    kept.append(obj)

with open(safe_dir / "safe.jsonl", "w", encoding="utf-8") as f:
    for obj in kept:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
with open(safe_dir / "pii_report.jsonl", "w", encoding="utf-8") as f:
    for item in report:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
with open(safe_dir / "deleted_report.jsonl", "w", encoding="utf-8") as f:
    for item in deleted:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("kept:", len(kept), "deleted:", len(deleted), "pii found:", len(report))
from collections import Counter
print("by_type:", dict(Counter(item["type"] for item in report)))
~~~

### Step 4 跑脱敏 + 再识别测试（5 分钟）

~~~bash
python data-pipeline/scripts/mask_pii.py
wc -l data-pipeline/dedup/unique.jsonl data-pipeline/safe/safe.jsonl
cat data-pipeline/safe/pii_report.jsonl
~~~

**再识别测试（关键）**：确认原始 PII 在脱敏文本中已不存在：

~~~bash
grep -c "13812345678" data-pipeline/safe/safe.jsonl || echo "phone NOT found (OK)"
grep -c "110101199003077512" data-pipeline/safe/safe.jsonl || echo "id NOT found (OK)"
grep -c "zhangsan@example.com" data-pipeline/safe/safe.jsonl || echo "email NOT found (OK)"
~~~

预期三条原始值都查不到；safe.jsonl 里出现的是 138****5678、110101********7512、z***@example.com 这类掩码值。

### Step 5 人工抽检 + 打版本（5 分钟）

抽 20 条脱敏结果人工看：掩码是否保留必要格式、有没有把正常数字误伤（比如把订单号当成手机号）。确认后：

~~~bash
cd llm-demo
git add data-pipeline
git commit -m "safe: PII mask v0.1 (phone/id/email + checksum + re-id test)"
git tag data-safe-v0.1
~~~

> 生产提示：pii_report 里记录了“原始值”，属于敏感信息本身——正式环境应只记录类型与位置，原始值仅存于受限审计库或直接不落盘。

## 06 参数详解：脱敏配置速查

| 参数 | 参考取值 | 说明 |
|---|---|---|
| phone keep_head/keep_tail | 3/4 | 138****5678；保留前 3 后 4 兼顾可读与隐私 |
| id_card keep_head/keep_tail | 6/4 | 110101********7512；保留地区码与后 4 位 |
| email keep_head | 1–2 | z***@example.com；域名完整保留便于业务识别 |
| MAX_PII_PER_DOC | 3–10 | PII 密度过高视为高危文档删除；按语料实测 |
| 校验位开关 | 身份证/银行卡必须开 | 没有校验位的纯正则会把订单号当证件号 |
| NER 置信度阈值 | 0.5–0.8 | 人名/地址类识别用；低置信度进人工复核 |
| 泛化粒度 | 省/市/区/街道 | 地址按业务需要选粒度，能不用精确地址就不用 |

> 参数铁律：**掩码规则必须“可逆检查”**——脱敏后跑一遍再识别，确认原始值不再出现；再用 20–50 条人工样本检查可用性（格式是否保留、是否误伤）。

---

## 07 高频踩坑排查

**坑 1：只靠正则，漏掉变体与上下文 PII**
症状：手机号中间加空格“138 1234 5678”就漏了；姓名“王总”识别不了。
解法：识别前先做文本归一化（去空白/全半角），正则负责强格式，NER+上下文规则补人名地址，LLM 只做兜底。

**坑 2：不做校验位，误伤业务数字**
症状：订单号、快递单号等 18 位数字被当身份证掩掉。
解法：身份证做国标校验位、银行卡做 Luhn 校验，校验不过一律不掩。

**坑 3：掩码破坏格式，下游没法用**
症状：把整串全打星，模型学到“手机号全是星号”。
解法：保留头尾与分隔符结构（138****5678）；掩码是“保结构去内容”，不是涂黑。

**坑 4：假名化映射表丢了**
症状：用假名化后没保存映射，后续想关联分析对不上，或映射表明文放仓库被拖库。
解法：映射表加密存放独立保险库，与训练数据隔离；没有恢复需求的场景直接用掩码/泛化更省事。

**坑 5：脱敏后不做再识别测试**
症状：以为掩码了，结果正则没覆盖的变体把真手机号带进了训练集。
解法：跑“再识别攻击测试”：用识别器+人工对脱敏结果复查，原始值出现率为 0 才算过。

**坑 6：PII 报告里明文记录原始值**
症状：pii_report.jsonl 把身份证原文和 doc_id 写一起，报告本身成了新的泄密点。
解法：报告只记类型、位置、掩码后值；原始值按最小化原则不落盘或进加密审计库。

**坑 7：忽略“组合识别”风险**
症状：单看“张三”不敏感，但“张三+1990-03-07+北京朝阳”组合起来就是高识别度 PII。
解法：对“间接标识符组合”单独设规则：同一文档出现姓名+生日+地域等组合时升级处理（删除或泛化）。

---

## 08 进阶优化：脱敏体系的进化方向

**① 假名化保险库**：需要跨文档关联（同一客户的多次对话要能串起来）时，用“真值→随机 token”映射；映射表加密存储、严格权限、定期轮换，训练数据里只有 token。

**② PII 分类器接力**：把人工复核过的“漏网/误伤”样本回流，训练 NER/分类模型识别人名、地址等无强格式 PII，逐步降低对正则的依赖。

**③ LLM 兜底抽检**：每批次随机抽 1% 走 LLM 复查“是否还有可识别个人的信息”，与规则结果交叉验证；LLM 输出只做标记，不做自动删除。

**④ 红队再识别测试**：模拟攻击者用“训练好的模型诱导回忆 + 对脱敏库做链接攻击”两条路径测试，把“能否拼回个人”作为脱敏效果指标。

**⑤ 合规审计链**：PII 策略版本、识别日志、人工复核记录、再识别测试报告全部归档，配合数据流图（谁在什么环节接触了什么数据），形成可审计的合规证据链。

---

## 09 本章核心总结（TOP3）

**TOP1**：PII 识别要用“正则召回 + 校验位降误伤 + NER/上下文补漏 + 人工复核”组合拳；带校验位的类型（身份证、银行卡）必须校验，否则误伤业务数字。

**TOP2**：脱敏策略按类型分级：强格式用掩码（保结构）、地址年龄用泛化、高危文档直接删除、需要跨文档关联才用假名化（映射表加密隔离）。

**TOP3**：脱敏的验收标准是“再识别测试通过”：原始值在脱敏结果中出现率为 0，且人工抽检确认格式可用、无误伤；PII 报告遵循最小化原则，不记录明文原始值。

---

## 10 连载衔接

上一章（第 07 章）消灭了重复；本章加上了 safe 层——现在进入训练候选池的文本，已经“干净、该留、不重复、无 PII”。

下一章给数据排座次：【第 09 章】数据质量打分体系：规则、分类器与多维度融合。不是所有活下来的数据都值得同等待遇——质量分将决定谁进训练、谁被降权。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #数据脱敏 #个人信息保护 #PII #数据合规 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 08 敏感数据脱敏（本篇）
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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 09 章 数据质量打分体系*