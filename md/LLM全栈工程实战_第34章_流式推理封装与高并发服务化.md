# 【LLM全栈工程·第34章】流式推理封装与高并发服务化：从单实例到生产网关

> 本篇为「LLM全栈工程」连载第 34 章（Voice前沿）。主标题以上为准；备选标题（供运营选用，不进正文）：① SSE 流式输出为什么会被网关“吃掉”？流式封装实战；② 多实例、限流、负载均衡：LLM 高并发服务化；③ 稳定性三件套：健康检查、队列与优雅退出。

---

## 单篇内容卡（排版本忽略）

| 字段 | 内容 |
|---|---|
| 章节定位 | 「基础部署与推理优化」第 8 篇：把单机 vLLM 变成可上线的流式高并发服务 |
| 适用场景 | 个人学习（流式客户端）；小团队上线对话服务；企业多实例网关 |
| 核心知识点 | SSE 流式协议；网关/负载均衡；限流与队列；稳定性与优雅退出 |
| 技术选型 | 直连 vs 网关（Nginx/Ingress）；多实例路由策略 |
| 分步实操 | 流式客户端 → 双实例 + Nginx 网关 → 流式压测 → 故障演练 → 监控基线 |
| 参数详解 | 流式超时、proxy_buffering、限流 QPS、队列深度、重试策略 |
| 踩坑排查 | 网关缓冲吞流、客户端超时过短、重试造成重复计费、实例热不均 |
| 进阶优化 | K8s HPA、PD 分离路由、小模型兜底、成本路由 |
| 本章 TOP3 | 见文末 |
| 下章预告 | 第 35 章：推理编译器核心原理 |

---

## 01 开篇导语

模型服务单实例跑通了，但“能用”和“能上线”之间还有一整层工程：**流式体验、多实例、网关、限流、稳定性**。

常见事故现场：

- 前端等了 5 秒一个字没出——Nginx 把 SSE 缓冲了；
- 客户端 60 秒超时，但 TTFT 要 80 秒——排队请求被掐死；
- 网关超时自动重试——同一个贵请求被算了两遍钱；
- 两个实例一个忙死一个闲死——负载均衡策略不对。

本章把服务化拆成四件事：流式封装、网关路由、限流队列、稳定性演练，每件都给可复现的配置与脚本。

> 一句话记住本章：**LLM 服务化 = 流式（SSE 不能被缓冲）+ 路由（按负载/缓存亲和）+ 限流（队列有界）+ 演练（挂了能恢复）。**

---

## 02 白话原理：SSE 流式与网关的冲突

### 2.1 SSE 是什么

SSE（Server-Sent Events）是 HTTP 上的流式协议：服务端持续向客户端推送数据块，直到 [DONE]。OpenAI 兼容接口的 stream=true 就是 SSE。

### 2.2 为什么网关会“吞掉”流

Nginx 等反向代理默认会**缓冲后端响应**，攒够一定量才发给客户端——流式输出的“逐字效果”没了，TTFT 也变成“等整段”。

解法：对 LLM 流式路由必须关缓冲：

~~~nginx
proxy_buffering off;
proxy_cache off;
proxy_read_timeout 300s;   # 覆盖“首字慢+长生成”场景
proxy_http_version 1.1;
~~~

### 2.3 客户端要处理的四件事

1. 按 chunk 解析 delta，而不是等完整响应；
2. 区分“网络中断”与“正常结束”（[DONE]）；
3. 超时按“首字超时”与“总超时”分开设；
4. 用户取消时主动断开，省后端算力。

> 记忆锚点：**流式体验 = 服务端分块 + 网关不缓冲 + 客户端逐块渲染 + 超时分级。**

## 03 高并发架构与选型

### 3.1 分层架构

~~~text
客户端/业务层
   → 网关层（限流、路由、鉴权、重试策略）
   → 推理实例池（多个 vLLM/SGLang，各自 max-num-seqs）
   → 监控/告警（指标、日志、trace）
~~~

### 3.2 路由策略

| 策略 | 适合 | 说明 |
|---|---|---|
| 轮询 | 通用 | 简单，但实例负载可能不均 |
| 最少连接 | 长请求多 | 按活动连接数路由 |
| 前缀缓存亲和 | system 前缀共享流量 | 同一前缀尽量进同一实例，缓存命中率高 |
| 队列感知 | 高并发 | 路由到队列最短的实例 |

### 3.3 限流与队列

- 网关限流：按 API Key/客户端限 QPS 与并发；
- 实例队列：vLLM max-num-seqs 之外的请求在网关排队（**队列必须有界**）；
- 拒绝策略：超队列返回 429 + Retry-After，而不是无限等待；
- 降级：队列满时把请求路由到小模型/备用池（可选）。

### 3.4 稳定性三板斧

1. **健康检查**：/health + 真实推理探针（每 30s 问一个 1-token 请求）；
2. **优雅退出**：停实例前先摘流量，等存量流式请求结束再杀进程；
3. **熔断**：实例连续错误超阈值自动摘除，恢复后自动回归。

> 生产提示：网关层不要做“无限重试”——LLM 请求贵且非幂等；只对“连接类错误”做 1 次重试，超时/业务错误直接返回。

## 05 分步实操：流式客户端与高并发演练（约 1 小时）

### Step 1 流式客户端（10 分钟）

保存 scripts/stream_client.py：

~~~python
# scripts/stream_client.py —— SSE 流式客户端：逐块输出+测 TTFT
# 用法：python stream_client.py <base_url> <model> [prompt]
import sys
import time

from openai import OpenAI

base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1"
model = sys.argv[2] if len(sys.argv) > 2 else "helpdesk"
prompt = sys.argv[3] if len(sys.argv) > 3 else "打印机连不上怎么办？"

client = OpenAI(base_url=base_url, api_key="EMPTY")
start = time.time()
first_ts = None
chars = 0
try:
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        if first_ts is None:
            first_ts = time.time()
            print("\n[TTFT %.0f ms]\n" % ((first_ts - start) * 1000))
        delta = chunk.choices[0].delta.content
        if delta:
            chars += len(delta)
            print(delta, end="", flush=True)
    print("\n[total %.1f s, chars %d]" % (time.time() - start, chars))
except KeyboardInterrupt:
    print("\n[user cancelled]")
except Exception as exc:
    print("\n[error]", exc)
~~~

运行：

~~~bash
python scripts/stream_client.py http://localhost:8000/v1 helpdesk
~~~

**检查**：是否逐块出现（不是整段一次性返回）；Ctrl+C 能否立即断开。

### Step 2 流式压测脚本（10 分钟）

保存 scripts/loadgen_stream.py：

~~~python
# scripts/loadgen_stream.py —— 并发流式压测：TTFT/错误率
# 用法：python loadgen_stream.py <名称> <base_url> <并发> <总请求>
import concurrent.futures
import json
import pathlib
import statistics
import sys
import time

from openai import OpenAI

name = sys.argv[1]
base_url = sys.argv[2]
concurrency = int(sys.argv[3]) if len(sys.argv) > 3 else 16
total = int(sys.argv[4]) if len(sys.argv) > 4 else 64
root = pathlib.Path(__file__).resolve().parent.parent
PROMPT = "请分点说明打印机连不上时的处理步骤。"

def run_once(_):
    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=120)
    start = time.time()
    first_ts = None
    chunks = 0
    try:
        stream = client.chat.completions.create(
            model="helpdesk",
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=150,
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            if first_ts is None:
                first_ts = time.time()
            if chunk.choices[0].delta.content:
                chunks += 1
        return {"ok": True, "ttft": first_ts - start if first_ts else None, "chunks": chunks}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:100]}

for _ in range(2):
    run_once(None)

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
    for res in pool.map(run_once, range(total)):
        results.append(res)

oks = [r for r in results if r["ok"]]
ttfts = [r["ttft"] for r in oks if r["ttft"] is not None]

def pct(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]

report = {"name": name, "total": total, "success": len(oks), "error": total - len(oks),
          "ttft_p50_ms": round(pct(ttfts, 0.5) * 1000, 1) if ttfts else None,
          "ttft_p95_ms": round(pct(ttfts, 0.95) * 1000, 1) if ttfts else None}
out = root / "logs" / ("loadgen_" + name + ".json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
~~~

### Step 3 双实例 + Nginx 网关（20 分钟）

1. 起两个 vLLM 实例（8001/8002 端口，同一模型，max-num-seqs 各 32）；
2. Nginx 配置（关键行）：

~~~nginx
upstream llm_backend {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}
server {
    listen 8000;
    location /v1/ {
        proxy_pass http://llm_backend;
        proxy_http_version 1.1;
        proxy_buffering off;      # 关键：不缓冲 SSE
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_connect_timeout 5s;
    }
}
~~~

3. 重启 Nginx 后通过 8000 访问，先跑 stream_client 确认流式正常；
4. 压测：

~~~bash
python scripts/loadgen_stream.py gw http://localhost:8000/v1 32 128
~~~

### Step 4 故障演练（10 分钟）

1. 压测进行中 kill 掉一个后端实例；
2. 观察：网关是否把新请求全部路由到存活实例（健康检查生效）；
3. 存量流式连接是否正常结束或快速报错（优雅处理）；
4. 重启被 kill 的实例，确认自动回归；把演练结论写进 docs/ha_runbook.md 并归档：

~~~bash
cd llm-demo
git add scripts logs docs
git commit -m "serving: stream client + loadgen + nginx gateway drill"
git tag serving-v1
~~~

> 生产进阶：K8s 下用 Service/Ingress + HPA（按队列深度/GPU 利用率扩缩），见第 43 章；监控大盘与告警见第 45 章。

## 06 参数详解：服务化配置速查

| 参数 | 参考 | 说明 |
|---|---|---|
| proxy_buffering | off | SSE 必需 |
| proxy_read_timeout | 120–600s | 覆盖长生成 |
| 首字超时 | 10–30s | 排队/TTFT 预算 |
| 总超时 | 60–600s | 按 max_tokens |
| 限流 | 按业务 QPS×峰值系数 | 429+Retry-After |
| 队列深度 | 每实例 max-num-seqs×2~4 | 有界，防雪崩 |
| 健康探针 | 每 30s 真实 1-token 请求 | 比 /health 更真 |
| 重试 | 仅连接类错误 1 次 | LLM 请求非幂等 |

---

## 07 高频踩坑排查

**坑 1：Nginx 缓冲吞流**
症状：前端等半天才整段出现。
解法：proxy_buffering off + http1.1 + 长 read timeout。

**坑 2：客户端超时一刀切**
症状：总超时 30s，排队 20s+生成 15s 的请求被掐。
解法：首字超时与总超时分开；超时按 P95 设计。

**坑 3：自动重试重复计费**
症状：网关超时重试，同一个生成请求跑了两遍。
解法：只重试连接类错误；给请求幂等 ID，业务层去重。

**坑 4：队列无限增长**
症状：高峰请求全堆内存里，实例没挂网关先挂。
解法：有界队列+429；压测队列深度监控。

**坑 5：实例热不均**
症状：轮询下某些实例排队、某些空闲。
解法：最少连接或队列感知路由；前缀缓存场景用亲和路由。

**坑 6：没有优雅退出**
症状：发布时 kill 实例，存量流式连接全断。
解法：摘流量→等存量结束→再杀；K8s 用 preStop+terminationGracePeriod。

**坑 7：只看平均时延**
症状：均值漂亮，P95 排队 10 秒没人发现。
解法：TTFT P50/P95/P99 + 队列深度 + 错误率一起看。

---

## 08 进阶优化：服务化进化方向

**① K8s 自动扩缩**：按队列深度/GPU 利用率 HPA，低峰缩容省钱、高峰扩容保体验（第 43 章）。

**② PD 分离路由**：prefill 与 decode 分实例后，网关按请求阶段路由（第 39 章），长 prompt 场景吞吐质变。

**③ 小模型兜底**：队列满或大模型故障时路由到 3B 学生模型（第 22 章），保可用性降体验损失。

**④ 成本路由**：低价值请求走小模型/低峰，高价值请求走大模型——网关变成成本优化器。

**⑤ 全链路观测**：请求 ID 贯穿网关→推理实例→GPU 指标，TTFT/TPOT/队列/错误一键归因（第 45 章）。

---

## 09 本章核心总结（TOP3）

**TOP1**：SSE 流式要过三关：服务端分块、网关关缓冲（proxy_buffering off）、客户端逐块渲染+分级超时。

**TOP2**：高并发 = 多实例 + 健康检查路由 + 有界队列/429 + 优雅退出；重试只限连接类错误一次，防重复计费。

**TOP3**：上线前做故障演练：kill 实例看自动摘除与恢复；监控看 P50/P95/P99 与队列深度，不看平均。

---

## 10 连载衔接

上一章（第 33 章）管好了 KV 显存；本章把单实例升级为流式高并发服务——网关、队列、健康检查与演练全部就位，部署阶段收官在即。

下一阶段进入“编译器级”深水区：【第 35 章】推理编译器核心原理：图编译、IR、调度与代码生成。

---

## 11 话题标签与系列目录索引

话题标签：**#LLM全栈工程 #SSE #流式输出 #高并发 #负载均衡 #服务化 #工程实战**（Voice前沿 出品，欢迎收藏追更）

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
- 34 流式推理封装与高并发服务化（本篇）

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

*本文由 Voice前沿 出品 · 转载注明出处 · 下一篇：第 35 章 推理编译器核心原理*