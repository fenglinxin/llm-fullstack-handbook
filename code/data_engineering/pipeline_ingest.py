# -*- coding: utf-8 -*-
"""
第04章 数据管道从零构建：多源摄取与统一目录（工程脚本）

【层级】L1（单机可跑的最小数据管道：源发现->解析->统一记录->报告）
【环境依赖】Python 3.10+，仅标准库；无需 GPU
【核心逻辑】扫描 txt/csv/jsonl 目录->按类型解析成 {id,source,text}->按内容
哈希去重->输出源分布/字符统计报告；真实项目在此之上加对象存储与调度。
【关键参数】
- --sample：自动生成演示数据到 _tmp/ingest_src 并处理（默认关）；
- --src：输入目录（缺省 _tmp/ingest_src）；--out：统一 jsonl 输出。
【避坑】csv 只取第一列；jsonl 优先取 text 字段；txt 按行拆分；
文件编码统一 utf-8，否则 GBK 文件会抛错（可加 errors=ignore 兜底）。
【运行结果示例】
$ python pipeline_ingest.py --sample
records total: 7
chars total: 43
  _tmp/ingest_src/logs/c.csv               3
  _tmp/ingest_src/web/a.txt                2
  _tmp/ingest_src/web/b.jsonl              2
raw_total: 8 unique_text_ratio: 0.875
【高频报错 Top5】
1. 目录不存在：先 --sample 生成或检查 --src 路径；
2. jsonl 有非 text 字段：解析为整行 JSON 字符串（已兜底）；
3. 编码错误：GBK 文件需 errors=ignore 或先转码；
4. 重复记录：id 由 hash(text) 生成，重复内容自动合并；
5. 大目录很慢：真实环境用 glob+多进程（工程改造方向）。
【输出解读】看到 source 分布与字符总量统计即管道成功。
【工程改造方向】接 S3/OSS 读取、Airflow/DVC 编排、按目录分区落盘。
"""

import argparse
import hashlib
import json
import pathlib


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true")
    p.add_argument("--src", default=None)
    p.add_argument("--out", default="_tmp/ingest_out.jsonl")
    return p.parse_args()


def gen_sample(root):
    root = pathlib.Path(root)
    (root / "web").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "web" / "a.txt").write_text("第一条网页文本\n第二条网页文本\n重复文本A\n", encoding="utf-8")
    (root / "web" / "b.jsonl").write_text('{"text": "jsonl 第一条"}\n{"text": "jsonl 第二条"}\n', encoding="utf-8")
    (root / "logs" / "c.csv").write_text("内容,other\n日志一,1\n日志二,2\n重复文本A,9\n", encoding="utf-8")


def read_txt(p):
    return [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]


def read_csv(p):
    rows = []
    for i, ln in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if not ln.strip():
            continue
        parts = ln.split(",")
        if i == 0 and len(parts) > 1:  # 跳过表头：第二列不是数字则视为 header
            try:
                float(parts[1])
            except ValueError:
                continue
        rows.append(parts[0].strip())
    return rows


def read_jsonl(p):
    rows = []
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
            rows.append(obj.get("text", ln.strip()))
        except json.JSONDecodeError:
            rows.append(ln.strip())
    return rows


def main():
    args = parse_args()
    src = pathlib.Path(args.src or "_tmp/ingest_src")
    if args.sample:
        gen_sample(src)
    records, seen = [], set()
    raw_total = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix == ".txt":
            texts = read_txt(p)
        elif p.suffix == ".csv":
            texts = read_csv(p)
        elif p.suffix == ".jsonl":
            texts = read_jsonl(p)
        else:
            continue
        raw_total += len(texts)
        for text in texts:
            rid = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
            if rid in seen:
                continue
            seen.add(rid)
            records.append({"id": rid, "source": str(p), "text": text})
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_src = {}
    for r in records:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print("records total:", len(records))
    print("chars total:", sum(len(r["text"]) for r in records))
    for s, n in sorted(by_src.items()):
        print("  %-40s %d" % (s, n))
    print("raw_total: %d unique_text_ratio: %.3f" % (raw_total, len(records) / max(1, raw_total)))


if __name__ == "__main__":
    main()