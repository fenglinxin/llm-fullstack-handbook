#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build static GitHub Pages site for LLM 全栈工程 handbook."""
import html as html_mod
import pathlib
import re

import markdown

ROOT = pathlib.Path(__file__).resolve().parents[1]
MD_DIR = ROOT / "md"
OUT_DIR = ROOT / "docs"
CH_DIR = OUT_DIR / "chapters"

SERIES = "LLM 全栈工程"
SUBTITLE = "LLM 工业级全栈工程落地手册 · 第 00 章 + 主线 46 章"
TOTAL = 46

STAGES = [
    (0, 3, "架构通识与第 0 阶段 · 开篇引路", "第 00 章架构通识 + 全栈地图、最小闭环、预训练原理"),
    (4, 13, "第 1 阶段 · 预训练工程·数据核心层", "数据管道、清洗、过滤、去重、脱敏、质量打分、领域数据、增量预训练、超参与稳定性"),
    (14, 26, "第 2 阶段 · 微调与对齐体系", "SFT、数据标注、模板、调参、拟合与遗忘、评估、多轮、蒸馏、RM、PPO、RLAIF、冷启动"),
    (27, 34, "第 3 阶段 · 基础部署与推理优化", "框架对比、vLLM、SGLang、TensorRT-LLM、量化、剪枝、KV Cache、流式与高并发"),
    (35, 40, "第 4 阶段 · 高阶推理编译器极致优化", "编译器原理、算子融合、动态 shape、投机解码、批量并行、内核与编译级量化"),
    (41, 46, "第 5 阶段 · 全栈工程联调与项目落地", "全链路、领域定制、高并发与端侧、模型迭代、线上排障、最佳实践"),
]

NUM_RE = re.compile(r"第(\d+)章")


def chapter_no(path: pathlib.Path) -> int:
    m = NUM_RE.search(path.name)
    return int(m.group(1)) if m else 999


def build_page(title, body, toc, prev_html, next_html, rel):
    return """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · LLM 全栈工程</title>
<link rel="stylesheet" href="{rel}assets/style.css">
</head>
<body>
<header class="site-header"><div class="wrap">
<a class="brand" href="{rel}index.html">LLM 全栈工程<small>LLM 工业级全栈工程落地手册 · 第 00 章 + 主线 46 章</small></a>
<nav><a href="{rel}index.html">目录</a><a href="https://github.com/fenglinxin/llm-fullstack-handbook">GitHub 仓库</a></nav>
</div></header>
<main class="wrap">
<nav class="crumbs"><a href="{rel}index.html">全部章节</a> / {title}</nav>
{toc}
<article class="article">
{body}
</article>
<nav class="prevnext">{prev}{next}</nav>
</main>
<footer class="wrap"><p>LLM 全栈工程 · Voice前沿 出品 · 47 篇 · 收藏即手册</p></footer>
</body>
</html>
""".format(
        title=html_mod.escape(title),
        rel=rel,
        toc=toc,
        body=body,
        prev=prev_html,
        next=next_html,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CH_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(MD_DIR.glob("*.md"), key=chapter_no)
    chapters = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        title = "章节"
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        chapters.append({"no": chapter_no(f), "title": title, "md": f})

    for idx, ch in enumerate(chapters):
        md = markdown.Markdown(
            extensions=["tables", "fenced_code", "toc", "sane_lists"],
            extension_configs={"toc": {"permalink": False, "toc_depth": "2-3"}},
        )
        body = md.convert(ch["md"].read_text(encoding="utf-8"))
        toc = ""
        if md.toc and "toc" in md.toc:
            toc = '<details class="toc-box"><summary>本章目录</summary>' + md.toc + "</details>"
        prev_html = next_html = ""
        if idx > 0:
            p = chapters[idx - 1]
            prev_html = '<a href="ch%02d.html">← 第 %02d 章 %s</a>' % (p["no"], p["no"], html_mod.escape(p["title"]))
        if idx < len(chapters) - 1:
            n = chapters[idx + 1]
            next_html = '<a href="ch%02d.html" style="text-align:right">第 %02d 章 %s →</a>' % (n["no"], n["no"], html_mod.escape(n["title"]))
        page = build_page(ch["title"], body, toc, prev_html, next_html, "../")
        out = CH_DIR / ("ch%02d.html" % ch["no"])
        out.write_text(page, encoding="utf-8")
        print("generated", out.name, len(page))

    cards = ""
    for lo, hi, name, desc in STAGES:
        cards += '<h2 class="stage">%s</h2><p style="color:#64748b;margin:-4px 0 6px">%s</p><div class="ch-grid">' % (name, desc)
        for ch in chapters:
            if lo <= ch["no"] <= hi:
                cards += ('<a class="ch-card" href="chapters/ch%02d.html">'
                          '<div class="no">第 %02d 章</div><div class="tt">%s</div></a>'
                          % (ch["no"], ch["no"], html_mod.escape(ch["title"])))
        cards += "</div>"

    index = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM 全栈工程 · LLM 工业级全栈工程落地手册</title>
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<header class="site-header"><div class="wrap">
<a class="brand" href="index.html">LLM 全栈工程<small>LLM 工业级全栈工程落地手册 · 第 00 章 + 主线 46 章</small></a>
<nav><a href="index.html">目录</a><a href="https://github.com/fenglinxin/llm-fullstack-handbook">GitHub 仓库</a></nav>
</div></header>
<div class="hero"><div class="wrap">
<h1>LLM 全栈工程</h1>
<p>从 0 到 1 打通 LLM 预训练 · 微调对齐 · 部署推理 · 编译器极致优化：原理白话 · 选型对比 · 分步实操 · 避坑手册</p>
<div class="badges">
<span class="badge">47 篇</span><span class="badge">数据核心</span><span class="badge">微调对齐</span>
<span class="badge">部署推理</span><span class="badge">编译器优化</span><span class="badge">全栈落地</span>
</div>
</div></div>
<main class="wrap">
{cards}
</main>
<footer class="wrap"><p>LLM 全栈工程 · Voice前沿 出品 · 单篇独立可读，全套即手册</p></footer>
</body>
</html>
""".format(cards=cards)
    (OUT_DIR / "index.html").write_text(index, encoding="utf-8")
    (OUT_DIR / ".nojekyll").write_text("", encoding="utf-8")
    print("index generated", len(index), "chapters:", len(chapters))


if __name__ == "__main__":
    main()
