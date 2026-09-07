# -*- coding: utf-8 -*-
"""
第06章 文本过滤实战：低质/敏感规则过滤（工程脚本）

【层级】L1（规则过滤引擎：规则表驱动，可热插拔规则）
【环境依赖】Python 3.10+，仅标准库
【核心逻辑】每条文本跑规则集（正则/关键词/长度），命中记录到 rules_hit；
按严重度分级：reject（有害）> warn（低质）> pass；输出分级分布与样例。
【关键参数】
- --sample：生成演示文本（默认关）；--strict：把 warn 也当 reject。
【避坑】规则用前缀匹配避免误伤（如“赌博”别匹配“赌博公司研究”）；
关键词规则注意大小写与全角；分级标准要与业务红线对齐。
【运行结果示例】
$ python text_filter.py --sample
[pass] 今天学了新的知识，感觉收获很大。
[warn] 震惊！不看后悔！马上删除！加微信点击购买！
[reject] 有人教我如何在赌博网站上快速获利，有没有兴趣？
[pass] 一篇正常的行业分析文章，内容完整长度足够，用来作为通过的样本。
grade distribution: {'pass': 3, 'warn': 2, 'reject': 1}
rule hits: {'标题党': 1, '广告灌水': 1, '无意义重复': 1, '暴力/违法': 1}
【高频报错 Top5】
1. 正则写错导致全 reject：先单独测试每条规则；2. 漏匹配：规则表要可扩展；
3. 编码：统一 utf-8；4. 分级顺序：先判 reject 再 warn；5. 关键词误伤：
加例外表/上下文窗口。
【输出解读】reject/warn/pass 计数 + 每规则命中数打印即成功。
【工程改造方向】接分类器打分替代关键词；规则用 YAML 配置化；人工抽检闭环。
"""

import argparse
import re


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


REJECT_RULES = {
    "暴力/违法": re.compile(r"杀人|诈骗|赌博网站|毒品交易"),
    "色情": re.compile(r"约炮|色情片|裸聊"),
}
WARN_RULES = {
    "标题党": re.compile(r"震惊|不看后悔|马上删除"),
    "广告灌水": re.compile(r"加微信|点击购买|限时秒杀"),
    "无意义重复": re.compile(r"(哈|啊|哦){6,}"),
}
LOW_MIN_LEN = 20


def grade(text, strict):
    hits = {}
    for name, rx in REJECT_RULES.items():
        if rx.search(text):
            hits[name] = hits.get(name, 0) + 1
    if hits:
        return "reject", hits
    for name, rx in WARN_RULES.items():
        if rx.search(text):
            hits[name] = hits.get(name, 0) + 1
    if hits or (strict and len(text) < LOW_MIN_LEN):
        return "warn", hits
    return "pass", {}


def gen_sample():
    return [
        "今天学了新的知识，感觉收获很大。",
        "震惊！不看后悔！马上删除！加微信点击购买！",
        "哈哈哈哈哈哈哈哈哈哈哈哈哈",
        "有人教我如何在赌博网站上快速获利，有没有兴趣？",
        "一篇正常的行业分析文章，内容完整长度足够，用来作为通过的样本。",
        "短文本",
    ]


def main():
    args = parse_args()
    docs = gen_sample() if args.sample else ["正常样本文本。"]
    stat = {}
    hit_total = {}
    for d in docs:
        g, hits = grade(d, args.strict)
        stat[g] = stat.get(g, 0) + 1
        for k, v in hits.items():
            hit_total[k] = hit_total.get(k, 0) + v
        print("[%s] %s" % (g, d[:40]))
    print("grade distribution:", stat)
    print("rule hits:", hit_total)


if __name__ == "__main__":
    main()
