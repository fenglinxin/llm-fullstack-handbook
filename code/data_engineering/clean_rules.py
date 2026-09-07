# -*- coding: utf-8 -*-
"""
第05章 数据清洗规范与脏数据剔除（工程脚本）

【层级】L1（规则清洗：归一化/去噪/去短/去乱码，可扩展成清洗流水线）
【环境依赖】Python 3.10+，仅标准库
【核心逻辑】清洗顺序：全角转半角->去控制符->URL/邮箱占位->压缩空白->
过滤空行/过短/重复字符爆炸行；输出逐条清洗前后对比与剔除统计。
【关键参数】
- --sample：生成演示脏文本（默认关）；--min_len 10（默认）：最短保留长度；
- --repeat_ratio 0.5（默认）：重复字符占比阈值，超过判乱码。
【避坑】规则顺序影响结果（先归一化再过滤）；URL 替换用占位符保留位置信息；
中文全角转半角要处理 ０-９ 与标点两套码表。
【运行结果示例】
$ python clean_rules.py --sample
raw 6 -> cleaned:
  [keep] '这是 一 条 正常 的 中文 文本，用来测试清洗规则。' -> '这是 一 条 正常 的 中文 文本,用来测试清洗规则。'
  [keep] 'ｈｔｔｐｓ：／／ｅｘａｍｐｌｅ．ｃｏｍ 全角乱入' -> '[URL] 全角乱入'
  [keep] '联系我：foo@bar.com 谢谢' -> '联系我:[EMAIL] 谢谢'
  [drop_repeat] '哈哈哈哈哈哈哈哈哈哈...' -> 同原文
  [drop_too_short] '短' -> '短'
  [drop_too_short] '空行\n\n\n多余换行' -> '空行 多余换行'
summary: {'keep': 3, 'drop_repeat': 1, 'drop_too_short': 2}
【高频报错 Top5】
1. 空输入报 0 条：目录/样例没生成；2. 乱码没被剔：repeat_ratio 阈值调低；
3. 全角转半角漏字符：码表不全，用 str.translate 字典补齐；4. 过滤太狠误删正文：
先跑报告看分布再定阈值；5. 编码错误：read 时 errors=ignore。
【输出解读】kept/dropped 计数与示例行前后对比打印即成功。
【工程改造方向】接入 pandas 批处理、保存清洗前后版本用于 A/B 质检。
"""

import argparse
import pathlib
import re


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true")
    p.add_argument("--min_len", type=int, default=10)
    p.add_argument("--repeat_ratio", type=float, default=0.5)
    return p.parse_args()


import unicodedata

def normalize_text(text):
    """NFKC：全角字母/数字/标点归一化为半角（比手写码表稳）。"""
    return unicodedata.normalize("NFKC", text)


def clean(text, min_len, repeat_ratio):
    t = normalize_text(text)
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", t)
    t = re.sub(r"https?://\S+|www\.\S+", "[URL]", t)
    t = re.sub(r"[\w.+-]+@[\w.-]+", "[EMAIL]", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) < min_len:
        return t, "drop_too_short"
    if not re.search(r"[\u4e00-\u9fff\w]", t):
        return t, "drop_no_text"
    if t and max(t.count(c) for c in set(t)) / len(t) > repeat_ratio:
        return t, "drop_repeat"
    return t, "keep"


def gen_sample():
    return [
        "这是 一 条 正常 的 中文 文本，用来测试清洗规则。",
        "ｈｔｔｐｓ：／／ｅｘａｍｐｌｅ．ｃｏｍ 全角乱入",
        "联系我：foo@bar.com 谢谢",
        "哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈",
        "短",
        "空行\n\n\n多余换行",
    ]


def main():
    args = parse_args()
    samples = gen_sample() if args.sample else [
        "这是 一 条 正常 的 中文 文本，用来测试清洗规则。"]
    stats = {}
    print("raw %d -> cleaned:" % len(samples))
    for s in samples:
        out, tag = clean(s, args.min_len, args.repeat_ratio)
        stats[tag] = stats.get(tag, 0) + 1
        print("  [%s] %r -> %r" % (tag, s[:30], out[:40]))
    print("summary:", stats)


if __name__ == "__main__":
    main()