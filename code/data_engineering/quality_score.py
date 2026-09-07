# -*- coding: utf-8 -*-
"""
第09章 数据质量打分体系：多维度规则评分（工程脚本）

【层级】L3（可扩展质量评分器：规则权重表驱动，可接分类器替代规则分）
【环境依赖】Python 3.10+，仅标准库（math/collections）
【核心逻辑】五维评分（0-100 加权）：
长度充足度、标点/结构合理性、字符熵（信息密度）、重复惩罚、
URL/邮箱占比惩罚；最终分=加权和，输出分布与最优/最差样例。
【关键参数】
- --sample：生成演示文档（默认关）；--top 3（默认）：展示最好/最差条数；
- 权重表 WEIGHTS（默认 0.3/0.2/0.3/0.1/0.1）：可按业务调。
【避坑】熵对短文本偏高，需先乘长度权重；重复惩罚对正式公文要放宽；
评分只做排序不做绝对结论，阈值（如 60 分及格）需抽样校准。
【运行结果示例】
$ python quality_score.py --sample
docs=5 mean=32.2 best=77.1 worst=0.0
  top 77.1 这是一篇结构完整、长度足够的行业分析文章，讨论了
  top 45.1 加微信加微信加微信点击购买点击购买
  low 16.1 短
  low 0.0（空文本 0 分）
【高频报错 Top5】
1. 空文本除零：len=0 直接 0 分；2. 全角标点没算进结构分：用 NFKC 归一化后再统计；
3. 权重和不为 1：校验 WEIGHTS 总和；4. 中文熵算错：按字符频率算 log2；
5. 抽样校准：先打印分布再定及格线。
【输出解读】mean/分位 + 最好最差样例打印即成功；分布偏左说明数据整体低质。
【工程改造方向】规则分换成分类器/嵌入打分并做在线校准；输出 Excel 报告。
"""

import argparse
import collections
import math


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true")
    p.add_argument("--top", type=int, default=3)
    return p.parse_args()


WEIGHTS = {"len": 0.3, "struct": 0.2, "entropy": 0.3, "repeat": 0.1, "noise": 0.1}


def entropy(text):
    cnt = collections.Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in cnt.values())


def score_doc(text):
    if not text:
        return 0.0, {"len": 0, "struct": 0, "entropy": 0, "repeat": 0, "noise": 0}
    L = len(text)
    s_len = min(100.0, L / 2.0)                     # 50 字封顶
    punct = sum(ch in "，。！？；：、,." for ch in text)
    s_struct = min(100.0, punct / max(1, L / 20) * 100) if punct else 30.0
    ent = entropy(text)
    s_ent = min(100.0, ent / math.log2(30) * 100)   # 30 字符集为参照
    top = max(collections.Counter(text).values())
    s_repeat = max(0.0, 100 - 100 * (top / max(1, L) - 0.15) / 0.6)
    url_noise = (text.count("http") + text.count("@")) * 5
    s_noise = max(0.0, 100 - url_noise)
    parts = {"len": s_len, "struct": s_struct, "entropy": s_ent,
             "repeat": s_repeat, "noise": s_noise}
    total = sum(parts[k] * w for k, w in WEIGHTS.items())
    return total, parts


def gen_sample():
    return [
        "这是一篇结构完整、长度足够的行业分析文章，讨论了模型训练与部署的关键问题。",
        "哈哈哈哈哈啊哈哈哈啊啊哈哈哈",
        "加微信加微信加微信点击购买点击购买",
        "短",
        "",
    ]


def main():
    args = parse_args()
    docs = gen_sample() if args.sample else ["示例文本"]
    scored = sorted(((score_doc(d)[0], d) for d in docs), reverse=True)
    mean = sum(s for s, _ in scored) / max(1, len(scored))
    print("docs=%d mean=%.1f best=%s worst=%s"
          % (len(scored), mean,
             scored[0][0] if scored else 0,
             scored[-1][0] if scored else 0))
    for s, d in scored[:args.top]:
        print("  top %.1f %s" % (s, d[:24]))
    for s, d in scored[-args.top:]:
        print("  low %.1f %s" % (s, d[:24]))
    print("weights:", WEIGHTS)


if __name__ == "__main__":
    main()
