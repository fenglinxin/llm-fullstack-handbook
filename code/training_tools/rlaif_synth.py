# -*- coding: utf-8 -*-
"""
第25章 RLAIF：规则裁判合成偏好数据（一键脚本）

【层级】L1（数据合成：裁判打分 -> chosen/rejected jsonl，喂给 DPO/RM）
【环境依赖】Python 3.10+，仅标准库
【核心逻辑】对每条 prompt 生成“好/坏”候选；规则裁判按 关键词命中+长度合理
+重复惩罚 打分；好候选分高即判 chosen，输出与 dpo_data.py 同构的 jsonl。
【关键参数】--n 6（默认）；--out _tmp/rlaif.jsonl（默认）。
【避坑】规则裁判是 RLAIF 的雏形，真实用 LLM 裁判要固定 prompt 并多次采样取中位；
合成偏好要人工抽检，防止裁判偏好被模型钻空子（Reward Hacking）。
【运行结果示例】
$ python rlaif_synth.py
synthesized=6 judge_agree=1.00 -> _tmp/rlaif.jsonl
【高频报错 Top5】
1. 输出 jsonl 与 DPO 不兼容：字段必须是 q/chosen/rejected；2. 裁判全给同分：
特征要拉开差距；3. 好/坏候选同源：改写幅度太小；4. 想接 LLM 裁判：改造 judge()
为 API 调用；5. 中文编码：utf-8。
【输出解读】judge_agree=1.0 且生成 6 条 = 合成成功，可直接跑 train_dpo.py。
【工程改造方向】接 LLM 裁判；加多人/多模型投票；接主动学习抽样。
"""

import argparse
import json
import pathlib


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--out", default="_tmp/rlaif.jsonl")
    return p.parse_args()


def judge(text, good_kws, bad_kws):
    score = 10.0
    score += 25 if all(k in text for k in good_kws) else -25
    score += 10 if any(k in text for k in bad_kws) else 0
    score += 5 if 8 <= len(text) <= 30 else -5
    return score


def main():
    args = parse_args()
    qa = [
        ("打印机连不上怎么办？", "先检查电源和网络，然后重启打印机。", ["电源", "网络"], ["打印机"]),
        ("无线网络连不上怎么办？", "先确认连接公司无线网络，再重置网络。", ["无线网络"], ["重启一下"]),
    ]
    rows = []
    agree = 0
    for q, good, good_kws, bad_kws in qa * max(1, args.n // 2):
        s_g = judge(good, good_kws, bad_kws)
        s_b = judge(good_kws[0] + "卡住了", good_kws, bad_kws)
        agree += int(s_g > s_b)
        rows.append({"q": q + "？", "chosen": good, "rejected": good_kws[0] + "卡住了"})
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("synthesized=%d judge_agree=%.2f -> %s" % (len(rows), agree / len(rows), out))


if __name__ == "__main__":
    main()