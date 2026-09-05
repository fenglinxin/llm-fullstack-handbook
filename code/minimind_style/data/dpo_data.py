# -*- coding: utf-8 -*-
"""生成微型 DPO 数据（chosen=真实答案，rejected=串词/错误答案）。

【约束】rejected 只使用语料中已出现的字符，保证 tokenizer 不会把
rejected 变成 UNK（UNK 位置无法提供有效梯度）。
"""
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "dpo.jsonl"

# q 与 sft_data.py 完全一致；rejected 故意选“同语料字符但答非所问”的句子
PAIRS = [
    {"q": "打印机连不上怎么办？",
     "chosen": "先检查电源和网络，然后重启打印机，并提交工单。",
     "rejected": "重启打印机并联系管理员。"},
    {"q": "无线网络连不上怎么办？",
     "chosen": "先确认连接公司无线网络，再重置网络。",
     "rejected": "等待半小时后重试或联系管理员。"},
    {"q": "账号被锁怎么办？",
     "chosen": "等待半小时后重试或联系管理员。",
     "rejected": "清理大附件并删除旧邮件。"},
    {"q": "邮箱满了怎么办？",
     "chosen": "清理大附件并删除旧邮件。",
     "rejected": "重启打印机并提交工单。"},
    {"q": "什么是人工智能？",
     "chosen": "人工智能是让机器模拟人类智能的技术。",
     "rejected": "人工智能正在改变世界。"},
    {"q": "如何提高模型质量？",
     "chosen": "提高数据质量并做好评测。",
     "rejected": "先通后优是工程落地的重要原则。"},
]

def main():
    with open(OUT, "w", encoding="utf-8") as f:
        for item in PAIRS:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("dpo pairs:", len(PAIRS), "->", OUT)

if __name__ == "__main__":
    main()
