# -*- coding: utf-8 -*-
"""生成微型 SFT 数据（与语料同一套字符，保证 tokenizer 覆盖）。"""
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "sft.jsonl"

PAIRS = [
    {"q": "打印机连不上怎么办？", "a": "先检查电源和网络，然后重启打印机，并提交工单。"},
    {"q": "无线网络连不上怎么办？", "a": "先确认连接公司无线网络，再重置网络。"},
    {"q": "账号被锁怎么办？", "a": "等待半小时后重试或联系管理员。"},
    {"q": "邮箱满了怎么办？", "a": "清理大附件并删除旧邮件。"},
    {"q": "什么是人工智能？", "a": "人工智能是让机器模拟人类智能的技术。"},
    {"q": "如何提高模型质量？", "a": "提高数据质量并做好评测。"},
]

def main():
    with open(OUT, "w", encoding="utf-8") as f:
        for item in PAIRS:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("sft pairs:", len(PAIRS), "->", OUT)

if __name__ == "__main__":
    main()
