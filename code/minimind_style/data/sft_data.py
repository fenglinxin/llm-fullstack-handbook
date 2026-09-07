# -*- coding: utf-8 -*-
"""
生成微型 SFT 数据（data/sft.jsonl，与语料同一套字符，保证 tokenizer 覆盖）

【层级】L1（数据工具：极简可跑）
【环境依赖】Python 3.10+，仅标准库（json/pathlib）
【核心逻辑】内置 6 条 QA（q 含“？”，a 为语料中的标准答案），按 jsonl 写出；
train_sft.py 用字符偏移 mask 只对 a 部分算 loss。
【关键参数】PAIRS：问答对列表（默认 6 条；想加样本直接追加，字符会自动进词表）
【避坑】q/a 里的字符若不在 corpus.txt 会以 UNK 出现；新增前先在 corpus 中确认
【运行结果示例】$ python data/sft_data.py → sft pairs: 6 -> .../data/sft.jsonl
【高频报错 Top5】
1. jsonl 找不到：从 minimind_style 目录运行；2. 中文乱码：加 PYTHONIOENCODING=utf-8；
3. 答案首字是 UNK：该字符不在词表；4. 训练不收敛先重跑 make_corpus+pretrain；
5. 行数不对：确认 main() 被调用（__main__ 保护）。
【输出解读】打印 “sft pairs: 6” 即成功；train_sft.py 启动时也会打印样本数核对。
【工程改造方向】换成自己的指令数据（jsonl 同构即可）；加 system/多轮字段见 tool_template_demo。
"""
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
