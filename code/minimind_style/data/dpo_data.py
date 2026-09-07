# -*- coding: utf-8 -*-
"""
生成微型 DPO 数据（data/dpo.jsonl：chosen=真实答案，rejected=串词/错误答案）

【层级】L1（数据工具：极简可跑）
【环境依赖】Python 3.10+，仅标准库（json/pathlib）
【核心逻辑】每行 {q, chosen, rejected}；rejected 只使用语料中已出现字符，
避免 tokenizer 把 rejected 变成 UNK（UNK 位置没有学习信号）。
【关键参数】PAIRS：6 条偏好对（默认值即为教学最优值；真实项目换大偏好数据集）
【避坑】rejected 不要引用其他问题的 chosen，会形成自相矛盾的偏好信号
【运行结果示例】$ python data/dpo_data.py → dpo pairs: 6 -> .../data/dpo.jsonl
【高频报错 Top5】
1. jsonl 找不到：从 minimind_style 目录运行；2. 中文乱码：加 PYTHONIOENCODING=utf-8；
3. 训练时 rejected logp 不下降：字符在词表外；4. margin 恒为 0：policy 与 ref 同起点正常；
5. 改 PAIRS 后没重跑：train_dpo.py 读的是 jsonl。
【输出解读】打印 “dpo pairs: 6” 即成功。
【工程改造方向】换真实偏好数据（hh-rlhf 风格）；加多人标注/投票字段。
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
