# -*- coding: utf-8 -*-
"""
生成 MiniMind-O 教学数据集（data/omni.jsonl）

【层级】L1（数据工具：极简可跑）
【环境依赖】Python 3.10+，仅标准库；依赖 vision_model/omni_model 的常量
【核心逻辑】audio：3 档音调 x 5 噪声实例；vision：8 图案 x 5 实例；
划分 noise_seed 0-2 训练 / 3-4 测试。
【关键参数】SEEDS=range(5)（默认即最优；想加难度可加噪声幅度参数）
【避坑】freq 与 TONE_NAMES 索引一一对应，别改顺序；改完重跑
【运行结果示例】$ python data/make_omni_data.py
omni rows: 55 (audio 15, vision 40)
【高频报错 Top5】
1. ModuleNotFoundError：从 minimind_style 根目录运行；2. 中文乱码：utf-8；
3. 行数不对：改了 SEEDS 没重跑；4. train_o.py 精度不升：确认两个 jsonl 都最新；
5. 索引错位导致“低音”答成“高音”：检查 TONE_NAMES 顺序。
【输出解读】rows=55（audio 15 + vision 40）即成功。
【工程改造方向】加真实音频/图片路径列；加第三模态（文本）字段。
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from vision_model import CAPTIONS  # noqa: E402
from omni_model import TONE_FREQS, TONE_NAMES  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "omni.jsonl"
SEEDS = list(range(5))


def main():
    rows = []
    for i, freq in enumerate(TONE_FREQS):
        for seed in SEEDS:
            rows.append({"modality": "audio", "freq": freq, "noise_seed": seed,
                         "q": "这是什么音调？",
                         "a": "这是" + TONE_NAMES[i] + "音调。"})
    for pid, caption in enumerate(CAPTIONS):
        for seed in SEEDS:
            rows.append({"modality": "vision", "pid": pid, "noise_seed": seed,
                         "q": "图中有什么？", "a": caption})
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    audio_n = sum(1 for r in rows if r["modality"] == "audio")
    print("omni rows: %d (audio %d, vision %d)" % (len(rows), audio_n,
                                                   len(rows) - audio_n))
    print("->", OUT)


if __name__ == "__main__":
    main()
