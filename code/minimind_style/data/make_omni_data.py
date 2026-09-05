# -*- coding: utf-8 -*-
"""生成 MiniMind-O 教学数据集（data/omni.jsonl）。

每行：{modality, freq 或 pid, noise_seed, q, a}
- audio：三档音调（440/660/880Hz）x 5 个噪声实例；
- vision：8 种图案 x 5 个噪声实例（与 vision.jsonl 同一套图案渲染）。
- 划分：noise_seed 0-2 训练，3-4 测试（没见过的“听感/图像实例”）。
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
