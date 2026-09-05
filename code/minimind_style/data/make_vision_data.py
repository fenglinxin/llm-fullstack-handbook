# -*- coding: utf-8 -*-
"""生成 MiniMind-V 教学数据集（data/vision.jsonl）。

每行：{pid, noise_seed, q, a}
- pid：8 种基础图案（角块/横线/竖线/十字/边框）；
- noise_seed：同一图案的不同“图像实例”（随机位置加一粒噪声点）。
- 划分：训练用每图案 noise_seed 0-2，测试用 noise_seed 3-4——
  测试都是“见过的图案、没见过的图像实例”，只有真正学到
  “图案概念”的模型才能答对，纯背图/背答案都会在测试集露馅。
- q/a：看图问答的文本（q 不含“问：/答：”外壳，训练时再拼模板）。
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from vision_model import CAPTIONS  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "vision.jsonl"
QUESTION = "图中有什么？"
PIDS = list(range(len(CAPTIONS)))
TRAIN_SEEDS = [0, 1, 2]
TEST_SEEDS = [3, 4]


def main():
    rows = []
    for pid in PIDS:
        for noise_seed in TRAIN_SEEDS + TEST_SEEDS:
            rows.append({"pid": pid, "noise_seed": noise_seed,
                         "q": QUESTION, "a": CAPTIONS[pid]})
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("vision rows: %d (train seeds %s, test seeds %s)"
          % (len(rows), TRAIN_SEEDS, TEST_SEEDS))
    print("->", OUT)


if __name__ == "__main__":
    main()
