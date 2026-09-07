# -*- coding: utf-8 -*-
"""
生成 MiniMind-V 教学数据集（data/vision.jsonl）

【层级】L1（数据工具：极简可跑）
【环境依赖】Python 3.10+，仅标准库；需同目录上级的 vision_model.py（CAPTIONS）
【核心逻辑】每行 {pid, noise_seed, q, a}；训练用 noise_seed 0-2、测试 3-4，
测试全是“见过的图案、没见过的图像实例”，防止纯背图。
【关键参数】TRAIN_SEEDS=[0,1,2]、TEST_SEEDS=[3,4]（默认即最优：40 行=8 图案 x 5 实例）
【避坑】noise_seed 直接传给 render_pattern，两者必须同源；改 CAPTIONS 后重跑
【运行结果示例】$ python data/make_vision_data.py
vision rows: 40 (train seeds [0, 1, 2], test seeds [3, 4])
【高频报错 Top5】
1. ModuleNotFoundError vision_model：从 minimind_style 根目录运行；
2. 中文乱码：PYTHONIOENCODING=utf-8；3. train_v.py 报样本数少：先重跑本脚本；
4. 测试集 acc 为 0 是正常教学结果（未见图案泛化难）；5. 忘加 noqa 注释导致 lint 报错。
【输出解读】rows=40 且划分正确即成功。
【工程改造方向】换真实图片时改为记录图片路径列；渲染函数换成图像加载。
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
