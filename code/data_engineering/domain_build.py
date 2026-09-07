# -*- coding: utf-8 -*-
"""
第10章 领域专属数据构建：领域语料 + 模板 QA 合成（工程脚本）

【层级】L2（领域构建工具：从领域 KB 生成语料与合成 QA，可接入数据管道）
【环境依赖】Python 3.10+，仅标准库
【核心逻辑】领域 KB（句子+问答三元组）-> 领域语料落盘 + QA 模板扩展
（每个问答按模板变体生成多条）-> 输出统计（KB 规模/合成倍数/字符量）。
【关键参数】
- --domain demo（默认）：领域名；--multiplier 3（默认）：每条 QA 模板变体数；
- --out_dir _tmp/domain_out（默认）：输出目录。
【避坑】模板变体要控制重复度（同义改写不要超过 N 倍），否则制造重复数据；
合成 QA 需人工抽检正确性；与通用语料混合时注意配比。
【运行结果示例】
$ python domain_build.py --sample
domain=demo kb_sentences=2 kb_qa=2 synthesized_qa=6 chars=138 files=domain_corpus.txt, qa.jsonl
【高频报错 Top5】
1. 输出目录不存在：程序自动创建（已做）；2. 模板占位符不匹配：用 format 前校验键；
3. 合成重复度过高：multiplier 调小或加去重（ch07 脚本）；
4. 中文编码：统一 utf-8；5. 与 SFT 格式不兼容：输出字段名 q/a 已对齐 sft_data.py。
【输出解读】看到 kb/qa 数量与合成总数即成功；目录下 domain_corpus.txt 与 qa.jsonl 已生成。
【工程改造方向】接领域文档采集（ch04）+清洗（ch05）+质量分（ch09）形成领域数据流水线。
"""

import argparse
import itertools
import json
import pathlib


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true")
    p.add_argument("--domain", default="demo")
    p.add_argument("--multiplier", type=int, default=3)
    p.add_argument("--out_dir", default="_tmp/domain_out")
    return p.parse_args()


KB = {
    "sentences": [
        "LLM 推理时 KV Cache 缓存历史键值以降低单步延迟。",
        "量化把权重从 FP16 压到 INT8 可以省显存。",
    ],
    "qa": [
        {"q": "什么是 KV Cache？", "a": "KV Cache 是缓存历史 token 键值、避免重复计算的技术。"},
        {"q": "量化有什么好处？", "a": "量化可以减小模型体积并降低显存占用，代价是精度损失。"},
    ],
}
TEMPLATES = [
    "{q}", "请简要回答：{q}", "帮我解释一下：{q}",
]


def main():
    args = parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # 领域语料：KB 句子 + QA 文本
    corpus = list(KB["sentences"]) + [qa["q"] + qa["a"] for qa in KB["qa"]]
    corpus_path = out / "domain_corpus.txt"
    corpus_path.write_text("\n".join(corpus) + "\n", encoding="utf-8")
    # 合成 QA：模板变体
    rows = []
    for qa in KB["qa"]:
        for tmpl in TEMPLATES[:args.multiplier]:
            rows.append({"q": tmpl.format(q=qa["q"]), "a": qa["a"],
                         "domain": args.domain, "source": "template"})
    qa_path = out / "qa.jsonl"
    with open(qa_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    chars = sum(len(s) for s in corpus)
    print("domain=%s kb_sentences=%d kb_qa=%d synthesized_qa=%d chars=%d files=%s, %s"
          % (args.domain, len(KB["sentences"]), len(KB["qa"]), len(rows), chars,
             corpus_path.name, qa_path.name))


if __name__ == "__main__":
    main()
