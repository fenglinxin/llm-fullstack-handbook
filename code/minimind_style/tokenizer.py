# -*- coding: utf-8 -*-
"""
CharTokenizer：MiniMind 式极简字符分词器

【定位】教学用：把文本切成字符级 token；
真实项目建议换 BPE/ByteLevel（参考 MiniMind 的 tokenizer 定位）。

【环境依赖】Python 3.10+（无第三方依赖）

【关键参数】无；词表由语料自动生成。

【避坑】
1. 字符级词表小、训练快，但生成质量上限低；
2. 换语料后必须重新 fit 并保存 vocab.json；
3. 推理时遇到 vocab 外字符用 UNK 兜底。
"""

import json


class CharTokenizer:
    def __init__(self):
        self.vocab = {}
        self.itos = []
        self.unk_id = 0

    def fit(self, texts):
        chars = set()
        for text in texts:
            chars.update(text)
        self.itos = ["<unk>", "<s>", "</s>"] + sorted(chars)
        self.vocab = {c: i for i, c in enumerate(self.itos)}
        self.unk_id = self.vocab["<unk>"]

    def encode(self, text):
        return [self.vocab.get(c, self.unk_id) for c in text]

    def decode(self, ids):
        return "".join(self.itos[i] if 0 <= i < len(self.itos) else "" for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "itos": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls()
        obj.vocab = data["vocab"]
        obj.itos = data["itos"]
        obj.unk_id = obj.vocab.get("<unk>", 0)
        return obj

"""
规范字段补充（全局强制代码落地规范）

【层级】L1（基础工具：字符级分词器，全部脚本共用）
【运行结果示例】（真实运行）
$ python -c "from tokenizer import CharTokenizer; import pathlib; tok=CharTokenizer.load(pathlib.Path('out/vocab.json')); print('vocab:',len(tok.vocab)); print(tok.decode(tok.encode('人工智能')))"
vocab: 367
人工智能
【高频报错 Top5】
1. 词表与 checkpoint 不一致（vocab 数不同）：重跑 pretrain.py 重新 fit+save；
2. 换语料后旧 vocab.json 有 UNK：必须重训 tokenizer；
3. JSON 解析失败：vocab.json 损坏/非 utf-8；修复：删掉重跑 pretrain.py；
4. decode 出现 <unk>：encode 时遇到过词表外字符；修复：先 fit 或加 fallback；
5. Windows 写文件乱码：open 时显式 encoding="utf-8"（本模块已做）。
【工程改造方向】真实项目把本模块换成 BPE/ByteLevel（主线 04-06 章）：
接口 fit/encode/decode/save/load 保持不变，上层无需改动。
"""

"""
规范字段补充 2（补齐缺失字段标记）

【核心逻辑】
- fit(texts)：收集字符集，排序后附 <unk>/<s>/</s> 生成词表；
- encode/decode：字符 <-> id 互转，vocab 外字符一律回退 <unk>；
- save/load：json 持久化词表，供训练/推理脚本共享。
【输出解读】vocab 大小随语料字符数变化；decode(encode(x))==x 要求 x 无 OOV 字符。
"""
