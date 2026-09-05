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
