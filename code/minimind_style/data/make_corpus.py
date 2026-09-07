# -*- coding: utf-8 -*-
"""
生成微型中文语料（CPU 分钟级预训练用）

【层级】L1（数据工具：极简可跑，直接生成教学数据文件）
【环境依赖】
- Python 3.10+；零第三方依赖（仅标准库）
【核心逻辑】
- 内置 41 句工程格言 + 6 条 QA，按行写入 data/corpus.txt；
- 字符级 tokenizer 在 pretrain.py 中 fit 本文件，换语料=换文件后重跑 pretrain。
【关键参数】（默认值 / 适配场景）
- lines：内置句子列表（想换语料直接改这里）；
- 输出位置固定为 minimind_style/data/corpus.txt（pretrain.py 默认读取）。
【避坑】
1. 新增句子含生僻字符时，pretrain 会自动扩词表，无需手动维护；
2. 不要手动编辑 corpus.txt 后又改本文件，两者会不同步。
【运行结果示例】（真实运行）
$ python data/make_corpus.py
corpus chars: 925 -> .../data/corpus.txt
【高频报错 Top5】
1. FileNotFoundError：请从 minimind_style 目录运行或用绝对路径；
2. UnicodeEncodeError：Windows 控制台用 PYTHONIOENCODING=utf-8；
3. 运行后 pretrain 报 UNK：词表未重训；修复：重跑 pretrain.py（它会重新 fit）；
4. 想加样本却忘保存：改 lines 后重跑本脚本；
5. 文件被占用无法写入：关闭打开着 corpus.txt 的编辑器。
【输出解读】
- 看到“-> 文件路径”且行数/字数与示例一致 = 生成成功；
- 每次修改本文件后必须重跑，训练脚本才会读到新数据。
【工程改造方向】
- 换真实语料：读 txt/md 文件列表拼接后写入；
- 加数据版本号字段，配合“模型版本与数据版本记录在案”的主线实践。
"""
