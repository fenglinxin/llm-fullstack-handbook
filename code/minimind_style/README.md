# MiniMind 式微型 LLM（minimind_style）

> 参考 https://github.com/jingyaogong/minimind 的定位：大道至简，纯 PyTorch 从 0 训练微型 LLM。
> 本目录是教学最小闭环，不依赖 transformers/trl/peft。

## 快速开始

~~~bash
pip install torch
python pretrain.py --steps 200   # CPU 1-3 分钟
python generate.py --prompt "人工智能"
~~~

## 文件说明

| 文件 | 作用 |
|---|---|
| model.py | TinyGPT：RMSNorm + RoPE + 因果注意力 + FFN（逐行注释） |
| tokenizer.py | 字符级分词器（fit/encode/decode/save/load） |
| data/make_corpus.py | 生成微型中文语料 data/corpus.txt |
| pretrain.py | 微型预训练：loss 监控 + 周期生成样例 + 保存 checkpoint |
| generate.py | 加载 checkpoint 做温度/top-k 采样生成 |

## 预期输出

- loss 从约 4-5 降到 0.1-0.5（语料小，会快速过拟合）；
- 生成会先出现语料中的词片段，随后趋于重复——这是小语料过拟合的正常现象；
- 想看到更自然文本：换更大语料并加 early stop（主线 04-13 章方法）。

## 下一步（任务书 P1/P2/P3）

- P1：SFT 指令微调（chat template + QA 数据）；
- P2：LoRA 与 DPO 最小实现；
- P3：MoE 变体、RLHF/GRPO、工具/思考模板。

## 进阶改造 Prompt

1. 把字符分词换成 BPE，对比词表大小与生成质量；
2. 加大语料并加入 warmup/cosine/梯度裁剪；
3. 加入 checkpoint 断点续训与日志可视化；
4. 给注意力加 KV Cache，对比生成速度；
5. 把 FFN 换成 MoE 并观察负载均衡。
