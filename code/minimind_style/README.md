# MiniMind 式微型 LLM（minimind_style）

> 参考 https://github.com/jingyaogong/minimind 的定位：大道至简，纯 PyTorch 从 0 训练微型 LLM。
> 本目录是教学最小闭环，不依赖 transformers/trl/peft。P0（预训练）+ P1（SFT）+ P2（LoRA/DPO）均已跑通。

## 快速开始

~~~bash
pip install torch
python pretrain.py --steps 400     # P0 预训练，CPU 1-3 分钟
python generate.py --prompt "人工智能"
python train_sft.py --epochs 60    # P1 SFT（先跑上面的预训练）
python train_lora.py --epochs 60   # P2a 用 LoRA 重跑 SFT（基座 = out/sft.pt）
python data/dpo_data.py
python train_dpo.py                # P2b 偏好对齐（基座 = out/sft.pt）
~~~

## 文件说明

| 文件 | 作用 |
|---|---|
| model.py | TinyGPT：RMSNorm + RoPE + 因果注意力 + FFN（逐行注释） |
| tokenizer.py | 字符级分词器（fit/encode/decode/save/load） |
| data/make_corpus.py | 生成微型中文语料 data/corpus.txt（含末尾 6 条 QA） |
| data/sft_data.py | 生成 6 条 SFT 样本 data/sft.jsonl |
| data/dpo_data.py | 生成 6 条偏好样本 data/dpo.jsonl（chosen/rejected） |
| pretrain.py | 微型预训练：loss 监控 + 周期生成样例 + 保存 checkpoint |
| train_sft.py | 微型 SFT：回答部分算 loss（response-only mask），指令部分屏蔽 |
| lora.py | LoRALinear + apply_lora/merge_lora/count_params（冻结基座只训 A/B） |
| train_lora.py | 用 LoRA 重跑 SFT，保存 adapter（out/lora.pt）与合并模型（out/lora_merged.pt） |
| train_dpo.py | 手写 DPO loss：policy vs 冻结 ref，chosen/rejected 平均 logp 做 margin |
| generate.py | 加载 checkpoint 做温度/top-k 采样生成 |

## 预期输出（本机实测）

- P0：loss 从约 ln(vocab)≈5.9 降到 0.02-0.1（语料仅 925 字符，属正常过拟合）；
- P0 生成：能复现语料句子片段，如「人工智能正在改变世界。大语言模型通过预测下一个词来学习」；
- P1 SFT：loss 从约 1.0 降到 0.15，按训练格式提问可背出答案开头，如：

~~~text
问：打印机连不上怎么办？答：先检查电源和网络，然后重启打印机，并提交工单。
~~~

- P2a LoRA：总参数 161,984，可训练 16,384（约 10%，微型模型骨干占比高）；
  loss 从约 1.0 降到 0.08，合并后模型与训练中行为一致；
- P2b DPO：默认参数（lr=1e-5）下 mean margin ≈ 1.45 → 2.38，贪心生成保持完整；
  把 lr 调大（如 5e-4）margin 会冲到 8+，但生成崩坏——这是小样本 DPO
  的 KL 漂移/过优化现象，教学上建议故意对比一次；
- 长回复随后会混乱/串词——6 条样本的玩具模型只能做到“背题级”，教学目的已达成；
- 想看到更自然文本：换更大语料并加 early stop / 评测集（主线 04-13 章方法）。

## 避坑（真实踩过）

1. **next-token 标签必须左移一位**：loss 里若用当前 token 当标签，
   模型会退化成“抄当前位置的字符”，loss 假性归零、生成却只会无限复读最后一个字；
2. 指令部分若不加 mask，模型会学“背问题”，SFT 质量变差；
3. 语料扩词表后必须重跑 pretrain.py，否则新字符（问/答/：）全是 UNK；
4. LoRA 合并时 ΔW=A@B 是 [in,out]，写回 nn.Linear.weight（[out,in]）要转置；
5. DPO 的 ref 模型必须冻结；小样本 + 大 lr 会过优化，警惕“loss 归零但生成变差”；
6. DPO 的 chosen/rejected 必须避免“A 的 rejected 恰好是 B 的 chosen”这类自相矛盾偏好。

## 下一步（任务书 P3）

- P3：MoE 变体、RLHF/GRPO、工具/思考模板（远期，参考 MiniMind 演进路线）。

## 进阶改造 Prompt

1. 把字符分词换成 BPE，对比词表大小与生成质量；
2. 加大语料并加入 warmup/cosine/梯度裁剪；
3. 加入 checkpoint 断点续训与日志可视化；
4. 给注意力加 KV Cache，对比生成速度；
5. 把 FFN 换成 MoE 并观察负载均衡；
6. LoRA 只打注意力层 vs 只打 FFN 层，对比效果与可训练参数；
7. DPO 换整段 logp 求和 + 真实偏好数据，观察 KL 控制。
