# LLM 全栈工程 · 模型结构落地代码库

> 配套《第 00 章 模型结构核心原理层》的可运行代码引导。**不改动已发布文档**，全部沉淀在本目录。
> 目标：看懂理论 → 跑通代码 → 会改模型 → 懂得选型。

## 目录与速查

| 目录 | 覆盖架构 | 运行文件 | 依赖 |
|---|---|---|---|
| 00_rnn_lstm_gru | RNN / LSTM / GRU | rnn_lstm_gru_demo.py | torch |
| 01_transformer | 手写注意力 + 小型编解码 | attention_from_scratch.py / train_seq2seq_mini.py | torch |
| 02_mamba | SSM 直觉 + Mamba 库接入 | mamba_demo.py | torch（mamba-ssm 可选，Linux+CUDA） |
| 03_moe | 稀疏专家路由 | moe_demo.py | torch |
| 04_sparse_attention | FlashAttention 调用 + 窗口注意力 | flash_attn_demo.py / window_attention.py | torch（flash-attn 可选） |
| 05_audio_vision_multimodal | 音频特征 / 视觉推理 / 多模态融合 | audio_feature_demo.py / vision_inference_demo.py / multimodal_fusion_demo.py | torch / torchaudio / torchvision（可降级） |
| 06_world_model | 简易时序推演（世界模型最小版） | world_model_demo.py | torch |
| minimind_style | MiniMind 式微型 LLM：P0–P3 全链路（预训练/SFT/LoRA/DPO/MoE/GRPO/工具模板） | pretrain.py / train_sft.py / train_lora.py / train_dpo.py / train_moe.py / train_grpo.py / tool_template_demo.py | torch |

## 环境依赖（最小公共集）

- Python 3.10+（3.11 亦可）
- PyTorch 2.x（CPU 即可跑通大部分 Demo）
- 可选加速库：flash-attn（Linux+CUDA）、mamba-ssm（Linux+CUDA）、torchaudio、torchvision

~~~bash
pip install torch numpy
# 可选
pip install torchaudio torchvision
~~~

## 每类模型知识沉淀（3 条核心干货）

### RNN / LSTM / GRU
1. 核心原理：用随时间复用的隐状态建模序列，LSTM 加门控与细胞通道解决长程记忆，GRU 是它的精简版。
2. 核心短板：串行训练无法吃满 GPU、长程依赖仍有限——这是被 Transformer 取代的主因。
3. 适配场景：低算力设备、在线流式预测、轻量时序基线；文本大模型主线请选并行架构。

### Transformer
1. 核心原理：自注意力让每个 token 看全上下文，多头提供多视角，位置编码补顺序。
2. 核心优势：训练可并行、长程建模强、扩展性好，是当前 LLM 的事实底座。
3. 核心短板：注意力 O(n²)+KV Cache 显存随长度增长，长文本成本高。

### Mamba / SSM
1. 核心原理：用固定大小的状态递推建模序列，Mamba 让状态参数随输入选择。
2. 核心优势：O(n) 复杂度、固定推理状态，超长序列/流式友好。
3. 核心短板：生态与 kernel 成熟度仍低于 Transformer，需专门部署适配。

### MoE
1. 核心原理：FFN 换成多个专家，Router 对每个 token 选 Top-k 专家，稀疏激活。
2. 核心优势：参数容量大而单 token 算力可控，是低成本扩容的重要杠杆。
3. 核心痛点：训练 all2all、负载均衡、显存驻留全部专家、推理 EP 路由。

### 稀疏注意力 / FlashAttention
1. 核心原理：稀疏注意力“少看”（窗口/全局锚点），FlashAttention“省着看”（IO 分块+在线 softmax）。
2. 核心优势：长文本显存/带宽显著下降，FlashAttention 数值等价且无需改结构。
3. 核心提醒：稀疏模式可能损失长程信息，收益必须结合 kernel 与任务实测。

### 音频 / 视觉 / 多模态
1. 核心原理：音频可用波形/频谱建模（Conv-TasNet/UNet/AudioTransformer），图像用 CNN 或 patch 化 ViT/Swin。
2. 核心优势：各模态先做专用编码，再投影进统一序列空间与 LLM 联合建模。
3. 核心短板：输入 token 预算爆炸是落地第一瓶颈，压缩/采样/分辨率分级决定成败。

### 世界模型
1. 核心原理：预测未来状态/观测，在隐空间滚动推演，支持规划。
2. 核心优势：把学习从“下一个 token”扩展到环境动态，具身/驾驶/游戏潜力大。
3. 核心现状：多处于研究与早期落地，评测要看“预测准不准”而非“生成像不像”。

### MiniMind 式微型 LLM（P0–P2）
1. 预训练：纯 PyTorch 手写 TinyGPT，loss 标签必须左移一位（next-token），否则模型只会“抄当前字符”导致生成复读。
2. SFT：指令部分用 mask 屏蔽，只对回答算 response-only loss；LoRA 冻结基座、只训 A/B 两个低秩矩阵，合并时注意 ΔW 转置。
3. DPO：policy 与冻结的 ref 同起点，用 chosen/rejected 的平均 logp 差做 margin；小样本 + 大 lr 会过优化（KL 漂移），需用小 lr 观察 margin 温和上升。
4. 数据注意：chosen/rejected 不要互相引用对方答案，避免自相矛盾的偏好；字符级词表小，UNK 字符无学习信号。
5. MoE：Router 打分 + Top-k 专家，aux loss 防路由坍塌；小模型+小语料下 MoE 采样质量不如 Dense，稀疏收益在大规模才显现。
6. GRPO：组内采样 + 规则奖励 + 组相对 advantage；过拟合模型会导致全组 reward 相同、无学习信号，需要更难的任务或更小温度。
7. 工具/思考模板：先有模板工程再有训练数据；字符级分词覆盖不了 <>{} 等模板字符，真实项目用 BPE。

## 使用流程

1. 每个文件头部有：环境依赖、核心逻辑注释、参数释义、避坑、输出解读、工程改造方向；
2. 基础代码跑通后，使用文末「进阶改造 Prompt」引导自主优化；
3. 目录内代码可直接复制进自己的项目改造。

## 说明

- 代码以教学与原型验证为目标，生产部署请回到主线 27–40 章的框架/量化/编译流程；
- 涉及模型版本与硬件 kernel 的库，以官方文档为准。