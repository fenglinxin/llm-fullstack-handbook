# 任务：MiniMind 式“从 0 微型 LLM”代码补充（优化扩展版）

> 参考项目：https://github.com/jingyaogong/minimind
> 参考定位：大道至简，纯 PyTorch 从 0 复现微型 LLM 全流程，不依赖 transformers/trl/peft 高层封装。
> 落地位置：llm-fullstack-handbook/code/minimind_style/（不改动已发布文档）。

## 一、任务目标（优化后）

在现有 code/（第 00 章架构 Demo）基础上，补充一个 **MiniMind 式微型 LLM 教学工程**：
用纯 PyTorch 从 0 实现“分词 → 预训练 → 生成 →（后续 SFT/DPO）”的最小闭环，
让读者在 CPU/单卡上几分钟跑完，真正理解模型每一行代码与主线 46 章工程知识的对应关系。

## 二、范围（MVP + 扩展）

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | TinyGPT 结构（RMSNorm/RoPE/因果注意力/FFN）+ 字符分词 + 微型预训练 + 文本生成 | 已完成 |
| P1 | 微型 SFT（指令模板 + 小 QA 数据） | 已完成 |
| P2 | LoRA/DPO 最小实现 | 已完成 |
| P3 | MoE 变体、RLHF/GRPO、工具调用/思考模板 | 已完成 |

## 三、与主线章节的映射

| 代码模块 | 主线章节 |
|---|---|
| TinyGPT 结构 | 00（Transformer 拆解）/ 03（预训练原理） |
| 数据与分词 | 04–10（数据管道/清洗/质量） |
| 预训练循环 | 11–13（CPT/超参/断点续训） |
| 生成与采样 | 34/38（流式/投机解码理解） |
| SFT/DPO（后续） | 15–25（SFT/DPO/RLHF） |

## 四、验收标准

1. 全部代码纯 PyTorch，不依赖 transformers/trl/peft（教学性）；
2. CPU 可运行：预训练 100–300 步 loss 明显下降；
3. 生成函数可输出学习后的中文片段（允许不流畅，但要体现“学到字符/词语搭配”）；
4. 每个文件含：环境依赖、核心逻辑注释、参数释义、避坑、输出解读、改造方向；
5. 代码通过 py_compile 并真实运行验证；
6. 任务范围、进度与参考定位写入本文件与 code/README.md。

## 五、执行原则

1. 原创实现，仅参考 MiniMind 的“从 0 教学”定位，不复制其代码；
2. 参数默认值保证普通 CPU 在 1–3 分钟内跑完；
3. 每一步先跑通再扩展；
4. 代码质量按“可读性 > 性能 > 完备性”排序。

## 六、进度

- [x] 调研 MiniMind 定位与结构
- [x] 输出本优化任务书
- [x] P0：模型/分词/预训练/生成（已跑通：loss ≈5.9 → 0.02，能复现语料句子）
- [x] P1：SFT（已跑通：loss ≈1.0 → 0.15，格式正确的问句能背出答案开头）
- [x] P2：LoRA/DPO（已跑通：LoRA 冻结基座只训 A/B，可训练 16,384/161,984≈10%，
      合并模型正常；DPO margin ≈1.45→2.38，小样本+大 lr 会过优化崩坏）
- [x] P3：MoE/GRPO/工具模板（已跑通：MoE 342,720 参数训练至 CE≈0.02；GRPO 规则奖励 0.850→0.869；模板演示三模式通过）

## 七、P1 实测记录与教训

- 修复了一个关键 bug：预训练 loss 标签未左移一位，导致模型退化成“抄当前字符”，
  loss 假性归零但生成只会复读；修复后 loss 从 ln(vocab)≈5.9 正常下降。
- SFT 用字符偏移实现 response-only mask（指令部分 -100 屏蔽），6 条 QA 上
  loss 从约 1.0 降到 0.15；因语料过小，长回复会串词，属教学预期。

## 八、P2 实测记录与教训

- LoRA：apply_lora 就地替换全部 nn.Linear（除 lm_head），基座全冻结；
  合并 ΔW=A@B 形状为 [in,out]，写回 nn.Linear.weight（[out,in]）必须先转置（已修复）。
- DPO：policy 与冻结 ref 同起点，chosen/rejected 平均 logp 差做 margin；
  默认 lr=1e-5 时 margin 温和上升且生成正常；lr=5e-4 时 margin 冲到 8+ 但生成崩坏，
  是典型的小样本 KL 漂移/过优化，教学上保留为对照。
- 数据教训：rejected 若引用其他问题的 chosen，会产生自相矛盾的偏好信号，
  真实偏好数据构造必须避免交叉引用。

## 九、P3 实测记录与教训

- MoE：Router+Top-k 专家+负载均衡 aux loss；Dense 145,600 → MoE 342,720 参数（2.35x），
  CE 均到 0.02 左右、aux≈2.0/块（均匀）；小模型+小语料下 MoE 采样质量反而差于 Dense——
  直观印证“MoE 的稀疏收益在大规模才显现”。
- GRPO：组内采样 5 个回答，规则奖励 = 与标准答案 LCP 占比，组内归一化 advantage；
  规则奖励 0.850 → 0.869；约半数组有奖励差异，其余因全组 reward=1.0 跳过——
  是过拟合背题的直观体现，也说明 RL 阶段需要更难的采样/更多样数据。
- 模板：CoT（<think>）与工具两轮闭环（<tool_call>→执行→<tool_result>→<answer>）
  纯标准库演示通过；check 模式实测字符级 tokenizer 对 <>{} 模板字符覆盖率仅 26-35%，
  佐证模板工程需 BPE/ByteLevel 分词。