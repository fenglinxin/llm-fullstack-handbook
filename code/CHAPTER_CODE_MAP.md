# 章节 ↔ 工程化代码映射表（47 章全覆盖）

> 原则：每章至少对应一个可直接运行的工程脚本/目录；
> 新代码遵守 code/CODE_STANDARD.md（9 个文档字段 + 三层标注）。
> 状态：✅=已有可运行代码；🛠=本轮/后续补建（见“计划批次”）。

| 章节 | 主题 | 对应代码（可运行） | 状态 |
|---|---|---|---|
| 00 | 模型结构核心原理 | code/00_rnn_lstm_gru ~ 06_world_model（7 目录三层）+ minimind_style 全链路 | ✅ |
| 01 | 一条大模型生产线全程 | minimind_style 主线脚本链（corpus→pretrain→sft→dpo→部署）+ tool_template_demo | ✅ |
| 02 | 2 小时跑通最小闭环 | minimind_style/pretrain.py + train_sft.py + generate.py（第 00 章 L2 引擎可作闭环替代） | ✅ |
| 03 | 预训练到底在练什么 | minimind_style/pretrain.py + model.py + tokenizer.py | ✅ |
| 04 | 数据管道从零构建 | data_engineering/pipeline_ingest.py | ✅ |
| 05 | 数据清洗规范与脏数据剔除 | data_engineering/clean_rules.py | ✅ |
| 06 | 文本过滤实战 | data_engineering/text_filter.py | ✅ |
| 07 | 数据去重算法 | data_engineering/dedup_minhash.py | ✅ |
| 08 | 敏感数据脱敏 | data_engineering/pii_mask.py | ✅ |
| 09 | 数据质量打分体系 | data_engineering/quality_score.py | ✅ |
| 10 | 领域专属数据构建 | data_engineering/domain_build.py | ✅ |
| 11 | 增量预训练方案 | minimind_style/pretrain.py（可续跑语料增量）+ training_tools/continue_pretrain.py（计划 C） | 🛠 C |
| 12 | 预训练超参选型与硬件适配 | training_tools/hyperparam_sweep.py | 🛠 C |
| 13 | 断点续训/收敛判断/失败排查 | 00_rnn_lstm_gru/ts_forecast_engine.py + 01_transformer/seq2seq_engine.py（resume/early stop 已实现） | ✅ |
| 14 | 全维度微调技术拆解 | minimind_style/lora.py+train_lora.py（LoRA）、train_sft.py（全参） | ✅ |
| 15 | SFT 监督微调实战 | minimind_style/train_sft.py | ✅ |
| 16 | SFT 数据集构建与标注规范 | minimind_style/data/sft_data.py + data_engineering/quality_score.py | ✅ |
| 17 | 微调模板设计 | minimind_style/tool_template_demo.py | ✅ |
| 18 | 微调超参调优策略 | training_tools/hyperparam_sweep.py（同 12，扩展 rank/alpha/lr） | 🛠 C |
| 19 | 过拟合/欠拟合/灾难遗忘 | training_tools/overfit_diagnoser.py | 🛠 C |
| 20 | 微调效果评估体系 | training_tools/eval_harness.py | 🛠 C |
| 21 | 多轮对话微调专项 | training_tools/multiturn_sft.py | 🛠 C |
| 22 | 模型蒸馏实战 | training_tools/distill_demo.py | 🛠 D |
| 23 | 奖励模型训练 | training_tools/rm_train.py（排序 loss 最小实现） | 🛠 D |
| 24 | RLHF/PPO 工程落地 | minimind_style/train_grpo.py（组相对基线）+ training_tools/ppo_mini.py（计划 D） | 🛠 D |
| 25 | RLAIF AI 反馈自动对齐 | training_tools/rlaif_synth.py | 🛠 D |
| 26 | 小样本/领域自适应微调 | training_tools/domain_adapt.py | 🛠 D |
| 27 | 部署框架横向对比 | deployment/framework_matrix.py（dry-run 检测+对比表） | 🛠 E |
| 28 | vLLM 部署实战与参数调优 | deployment/vllm_launch.py（dry-run/参数校验） | 🛠 E |
| 29 | SGLang 高性能推理落地 | deployment/sglang_launch.py | 🛠 E |
| 30 | TensorRT-LLM 固化与加速 | deployment/trtllm_build.py（dry-run 流程） | 🛠 E |
| 31 | 模型量化实操 | deployment/quantize_demo.py（PyTorch INT8/FP16 CPU 可跑） | 🛠 E |
| 32 | 模型剪枝与蒸馏压缩 | deployment/prune_demo.py（幅度剪枝 CPU 可跑） | 🛠 E |
| 33 | KV Cache 优化与上下文拓展 | 01_transformer/attention_opt.py（KV Cache 正确性+加速基准） | ✅ |
| 34 | 流式推理封装与高并发服务化 | deployment/stream_server_demo.py + concurrent_load.py | 🛠 F |
| 35 | 推理编译器核心原理 | deployment/compiler_ir_demo.py（极简 IR+拓扑+代码生成） | 🛠 F |
| 36 | 算子融合与计算图编译实战 | deployment/fusion_bench.py（手动 vs 融合 kernel 计时） | 🛠 F |
| 37 | 动态 shape 与显存编译器优化 | deployment/dynshape_demo.py | 🛠 F |
| 38 | Speculative Decoding 投机推理进阶 | deployment/speculative_demo.py（草稿+验证 CPU 可跑） | 🛠 F |
| 39 | 批量调度与并行深度适配 | deployment/continuous_batching_demo.py | 🛠 F |
| 40 | 内核重构与编译级量化 | 复用 deployment/quantize_demo.py + fusion_bench.py + compiler_ir_demo.py | 🛠 F |
| 41 | 全链路串联 | ops_pipeline/end_to_end.py（corpus→pretrain→sft→eval 一键编排） | 🛠 G |
| 42 | 领域大模型定制化落地 | ops_pipeline/domain_launch.py | 🛠 G |
| 43 | 高并发生产适配与端侧部署 | deployment/stream_server_demo.py + ops_pipeline/onnx_export_demo.py | 🛠 G |
| 44 | 模型迭代升级与性能对标评测 | ops_pipeline/ab_compare.py | 🛠 G |
| 45 | 线上问题闭环排查 | ops_pipeline/monitor_demo.py（日志+告警规则） | 🛠 G |
| 46 | 工程化最佳实践汇总 | 本映射表 + code/CODE_STANDARD.md（汇总性文档即交付物） | ✅ |

## 计划批次
- A（数据工程 04-08）：pipeline_ingest / clean_rules / text_filter / dedup_minhash / pii_mask
- B（数据工程 09-10）：quality_score / domain_build
- C（训练工具 11-13/18-21）：continue_pretrain / hyperparam_sweep / overfit_diagnoser / eval_harness / multiturn_sft
- D（对齐 22-26）：distill_demo / rm_train / ppo_mini / rlaif_synth / domain_adapt
- E（部署 27-32）：framework_matrix / vllm_launch / sglang_launch / trtllm_build / quantize_demo / prune_demo
- F（编译器/推理 34-40）：stream_server_demo / concurrent_load / compiler_ir_demo / fusion_bench / dynshape_demo / speculative_demo / continuous_batching_demo
- G（运营 41-45）：end_to_end / domain_launch / onnx_export_demo / ab_compare / monitor_demo

## 目录约定
- 新代码放 code/data_engineering、code/training_tools、code/deployment、code/ops_pipeline；
- 每个 .py 遵守 CODE_STANDARD.md 九字段；每个目录内含 L1/L2/L3 三层标注文件；
- “dry-run/检测类”脚本在无 GPU/框架的机器上也能输出真实环境诊断结果。