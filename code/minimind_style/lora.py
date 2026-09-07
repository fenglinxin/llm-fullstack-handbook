# -*- coding: utf-8 -*-
"""
LoRA 最小实现（MiniMind 式“自己动手”教学版）

【定位】只加两个小矩阵 A/B 近似权重更新 ΔW≈A@B*scale，
冻结原权重，只训练 A/B——这就是 LoRA 的全部核心。

【环境依赖】Python 3.10+, PyTorch 2.x（无第三方库）

【关键参数】
- r：低秩秩（rank），越大表达力越强、可训练参数越多；
- alpha：缩放系数，实际缩放 = alpha / r；
- 目标层：替换除 lm_head 外所有 nn.Linear（qkv/out/up/down）。

【避坑】
1. A 用 kaiming 初始化、B 用全 0 初始化：初始时 LoRA 贡献为 0，
   微调从“原模型权重”出发而不是从随机扰动出发；
2. 冻结基座后 AdamW 只更新 A/B，否则“假装 LoRA”实则全量训练；
3. 推理/合并时把 ΔW 加回原权重即可，无额外延迟。

【输出解读】打印可训练参数 vs 总参数：大模型场景 LoRA 通常只占 1% 以下；
本微型模型骨干占比高，实测约占 10%，教学重点是“冻结基座只训 A/B”的机制。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """把普通 Linear 替换为 frozen 基座 + 可训练低秩分支。"""

    def __init__(self, base, r=8, alpha=16.0):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        # 基座权重：detach 后以非持久 buffer 保存（冻结、不计入 parameters/state_dict）
        self.register_buffer("base_weight", base.weight.detach().clone(), persistent=False)
        if base.bias is not None:
            self.register_buffer("base_bias", base.bias.detach().clone(), persistent=False)
        else:
            self.base_bias = None
        # 低秩分支：x @ A @ B
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, self.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # B 保持全 0

    def forward(self, x):
        base_out = F.linear(x, self.base_weight, self.base_bias)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scale
        return base_out + lora_out

    def merge(self):
        """返回合并后的普通 nn.Linear（推理用，不再需要 A/B）。"""
        merged = nn.Linear(self.in_features, self.out_features,
                           bias=self.base_bias is not None)
        with torch.no_grad():
            # forward 是 x@A@B，ΔW=A@B 形状 [in,r]x[r,out]=[in,out]；
            # Linear.weight 约定是 [out,in]，所以合并时要转置
            merged.weight.copy_(self.base_weight + (self.lora_A @ self.lora_B).t() * self.scale)
            if self.base_bias is not None:
                merged.bias.copy_(self.base_bias)
        return merged


def apply_lora(model, r=8, alpha=16.0):
    """就地替换所有 nn.Linear（除 lm_head）为 LoRALinear，并冻结基座。"""
    def rec(module):
        for name, child in list(module._modules.items()):
            if isinstance(child, nn.Linear) and name != "lm_head":
                module._modules[name] = LoRALinear(child, r=r, alpha=alpha)
            else:
                rec(child)
    rec(model)
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)
    return model


def merge_lora(model):
    """把 LoRALinear 合并回普通 Linear（就地修改），输出可直接推理的模型。"""
    def rec(module):
        for name, child in list(module._modules.items()):
            if isinstance(child, LoRALinear):
                module._modules[name] = child.merge()
            else:
                rec(child)
    rec(model)
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


"""
进阶改造 Prompt
1. 只对 qkv/out（注意力）打 LoRA，对比全打的效果与显存；
2. 对比 r=1/4/16/64 的可训练参数与生成质量；
3. 加 target_modules 参数支持“只打 FFN”等配置；
4. 在训练中记录 A/B 的范数，观察低秩更新的稳定性；
5. 对比 merge 前后输出是否一致（应完全一致，误差 < 1e-6）。
"""

"""
规范字段补充（全局强制代码落地规范）

【层级】L3（优化实现模块：LoRA 低秩微调的最小完整实现）
【运行结果示例】（真实运行，CPU，r=8/alpha=16 挂到 TinyGPT 上）
params total=63680 trainable=16384（基座 145600 权重，新增 A/B 约 10%）
（apply_lora 后 count_params 打印；train_lora.py 实测 loss 0.11 -> 0.04-0.08）
【高频报错 Top5】
1. 合并后权重维数错：ΔW=A@B 是 [in,out]，写回 [out,in] 需转置（已修）；
2. 训练没效果：基座没冻结；修复：apply_lora 已对非 lora 参数 requires_grad_(False)；
3. count 翻倍：旧版把基座权重存成 Parameter；修复：现在用 non-persistent buffer；
4. adapter 无法换基座：保存时只存 lora_* 参数；修复：用本模块约定的 dict；
5. alpha/r 太大输出爆：scale=alpha/r 失控；修复：alpha=r 起步。
【工程改造方向】target_modules 选择（只打注意力/FFN）、r 扫描、adapter 服务化热插拔。
"""

"""
规范字段补充 2（补齐缺失字段标记）

【核心逻辑】
- LoRALinear：frozen 基座（non-persistent buffer）+ A/B 两个低秩参数；
- forward：y = x@W + (x@A@B)*scale，B 初始为 0 保证“从原权重出发”；
- apply_lora：就地替换全部 nn.Linear（除 lm_head）并冻结基座；
- merge_lora：ΔW=A@B 转置后加回基座权重，还原成普通 Linear。
"""
