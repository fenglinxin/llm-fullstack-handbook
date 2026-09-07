# -*- coding: utf-8 -*-
"""
MiniMind-O 教学版：微型全模态（Omni）模型——音频 + 视觉 + 文本

【定位】对标 MiniMind-O“全模态”思路的教学原创实现：不同模态各有一个
极简编码器，把音频/图像变成 token，共享同一个 decoder-only 文本解码器。
本版覆盖：音调分类（音频->文本）与看图说话（图像->文本）两类任务。

【环境依赖】Python 3.10+, PyTorch 2.x（无 torchaudio/torchvision 依赖）。

【关键参数】
- 音频：8kHz、0.1s 波形 -> 8 帧 RMS + 3 路候选频率幅度 = 每半段 7 维特征，
  共 2 个音频 token；
- 图像：12x12 二值图 -> 36 个图像 token（复用 VisionTower）；
- 文本解码器与 MiniMind-V 相同（TinyGPT 结构）。

【避坑】
1. 不同模态序列长度不同，batch 内不要混排成同一个张量，
   教学版做法：音频/图像样本分开 forward，共享 decoder 一起反传；
2. 音频特征要归一化到同一量纲（RMS 与幅度都在 0~1 附近），
   否则梯度被大数值维度带走；
3. 测试集用“没见过的噪声实例”：频率相位/噪声都换掉，
   只有真正学会“听音高/看图案”才能答对。

【输出解读】audio acc / vision acc 分开打印：都明显高于随机
（3 类音调=33%，8 类图案=12.5%）即说明 Omni 解码器真的吃到了两种模态。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TinyGPT
from vision_model import VisionTower, render_pattern  # 复用视觉塔与图案渲染

SR = 8000          # 采样率
DUR = 0.1          # 秒
TONE_FREQS = [440.0, 660.0, 880.0]   # 低/中/高音
TONE_NAMES = ["低", "中", "高"]
FRAMES = 8         # RMS 帧数


def render_tone(freq, noise_seed=None):
    """生成一段 0.1s 正弦音（可加随机相位/幅度噪声），返回 [800] 波形。"""
    n = int(SR * DUR)
    t = torch.arange(n).float() / SR
    if noise_seed is not None:
        torch.manual_seed(noise_seed)
        phase = torch.rand(1).item() * 2 * math.pi
        amp = 0.8 + 0.2 * torch.rand(1).item()
        wave = amp * torch.sin(2 * math.pi * freq * t + phase)
        wave = wave + 0.05 * torch.randn(n)
    else:
        wave = torch.sin(2 * math.pi * freq * t)
    return wave


def audio_features(wave, sr=SR):
    """波形 -> [2, 7] 特征：8 帧 RMS 分成两半，各拼 3 路候选频率幅度。"""
    n = wave.shape[0]
    frame = n // FRAMES
    rms = []
    for i in range(FRAMES):
        seg = wave[i * frame:(i + 1) * frame]
        rms.append(seg.pow(2).mean().sqrt())
    rms = torch.stack(rms)
    amps = []
    t = torch.arange(n).float() / sr
    for f in TONE_FREQS:
        amps.append((wave * torch.sin(2 * math.pi * f * t)).mean().abs())
    amps = torch.stack(amps)
    feats = []
    for half in range(2):
        feats.append(torch.cat([rms[half * 4:(half + 1) * 4], amps]))  # [7]
    return torch.stack(feats)  # [2, 7]


class AudioTower(nn.Module):
    """极简音频编码器：每半段 7 维特征 -> 1 个 token（共 2 token）。"""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(7, dim, bias=False)
        self.pos = nn.Parameter(torch.randn(1, 2, dim) * 0.02)
        self.norm = nn.RMSNorm(dim)

    def forward(self, feats):
        # feats: [B, 2, 7]
        x = self.proj(feats) + self.pos
        return self.norm(x)


class MiniMindO(nn.Module):
    """MiniMind-O 教学版：AudioTower + VisionTower + 共享文本解码器。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.audio = AudioTower(cfg.dim)
        self.vision = VisionTower(cfg.dim)
        self.text = TinyGPT(cfg)

    # ---------- 音频路径 ----------
    def _forward_tokens(self, modal_emb, ids):
        t = self.text.token_emb(ids)
        x = torch.cat([modal_emb, t], dim=1)
        for block in self.text.blocks:
            x = block(x, self.text.cos, self.text.sin)
        return self.text.lm_head(self.text.norm(x))

    def forward_audio(self, feats, ids):
        return self._forward_tokens(self.audio(feats), ids)

    def loss_audio(self, feats, ids):
        logits = self.forward_audio(feats, ids)
        p = logits.shape[1] - ids.shape[1]
        labels = ids[:, 1:].clone().masked_fill(ids[:, 1:] == 0, -100)
        return F.cross_entropy(logits[:, p:-1].reshape(-1, logits.shape[-1]),
                               labels.reshape(-1))

    @torch.no_grad()
    def generate_audio(self, feats, prompt_ids, max_new=16,
                       temperature=0.6, top_k=10):
        self.eval()
        gen = list(prompt_ids)
        for _ in range(max_new):
            ids = torch.tensor([gen[-self.cfg.max_seq + 2:]])
            logits = self.forward_audio(feats, ids)[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            gen.append(int(torch.multinomial(F.softmax(logits, -1), 1).item()))
        return gen

    # ---------- 视觉路径 ----------
    def forward_vision(self, img, ids):
        if img.dim() == 3:
            img = img.unsqueeze(0)
        return self._forward_tokens(self.vision(img), ids)

    def loss_vision(self, img, ids):
        logits = self.forward_vision(img, ids)
        p = logits.shape[1] - ids.shape[1]
        labels = ids[:, 1:].clone().masked_fill(ids[:, 1:] == 0, -100)
        return F.cross_entropy(logits[:, p:-1].reshape(-1, logits.shape[-1]),
                               labels.reshape(-1))

    @torch.no_grad()
    def generate_vision(self, img, prompt_ids, max_new=16,
                        temperature=0.6, top_k=10):
        self.eval()
        if img.dim() == 3:
            img = img.unsqueeze(0)
        gen = list(prompt_ids)
        for _ in range(max_new):
            ids = torch.tensor([gen[-self.cfg.max_seq + 36:]])
            logits = self.forward_vision(img, ids)[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            gen.append(int(torch.multinomial(F.softmax(logits, -1), 1).item()))
        return gen


"""
进阶改造 Prompt
1. 把两条独立 forward 合成“统一 token 序列 + 模态类型 token”的架构，
   让任意模态组合进同一个 batch（参考 MiniMind-O 论文思路）；
2. 音频加真实噪声/混响，图像换真实照片，观察编码器瓶颈；
3. 加“听音+看图联合问答”样本：如“图里物体发出的声音是什么？”
4. 对比模态 token 数（1/2/4）与两类任务精度的关系；
5. 思考真实 Omni 模型为何常用模态专用 tokenizer + 统一 projector。
"""

"""
规范字段补充（全局强制代码落地规范）

【层级】L3（模型实现模块：MiniMind-O 音画双模态共享解码器）
【核心逻辑】AudioTower 把 8 帧 RMS+3 路频率幅度特征压成 2 token；
VisionTower 复用视觉 patch；两路径共享同一文本解码器（分 forward 便于 CPU 教学）。
【运行结果示例】（真实运行，train_o.py 60 epochs）
audio_acc 1.000 vision_acc 0.875；loss_a 0.030 loss_v 0.011
【高频报错 Top5】
1. 音频特征训练/推理必须同一函数（audio_features），否则频点错位；
2. audio 与 vision 样本不能混 batch：token 数不同，分路径 forward（已实现）；
3. 两条 loss 量级失衡：可加权重（本任务量级接近，未加）；
4. 波形噪声 seed 与数据行不对应：row 里的 noise_seed 要原样传给 render；
5. 换成真实音频后采样率不一致：所有环节统一 sr。
【工程改造方向】统一 token 序列+模态标记训练；接真实 encoder；联合问答任务。
"""
