# -*- coding: utf-8 -*-
"""
多模态融合最小落地 Demo：文本 + 图像 + 音频 -> 简单融合分类

【环境依赖】
- Python 3.10+, PyTorch 2.x；CPU 可运行

【核心逻辑】
- 三种模态分别抽特征（本 Demo 用随机/简单编码器代表）：
  文本：embedding 平均；
  图像：随机卷积特征池化；
  音频：帧特征均值；
- 特征 concat 后过 MLP 做分类（late fusion 晚期融合）；
- 用合成数据训练几步，验证梯度与结构可跑通。

【关键参数】
- text_dim=32, image_dim=64, audio_dim=16, hidden=64, n_class=3

【避坑】
1. 真实多模态要对齐 token/帧数，避免长度爆炸；
2. 不同模态特征尺度差异大，先各自归一化再融合；
3. 早期融合/晚期融合要按任务实测，不是越早越好。

【输出解读】
- loss 下降说明端到端可训；
- 打印融合特征维度 = 各模态维度之和。

【工程改造方向】
- 把随机编码器换成真实 CLIP/Whisper 特征；
- 对图文任务加对比学习对齐 loss；
- 与 LLM 拼接时关注 projector 设计与 token 预算。
"""

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab=50, dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)

    def forward(self, ids):
        return self.emb(ids).mean(dim=1)  # 平均池化


class ImageEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2), nn.ReLU(), nn.AdaptiveAvgPool2d(4))

    def forward(self, img):
        return self.conv(img).flatten(1)


class AudioEncoder(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.proj = nn.Linear(20, dim)

    def forward(self, frames):
        return self.proj(frames).mean(dim=1)


class LateFusion(nn.Module):
    def __init__(self, text_dim=32, image_dim=128, audio_dim=16, hidden=64, n_class=3):
        super().__init__()
        self.text_enc = TextEncoder(dim=text_dim)
        self.image_enc = ImageEncoder(dim=image_dim)
        self.audio_enc = AudioEncoder(dim=audio_dim)
        fusion_in = text_dim + image_dim + audio_dim
        self.head = nn.Sequential(nn.Linear(fusion_in, hidden), nn.ReLU(), nn.Linear(hidden, n_class))

    def forward(self, text_ids, img, frames):
        ft = self.text_enc(text_ids)
        fi = self.image_enc(img)
        fa = self.audio_enc(frames)
        return self.head(torch.cat([ft, fi, fa], dim=-1))


def main():
    torch.manual_seed(0)
    model = LateFusion()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(60):
        text = torch.randint(1, 50, (8, 10))
        img = torch.randn(8, 3, 32, 32)
        frames = torch.randn(8, 5, 20)
        label = torch.randint(0, 3, (8,))
        loss = loss_fn(model(text, img, frames), label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print("step %d loss %.4f" % (step, loss.item()))


"""
进阶改造 Prompt
1. 换真实数据：文本用 tokenizer、图像用 torchvision、音频用 torchaudio；
2. 尝试 early fusion（先对齐再融合）与 late fusion 对比；
3. 加对比学习让图文特征对齐，观察零样本检索效果；
4. 把融合特征接到 LLM 做图文问答（projector 方案）；
5. 评测时同时报告单模态与多模态效果，判断融合是否真有用。
"""

if __name__ == "__main__":
    main()
