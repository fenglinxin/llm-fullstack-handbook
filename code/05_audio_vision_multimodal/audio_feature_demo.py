# -*- coding: utf-8 -*-
"""
音频模型落地 Demo：音频特征提取（stft/mel 可选）+ 最小 CNN 推理

【环境依赖】
- Python 3.10+, PyTorch 2.x（torch.stft 即可，无需 torchaudio）
- 可选 torchaudio：用于 mel 频谱

【核心逻辑】
- 波形 -> 分帧 STFT -> 幅度谱（特征）；
- 幅度谱喂给一个小 CNN（Conv1d 系列）做“合成分类任务”演示；
- 本 Demo 数据为合成正弦，重点是特征管线而不是效果。

【关键参数】
- sample_rate=16000, duration=1s；
- n_fft=256, hop=64（帧移越小时间分辨率越高、特征越长）

【避坑】
1. 波形要先归一化，避免数值溢出；
2. stft 的窗函数与 hop 会影响频率/时间分辨率；
3. 真实任务请用 torchaudio/compute 的 mel 特征并做均值方差归一化。

【输出解读】
- 打印 spectrogram 形状 [channel, freq, time]；
- CNN 输出概率和为 1 说明管线正确。

【工程改造方向】
- 换真实音频文件：torchaudio.load 后走同一管线；
- ASR 用帧级特征+CTC，TTS 用 mel+声码器（主线 12–19/27–33 章）；
- 特征太长时降采样/池化控制 token 预算。
"""

import torch
import torch.nn as nn


def make_wave(duration=1.0, sr=16000):
    t = torch.arange(int(sr * duration), dtype=torch.float32) / sr
    return (torch.sin(2 * 3.14159 * 440 * t) + 0.5 * torch.sin(2 * 3.14159 * 880 * t)).unsqueeze(0)


def stft_feature(wave, n_fft=256, hop=64):
    spec = torch.stft(
        wave,
        n_fft=n_fft,
        hop_length=hop,
        window=torch.hann_window(n_fft),
        return_complex=True,
    )
    mag = spec.abs().unsqueeze(0)  # [1, 1, freq, time]
    return mag


class TinyAudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.head = nn.Linear(8 * 8 * 8, 2)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.head(h)


def main():
    torch.manual_seed(0)
    wave = make_wave()
    feat = stft_feature(wave)
    print("spectrogram shape:", tuple(feat.shape))

    model = TinyAudioCNN()
    logits = model(feat)
    prob = logits.softmax(-1)
    print("model prob:", prob.detach().tolist())


"""
进阶改造 Prompt
1. 用 torchaudio 加载真实 wav 并替换合成波形；
2. 把 CNN 换成 AudioTransformer（patch 化频谱）对比效果；
3. 对噪声数据做增强，观察鲁棒性；
4. 把特征压缩到固定 token 数，测试与 LLM 拼接的可行性；
5. 结合主线 12–19 章做 ASR 小任务（音素/字符分类）。
"""

if __name__ == "__main__":
    main()
