# -*- coding: utf-8 -*-
"""
视觉模型落地 Demo：torchvision 预训练推理 + 无依赖时的 patch 特征回退

【环境依赖】
- Python 3.10+, PyTorch 2.x
- 可选 torchvision（有预训练权重时走真实推理；否则走合成图像教学路径）

【核心逻辑】
- 有 torchvision：加载 ResNet/ViT 对图像分类，展示特征提取与推理；
- 无 torchvision：用随机图像 + 手写 patch embedding + 线性头演示 ViT 骨架。

【关键参数】
- image_size=224（ResNet 输入）；
- patch=16（ViT patch 大小）

【避坑】
1. 预训练模型输入要按官方 normalize 做预处理；
2. 第一次运行会下载权重，离线环境请提前准备；
3. 教学回退路径不产生真实语义，只验证结构能跑。

【输出解读】
- 打印 top1 类别 id 与 logits 形状；
- 回退路径打印 patch 数 = (224/16)^2 = 196。

【工程改造方向】
- 换成 Swin/自研 backbone（主线第 00 章对比）；
- 视觉 encoder 输出接 LLM 时注意 token 压缩；
- 高分辨率输入先做分级采样控制显存。
"""

import torch
import torch.nn as nn


def try_torchvision():
    try:
        from torchvision import models, transforms
        from PIL import Image

        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.eval()
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # 演示用随机图代替真实图片；真实使用请替换为 Image.open(path)
        img = torch.randn(3, 224, 224)
        batch = img.unsqueeze(0)
        with torch.no_grad():
            out = model(batch)
        print("torchvision ResNet logits:", tuple(out.shape), "top1=", out.argmax(-1).item())
        return True
    except Exception as exc:
        print("torchvision 不可用，走教学回退：", str(exc)[:80])
        return False


class PatchEmbed(nn.Module):
    """ViT 最小 patch embedding：把 [B,3,H,W] 切成 patch 并线性投影。"""

    def __init__(self, in_ch=3, patch=16, d_model=64):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)

    def forward(self, x):
        # [B, d_model, H/patch, W/patch] -> [B, num_patch, d_model]
        return self.proj(x).flatten(2).transpose(1, 2)


def fallback_demo():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)
    pe = PatchEmbed()
    tokens = pe(x)
    head = nn.Linear(64, 10)
    logits = head(tokens.mean(dim=1))
    print("fallback patch tokens:", tuple(tokens.shape), "logits:", tuple(logits.shape))


def main():
    ok = try_torchvision()
    if not ok:
        fallback_demo()


"""
进阶改造 Prompt
1. 用真实图片目录替换随机图，跑 ResNet/ViT 分类并对比准确率；
2. 对比 CNN/ResNet 与 ViT/Swin 在小数据与大数据的表现差异；
3. 抽取倒数第二层特征做检索/embedding，验证特征质量；
4. 把视觉 encoder 输出接到文本 LLM 的 projector，做图文问答；
5. 对高分辨率长图做 token 压缩策略实验。
"""

if __name__ == "__main__":
    main()
