# -*- coding: utf-8 -*-
"""
MiniMind-V 教学版：微型视觉-语言模型（图片 -> 文本描述）

【定位】对标 MiniMind-V“视觉模态”思路的教学原创实现：用极简
Vision Tower（patch 化 + 可学习位置编码 + 投影）把图片变成 token，
与文本 token 拼接后交给同一个 decoder-only 语言模型自回归生成。
不依赖 torchvision/transformers，CPU 可跑。

【环境依赖】Python 3.10+, PyTorch 2.x。

【关键参数】
- IMG=12：12x12 二值图（可理解成 12x12 像素/边缘图）；
- patch=2x2 -> 36 个图像 token，每个 token 维度=dim（与文本对齐）；
- 8 种基础图案：四个角方块/横线/竖线/十字/边框。

【避坑】
1. 图像 token 必须放在文本 prompt 之前，且要参与同一个因果注意力，
   否则模型“看不到图”只会背答案；
2. 训练 loss 只统计文本段（图像 token 不预测下一个 token），
   类似 SFT 的 response-only mask 思想；
3. 图像尺寸/图案太简单时模型可能直接过拟合文字，评测用
   “没见过的图案”才能看出是否真的学会看图。

【输出解读】val acc = 用图片+问题生成，回答前缀与标注答案一致的占比；
acc 明显高于瞎猜（1/8=12.5%）说明视觉通路真的把信息传给了文本解码器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TinyGPT, TinyConfig

IMG = 12          # 图像边长（像素）
PATCH = 2         # patch 边长
N_PATCH = (IMG // PATCH) ** 2  # 图像 token 数 = 36

# 图案 id -> 中文描述（字符需全部能被训练时的 tokenizer 覆盖）
CAPTIONS = [
    "图中有方块在左上角。",
    "图中有方块在右上角。",
    "图中有方块在左下角。",
    "图中有方块在右下角。",
    "图中有横线。",
    "图中有竖线。",
    "图中有十字。",
    "图中有边框。",
]


def render_pattern(pid, noise_seed=None, img=IMG, patch=PATCH):
    """把图案 id 渲染成 [1, img, img] 的二值张量（0/1，float）。

    图案规律清晰、可逆：模型只要学会“看”就能答对；
    noise_seed 给图案加一粒噪声点，生成同一图案的多个图像实例。
    """
    g = torch.zeros(1, img, img)
    s = img // 2
    if pid == 0:    # 左上角方块
        g[0, 1:5, 1:5] = 1
    elif pid == 1:  # 右上角方块
        g[0, 1:5, img - 5:img - 1] = 1
    elif pid == 2:  # 左下角方块
        g[0, img - 5:img - 1, 1:5] = 1
    elif pid == 3:  # 右下角方块
        g[0, img - 5:img - 1, img - 5:img - 1] = 1
    elif pid == 4:  # 横线（中间行）
        g[0, s - 1:s + 1, 1:img - 1] = 1
    elif pid == 5:  # 竖线（中间列）
        g[0, 1:img - 1, s - 1:s + 1] = 1
    elif pid == 6:  # 十字
        g[0, 1:img - 1, s - 1:s + 1] = 1
        g[0, s - 1:s + 1, 1:img - 1] = 1
    else:           # 边框
        g[0, 0, :] = 1
        g[0, -1, :] = 1
        g[0, :, 0] = 1
        g[0, :, -1] = 1
    if noise_seed is not None:
        torch.manual_seed(noise_seed)
        nr, nc = torch.randint(0, img, (2,)).tolist()
        g[0, nr, nc] = 1 - g[0, nr, nc]
    return g


class VisionTower(nn.Module):
    """极简视觉编码器：patch 化 -> 线性投影 -> 位置编码 -> RMSNorm。

    真实 MiniMind-V 会用 SigLIP/CLIP 这类预训练视觉塔，
    教学版从零学一个“看图说话”的投影即可。
    """

    def __init__(self, dim, patch=PATCH, img=IMG):
        super().__init__()
        self.patch = patch
        self.img = img
        n_patch = (img // patch) ** 2
        self.proj = nn.Linear(patch * patch, dim, bias=False)
        self.pos = nn.Parameter(torch.randn(1, n_patch, dim) * 0.02)
        self.norm = nn.RMSNorm(dim)

    def forward(self, img):
        # img: [B, 1, img, img] -> [B, n_patch, patch*patch]
        b = img.shape[0]
        patches = (img
                   .unfold(2, self.patch, self.patch)
                   .unfold(3, self.patch, self.patch)
                   .reshape(b, -1, self.patch * self.patch))
        x = self.proj(patches) + self.pos
        return self.norm(x)


class MiniMindV(nn.Module):
    """MiniMind-V 最小闭环：VisionTower + TinyGPT 文本解码器。"""

    def __init__(self, cfg, patch=PATCH, img=IMG):
        super().__init__()
        self.cfg = cfg
        self.vision = VisionTower(cfg.dim, patch=patch, img=img)
        self.text = TinyGPT(cfg)  # 复用模型结构（含因果注意力/RoPE/FFN）

    def forward(self, img, ids):
        """img:[B,1,H,W] ids:[B,T]。返回全部位置 logits（图像段+文本段）。"""
        v = self.vision(img)                       # [B, P, D]
        t = self.text.token_emb(ids)               # [B, T, D]
        x = torch.cat([v, t], dim=1)               # [B, P+T, D]
        for block in self.text.blocks:
            x = block(x, self.text.cos, self.text.sin)
        return self.text.lm_head(self.text.norm(x))

    def loss(self, img, ids):
        """只统计文本段：文本位置 j 用 logits[P+j] 预测 ids[j+1]。

        ids 允许按行 padding（pad_id=0=<unk>，语料 fit 后正文不会出现 0），
        padding 位置的标签置 -100 交给 CrossEntropy 忽略。
        """
        logits = self.forward(img, ids)            # [B, P+T, V]
        p = logits.shape[1] - ids.shape[1]
        txt_logits = logits[:, p:-1, :]            # 丢掉最后一个文本位置（无下一 token）
        labels = ids[:, 1:].clone()
        labels = labels.masked_fill(labels == 0, -100)  # pad 位置不参与 loss
        return F.cross_entropy(
            txt_logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1))

    @torch.no_grad()
    def generate(self, img, prompt_ids, max_new=24, temperature=0.6, top_k=10):
        """看图 + 文本 prompt 自回归生成。

        img 允许传 [1, H, W]（单通道）或 [1, 1, H, W]（带 batch 维）。
        """
        self.eval()
        if img.dim() == 3:
            img = img.unsqueeze(0)  # [1,H,W] -> [1,1,H,W]
        gen = list(prompt_ids)
        for _ in range(max_new):
            ids = torch.tensor([gen[-self.cfg.max_seq + 36:]])
            logits = self.forward(img, ids)[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            gen.append(torch.multinomial(probs, 1).item())
        return gen


"""
进阶改造 Prompt
1. 把二值图案换成真实小图片（灰度 PNG），观察投影层能否直接学；
2. 换“看图问答”数据：同一张图配多个问题（位置？数量？颜色？）；
3. 给视觉塔加一层双向自注意力，让 patch 之间先“互相看”再进文本解码器；
4. 对比图像 token 放在 prompt 前/后、是否带位置编码的效果；
5. 思考 MiniMind-V 真实做法：冻结 SigLIP + 训练 Projector 两阶段，为什么。
"""
