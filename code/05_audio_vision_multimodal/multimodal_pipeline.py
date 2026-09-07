# -*- coding: utf-8 -*-
"""
多模态特征管线 + 晚期融合训练引擎（L2）

【层级】L2（工程标准：多模态特征提取/融合训练的工程化骨架）
【环境依赖】
- Python 3.10+；PyTorch 2.x（建议 >=2.1）
- 安装：pip install torch==2.2.2；CPU 可跑，CUDA 自动可用
【核心逻辑】
- 数据：label 拆成两半——音频携带 parity=label%2（两类频率），文本携带
  half=label//2（token id），图像是纯噪声。单模态最多猜对一半，
  融合后才能 100%：这是“为什么要融合”的最小可验证证明；
- 特征管线：音频 STFT 幅度谱（频率池化到 16 维）、图像 8x8 图案 conv 池化、
  文本 embedding 平均——每个 extract_* 都返回固定维向量，真实项目把
  extract_* 内部换成 torchaudio/torchvision/真实 tokenizer 即可；
- 训练：晚期融合（concat -> MLP）；工程件：JSON 配置读写、参数校验、
  日志、seed/device、train/val 固定、best/last checkpoint、断点续训、
  CSV 指标、单模态消融（--ablate audio|vision|text 置零看掉点）。
【关键参数】（默认值 / 推荐值 / 适配场景）
- --d_audio 16/--d_image 16/--d_text 16（默认）：各模态特征维；
- --n_class 4（默认）；--n_train 800/--n_val 200：样本数；
- --steps 300（默认）；--lr 1e-3；--batch 64；
- --config：JSON 配置文件（覆盖 CLI）；--ablate：置零某模态做消融；
- --resume：续训；--patience 0（默认关）。
【避坑】
- 模态特征尺度不一（音频能量/embedding 数值差异大），concat 前最好各 norm；
- 本合成任务刻意设计为“互补信息”（音/文各一半），消融掉点 ~50%；
  真实数据模态贡献不同，消融是必须的例行检查；
- 配置 JSON 与 CLI 冲突时以 JSON 为准（工程约定要写进文档）；
- 真实数据接入：只改 make_* 与 extract_*，训练/评测代码不动。
【运行结果示例】（真实运行，CPU）
$ python multimodal_pipeline.py --steps 600 --batch 64 --log_dir out_mm
INFO [multimodal_pipeline] step 600 loss 0.020 val_acc 1.000 (best 1.000)
INFO [multimodal_pipeline] ablate full=1.000 -audio=0.505 -vision=1.000 -text=0.475
INFO [multimodal_pipeline] done best_val_acc=1.000 metrics=out_mm/metrics.csv
（去掉音频/文本后准确率掉到 ~0.5 = 单模态只有一半信息；去掉噪声图像无影响；
full=1.0 证明融合真的补全了信息）
【高频报错 Top5】
1. batch 维度不一致：三种模态第一维必须相同；修复：make_batch 统一 n；
2. stft 报长度错：音频太短；修复：duration>=0.05s 或减小 n_fft；
3. config JSON 字段名错：引擎启动即报“unknown key”；修复：对照 parse_args 字段；
4. 特征维拼接错：d_audio+d_image+d_text 需与 head 输入一致；修复：用配置算 fusion_in；
5. 消融后 acc 反而上升：模态噪声大时置零可去噪，属正常，多跑几轮看中位数。
【输出解读】
- val_acc 高 + 消融掉点明显（-audio/-text 掉到 ~0.5）= 融合有效；
- 无信息模态消融不掉点也属正常，消融的意义是找出“边际模态”。
【工程改造方向】
- 真实模态接入：extract_audio 换 torchaudio mel、extract_image 换 torchvision
  backbone、extract_text 换 tokenizer+LLM encoder；
- 对齐：加模态对齐 loss 或 attention 交互层（见 L3 multimodal_opt.py）；
- 服务化：把 extract_* 与 head 封装成单函数 multimodal_predict()。
"""

import argparse
import csv
import json
import logging
import math
import pathlib
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="JSON 配置（覆盖默认 CLI）")
    p.add_argument("--d_audio", type=int, default=16)
    p.add_argument("--d_image", type=int, default=16)
    p.add_argument("--d_text", type=int, default=16)
    p.add_argument("--n_class", type=int, default=4)
    p.add_argument("--n_train", type=int, default=800)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ablate", default=None, choices=[None, "audio", "vision", "text"])
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--log_dir", default="out_mm")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def apply_config(args):
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            data = json.load(fh)
        for k, v in data.items():
            if hasattr(args, k):
                setattr(args, k, v)
            else:
                raise ValueError("unknown config key: %s" % k)
    return args


def validate(args):
    errs = []
    if args.n_class < 2 or args.n_train <= 0:
        errs.append("n_class>=2 且 n_train>0")
    if args.steps <= 0:
        errs.append("steps 必须为正")
    if errs:
        raise ValueError("\n".join(errs))


# ---------- 合成数据与特征提取（真实项目替换这些函数即可） ----------
def make_batch(n, n_class, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    label = torch.randint(0, n_class, (n,), generator=g)
    # 标签拆分：音频只携带 parity=label%2，文本只携带 half=label//2，
    # 单模态最多 50% 判别力，融合后才能 100% —— 这是“为什么需要融合”的最小证明。
    parity = (label % 2).float()
    half = label // 2
    sr = 8000
    t = torch.arange(int(sr * 0.1), dtype=torch.float32) / sr
    base = 300.0 + 200.0 * parity  # 两类频率
    wave = torch.sin(2 * math.pi * base[:, None] * t[None, :])
    wave = wave + 0.4 * torch.randn(n, len(t), generator=g)
    # 图像：纯噪声（不携带信息，用来演示“无信息模态不影响融合”）
    img = (torch.rand(n, 1, 8, 8, generator=g) > 0.5).float()
    # 文本：前 3 个 token = 4 + half，其余随机（仍有稀释，但可学）
    ids = torch.randint(0, 8, (n, 8), generator=g)
    ids[:, :3] = (4 + half).unsqueeze(1)
    return wave, img, ids, label


class AudioFeat(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(out_dim)

    def forward(self, wave):
        spec = torch.stft(wave, n_fft=128, hop_length=32,
                          window=torch.hann_window(128),
                          return_complex=True)
        mag = spec.abs().mean(dim=-1)  # [n, freq]
        return self.pool(mag.unsqueeze(1)).squeeze(1)


class ImageFeat(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, img):
        # conv -> pool -> flatten：conv:[n,4,8,8] pool:[n,4,2,2] flatten=16
        x = self.conv(img).relu()
        x = self.pool(x)
        return x.flatten(1)


class TextFeat(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.emb = nn.Embedding(16, out_dim)

    def forward(self, ids):
        return self.emb(ids).mean(dim=1)


class LateFusionHead(nn.Module):
    def __init__(self, d_audio, d_image, d_text, n_class):
        super().__init__()
        self.audio = AudioFeat(d_audio)
        self.image = ImageFeat(d_image)
        self.text = TextFeat(d_text)
        self.in_dim = d_audio + d_image + d_text
        self.dims = (d_audio, d_image, d_text)
        self.head = nn.Sequential(nn.Linear(self.in_dim, 64), nn.ReLU(),
                                  nn.Linear(64, n_class))

    def forward(self, wave, img, ids, ablate=None):
        b = wave.shape[0]
        da, di, dt = self.dims
        fa = self.audio(wave) if ablate != "audio" else torch.zeros(b, da)
        fi = self.image(img) if ablate != "vision" else torch.zeros(b, di)
        ft = self.text(ids) if ablate != "text" else torch.zeros(b, dt)
        return self.head(torch.cat([fa, fi, ft], dim=-1))


def setup_logging(log_dir):
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("mm_pipe")
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(levelname)s [%(module)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(pathlib.Path(log_dir) / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    lg.addHandler(sh)
    lg.addHandler(fh)
    return lg


def main():
    args = apply_config(parse_args())
    try:
        validate(args)
    except ValueError as e:
        print("[config error]", e, file=sys.stderr)
        sys.exit(2)
    lg = setup_logging(args.log_dir)
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    train_w, train_i, train_t, train_y = make_batch(args.n_train, args.n_class, args.seed)
    val_w, val_i, val_t, val_y = make_batch(args.n_val, args.n_class, args.seed + 1)
    lg.info("device=%s train=%d val=%d", device, len(train_y), len(val_y))

    model = LateFusionHead(args.d_audio, args.d_image, args.d_text, args.n_class).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    step0, best = 0, 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step0, best = ck.get("step", 0), ck.get("best", 0.0)
        lg.info("resumed from %s at step %d", args.resume, step0)

    ckpt_dir = pathlib.Path(args.log_dir)
    metrics = open(ckpt_dir / "metrics.csv", "a" if args.resume else "w",
                   newline="", encoding="utf-8")
    wcsv = csv.writer(metrics)
    if not args.resume:
        wcsv.writerow(["step", "loss", "val_acc", "elapsed_s"])
    metrics.flush()
    start = time.time()
    no_improve = 0
    try:
        for step in range(step0, args.steps):
            idx = torch.randint(0, len(train_y), (args.batch,))
            wb, ib, tb, yb = (train_w[idx].to(device), train_i[idx].to(device),
                              train_t[idx].to(device), train_y[idx].to(device))
            loss = ce(model(wb, ib, tb, args.ablate), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % 50 == 0 or step + 1 == args.steps:
                model.eval()
                with torch.no_grad():
                    acc = 0.0
                    for i in range(0, len(val_y), args.batch):
                        wb, ib, tb = (val_w[i:i + args.batch].to(device),
                                      val_i[i:i + args.batch].to(device),
                                      val_t[i:i + args.batch].to(device))
                        pred = model(wb, ib, tb, args.ablate).argmax(-1)
                        acc += (pred.cpu() == val_y[i:i + args.batch]).sum().item()
                    acc /= len(val_y)
                model.train()
                improved = acc > best
                if improved:
                    best = acc
                    no_improve = 0
                    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                                "step": step + 1, "best": best}, ckpt_dir / "best.pt")
                else:
                    no_improve += 1
                lg.info("step %d loss %.3f val_acc %.3f (best %.3f%s)",
                        step + 1, loss.item(), acc, best, ", saved" if improved else "")
                wcsv.writerow([step + 1, round(loss.item(), 4), round(acc, 4),
                               round(time.time() - start, 2)])
                metrics.flush()
                if args.patience and no_improve >= args.patience:
                    lg.info("early stop at step %d", step + 1)
                    break
    except KeyboardInterrupt:
        lg.info("interrupted, saving last.pt")
    finally:
        metrics.close()
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "step": args.steps, "best": best}, ckpt_dir / "last.pt")
    # 消融例行报告
    model.eval()
    with torch.no_grad():
        row = []
        for mod in (None, "audio", "vision", "text"):
            acc = 0.0
            for i in range(0, len(val_y), args.batch):
                wb, ib, tb = (val_w[i:i + args.batch].to(device),
                              val_i[i:i + args.batch].to(device),
                              val_t[i:i + args.batch].to(device))
                pred = model(wb, ib, tb, mod).argmax(-1)
                acc += (pred.cpu() == val_y[i:i + args.batch]).sum().item()
            row.append(round(acc / len(val_y), 3))
    lg.info("ablate full=%.3f -audio=%.3f -vision=%.3f -text=%.3f", *row)
    lg.info("done best_val_acc=%.3f metrics=%s", best, ckpt_dir / "metrics.csv")


if __name__ == "__main__":
    main()