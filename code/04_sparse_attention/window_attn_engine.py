# -*- coding: utf-8 -*-
"""
窗口注意力因果语言训练引擎（L2）

【层级】L2（工程标准：可直接用于个人长文本/局部依赖任务的训练脚本）
【环境依赖】
- Python 3.10+；PyTorch 2.x（建议 >=2.1）
- 安装：pip install torch==2.2.2；CPU 可跑，CUDA 自动可用
【核心逻辑】
- 任务：合成“隔 2 位拷贝”：每个位置预测“两个时间步之前”的 token
  （label_i = x[i-2]）。依赖距离固定为 2：
  window>=3 的模型能看 t-2 所以能学会，window=2 看不到 t-2 必然失败——
  用来精确演示“窗口大小 vs 依赖距离”的边界；
- 模型：Embedding -> 带因果+窗口 mask 的单层多头注意力 -> Linear 头；
- 工程点：参数校验、日志（控制台+文件）、seed/device、train/val 固定、
  best/last checkpoint、断点续训、early stop、CSV 指标、mask 统计输出。
【关键参数】（默认值 / 推荐值 / 适配场景）
- --window 8（默认；0=全注意力对照；局部任务 4-16 即可）：窗口宽度；
- --vocab 12（默认）；--seq_len 16：序列长度（copy 距离 2 远小于窗口，无长程干扰）；
- --heads 2、--dim 16：注意力头数与宽度；
- --steps 400（默认）；--lr 1e-3；--batch 32；
- --resume：续训；--patience 0（默认关）。
【避坑】
- window=0 表示全注意力（内部转成 seq 全窗口），别当 bug；
- 窗口 mask 必须与因果 mask 取交集（min），否则看到未来；
- 本任务是局部依赖，window 大与小都能收敛——真正能拉开差距的是
  长程任务（bench 见 L3），不要在局部任务上下“窗口无用”的结论。
【运行结果示例】（真实运行，CPU；lr=2e-3）
$ python window_attn_engine.py --window 8 --steps 600 --lr 2e-3 --log_dir out_win
INFO [window_attn_engine] device=cpu window=8 train=600 val=200
INFO [window_attn_engine] step 550 loss 0.001 acc 1.000 (best 1.000)
INFO [window_attn_engine] step 600 loss 0.000 acc 1.000 (best 1.000)
INFO [window_attn_engine] done best_val_acc=1.000 metrics=out_win/metrics.csv

$ python window_attn_engine.py --window 2 --steps 600 --lr 2e-3 --log_dir out_win
INFO [window_attn_engine] step 600 loss 2.438 acc 0.086 (best 0.086)
INFO [window_attn_engine] done best_val_acc=0.086 metrics=out_win/metrics.csv
（同一任务同一模型：window=8 看得到 t-2 -> acc 100%；window=2 看不到 -> 停在 1/12 随机水平）
【高频报错 Top5】
1. mask 全 False -> softmax NaN：window<2 或 seq_len<window；修复：参数校验已内置；
2. shape 错：输入 ids 需 [batch, seq]；修复：检查数据生成；
3. 因果+窗口同时用错：未来泄漏导致 acc 虚高；修复：mask = causal & window 取交集；
4. resume 后 acc 跳变：验证集不固定；修复：不要改 --seed；
5. CUDA OOM：seq_len/batch 太大；修复：--seq_len 16、--batch 32。
【输出解读】
- acc>0.9 说明模型学会“隔 2 位拷贝”（能 attend 到 t-2）；
- window=2 时 acc 应停在 1/vocab 附近（看不到 t-2）——这是本任务的教学核心；
- window>=3 与 full 的 acc 应一致 = 依赖距离内的窗口注意力无损失。
【工程改造方向】
- 长文档：把 mask 工程换成块式窗口实现（参考 L3 sparse_attn_bench.py）；
- 换数据：把自己语料切成定长 ids，替换 make_data；
- 加 FFN/层数即可变成可用的微型窗口 LM。
"""

import argparse
import csv
import logging
import math
import pathlib
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=8, help="0=全注意力")
    p.add_argument("--vocab", type=int, default=12)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--n_train", type=int, default=600)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--log_dir", default="out_win")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def validate(args):
    errs = []
    if args.vocab <= 4 or args.seq_len <= 4:
        errs.append("vocab/seq_len 至少 4")
    if args.window < 0:
        errs.append("window 不能为负（0=全注意力）")
    if args.window > 0 and args.window < 2:
        errs.append("window>=2 否则 mask 可能全 False")
    if errs:
        raise ValueError("\n".join(errs))


def make_data(n, seq_len, vocab, seed):
    """随机 token 序列（标签在训练循环里按“隔 2 位拷贝”构造）。"""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randint(0, vocab, (n, seq_len), generator=g)


def causal_window_mask(seq_len, window):
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    if window == 0:
        return causal
    idx = torch.arange(seq_len)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    win = dist < window
    return causal & win


class WinAttnLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask):
        b, s, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(b, s, self.heads, self.hd).transpose(1, 2)
        k = k.view(b, s, self.heads, self.hd).transpose(1, 2)
        v = v.view(b, s, self.heads, self.hd).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hd)
        scores = scores.masked_fill(~mask.to(scores.device), float("-inf"))
        o = torch.matmul(F.softmax(scores, -1), v)
        return self.out(o.transpose(1, 2).contiguous().view(b, s, -1))


class WindowLM(nn.Module):
    def __init__(self, vocab, dim, heads, window, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.attn = WinAttnLayer(dim, heads)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
        self.window = window
        self.mask = None

    def forward(self, ids):
        b, s = ids.shape
        x = self.emb(ids) + self.pos[:, :s]  # 位置编码：没有它模型学不会“隔 2 位”
        if self.mask is None or self.mask.shape[-1] != s:
            self.mask = causal_window_mask(s, self.window)
        x = x + self.attn(self.norm(x), self.mask)
        return self.head(x)


def setup_logging(log_dir):
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("win_engine")
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
    args = parse_args()
    try:
        validate(args)
    except ValueError as e:
        print("[config error]", e, file=sys.stderr)
        sys.exit(2)

    lg = setup_logging(args.log_dir)
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_x = make_data(args.n_train, args.seq_len, args.vocab, args.seed)
    val_x = make_data(args.n_val, args.seq_len, args.vocab, args.seed + 1)
    lg.info("device=%s window=%d train=%d val=%d",
            device, args.window, len(train_x), len(val_x))

    model = WindowLM(args.vocab, args.dim, args.heads, args.window).to(device)
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
    metrics = open(ckpt_dir / "metrics.csv",
                   "a" if args.resume else "w", newline="", encoding="utf-8")
    w = csv.writer(metrics)
    if not args.resume:
        w.writerow(["step", "loss", "val_acc", "elapsed_s"])
    metrics.flush()
    start = time.time()
    no_improve = 0

    def evaluate(xs):
        model.eval()
        right = total = 0
        with torch.no_grad():
            for i in range(0, len(xs), args.batch):
                xb = xs[i:i + args.batch].to(device)
                logits = model(xb)  # 位置 i 预测 x[i-2]
                pred = logits.argmax(-1)
                label = torch.roll(xb, 2, dims=1)
                label[:, :2] = -1  # 前 2 位无标签
                m = label != -1
                right += (pred[m] == label[m]).sum().item()
                total += m.sum().item()
        model.train()
        return right / total

    try:
        for step in range(step0, args.steps):
            idx = torch.randint(0, len(train_x), (args.batch,))
            xb = train_x[idx].to(device)
            logits = model(xb)
            label = torch.roll(xb, 2, dims=1)
            label[:, :2] = -100  # 前 2 位不参与 loss
            loss = ce(logits.reshape(-1, args.vocab), label.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % 50 == 0 or step + 1 == args.steps:
                acc = evaluate(val_x)
                improved = acc > best
                if improved:
                    best = acc
                    no_improve = 0
                    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                                "step": step + 1, "best": best}, ckpt_dir / "best.pt")
                else:
                    no_improve += 1
                lg.info("step %d loss %.3f acc %.3f (best %.3f%s)",
                        step + 1, loss.item(), acc, best,
                        ", saved" if improved else "")
                w.writerow([step + 1, round(loss.item(), 4), round(acc, 4),
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
    lg.info("done best_val_acc=%.3f metrics=%s", best, ckpt_dir / "metrics.csv")


if __name__ == "__main__":
    main()
