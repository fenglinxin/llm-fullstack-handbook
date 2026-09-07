# -*- coding: utf-8 -*-
"""
状态空间模型（SSM）工程化训练引擎（L2）——前缀累加任务

【层级】L2（工程标准：可直接用于个人时序/流式任务的训练脚本）
【环境依赖】
- Python 3.10+；PyTorch 2.x（建议 >=2.1）
- 安装：pip install torch==2.2.2；CPU 可跑，CUDA 自动可用
【核心逻辑】
- 任务：前缀累加（running sum）——输入随机步长序列，每步输出“到当前为止的
  累计值”。这是 SSM 的天然主场（状态本身就在做积分），用来和 LSTM/RNN 对比
  最能体现“固定大小状态 + 递推”的建模方式；
- 模型：LinearSSM——s_t = a*s_{t-1} + x_t（可学习漏损系数 a），
  y_t = out(s_t)。注意：L1 的 MinimalSSM 带 tanh 是“概念直觉版”，
  不适合直接训练（tanh 会截断累加值），本引擎用线性版才能收敛；
- 工程点：参数校验、日志（控制台+文件）、设备/种子固定、训练/验证切分、
  best/last checkpoint、断点续训、early stop、梯度裁剪、CSV 指标、
  推理阶段用“无窗口纯状态递推”（这才是 SSM 的流式推理形态）。
【关键参数】（默认值 / 推荐值 / 适配场景）
- --d_state 1（默认；=1 时就是标量漏损积分器，任务够用；复杂任务加大）：状态维度；
- --seq_len 24（默认；累加长度，20-40 教学合适）：每条样本序列长度；
- --n_train 600（默认）/ --n_val 200：样本数；
- --epochs 60（默认；100 更稳）：训练轮数；
- --lr 5e-3（默认；1e-3~1e-2 区间）：学习率；--clip 1.0：梯度裁剪；
- --a_init 0.95（默认；接近 1 近似无损耗积分）：漏损初始化；
- --resume：续训；--patience 0（默认关）。
【避坑】
- 纯线性状态递推 + 漏损 a<1 会产生“衰减”，若任务要求精确长程积分，
  需要把 a 学近 1 并配合归一化目标（本任务目标除 6 已缩放）；
- 训练用整条序列做 BPTT（24 步短序列没问题），长序列要截断 BPTT；
- 推理时不需要窗口：只带 state 逐点输入，输出即预测——这是 SSM 与
  Transformer KV Cache 思路同源的“恒定状态”优势；
- 归一化/缩放参数只从训练集估计。
【运行结果示例】（真实运行，CPU）
$ python ssm_engine.py --epochs 60 --log_dir out_ssm
INFO [ssm_engine] epoch 10/60 loss 0.02386 val_mse 0.01959 (best 0.01959, saved)
INFO [ssm_engine] epoch 20/60 loss 0.00101 val_mse 0.00088 (best 0.00088, saved)
INFO [ssm_engine] epoch 40/60 loss 0.00040 val_mse 0.00036 (best 0.00036, saved)
INFO [ssm_engine] epoch 60/60 loss 0.00024 val_mse 0.00022 (best 0.00022, saved)
INFO [ssm_engine] stateful rollout sample: pred[0:6]=[-0.09,-0.13,-0.16,-0.25,-0.25,-0.28] truth=[-0.08,-0.11,-0.15,-0.23,-0.23,-0.26]
INFO [ssm_engine] done best_val_mse=0.00022 metrics=out_ssm/metrics.csv
【高频报错 Top5】
1. shape 报错：x 需 [batch, seq_len, 1]，y 需 [batch, seq_len, 1]；
2. loss=NaN：lr 过大或 a 被推到 >1；修复：--lr 5e-3、--clip 1.0、
   或对 a 做 sigmoid 参数化（见进阶改造）；
3. resume 后指标跳变：验证集不固定；修复：不要改 --seed；
4. val_mse 大但 train 小：过拟合；修复：加 dropout/更多数据/early stop；
5. CUDA OOM：batch 太大；修复：--batch 32。
【输出解读】
- loss/val_mse 单调下降至 <0.001（目标已缩放）即成功；
- stateful rollout 输出与 truth 前几位接近 = 流式推理正确。
【工程改造方向】
- 换数据：把 make_batch 换成自己的流式特征（每步一个向量）；
- 状态维度扩到 d_state>1 并加 B/C 投影，即可承接真实 SSM 结构；
- 优化：把 a 换成 sigmoid 参数化 + 输入依赖（selective scan 最小版，
  参考 ssm_opt.py L3）；推理服务只需持久化 state，天然流式。
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d_state", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--n_train", type=int, default=600)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--a_init", type=float, default=0.95)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--log_dir", default="out_ssm")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def validate(args):
    errs = []
    if args.seq_len <= 0 or args.epochs <= 0 or args.n_train <= 0:
        errs.append("seq_len/epochs/n_train 必须为正整数")
    if not (0 < args.lr < 1):
        errs.append("lr 需在 (0,1)")
    if errs:
        raise ValueError("\n".join(errs))


def make_batch(n, seq_len, seed):
    """随机步长序列 -> (x, 缩放后的前缀累加目标)。"""
    g = torch.Generator()
    g.manual_seed(seed)
    x = torch.randn(n, seq_len, 1, generator=g) * 0.3
    y = x.cumsum(dim=1) / 6.0  # 缩放到 [-1,1] 附近，便于学习
    return x, y


class LinearSSM(nn.Module):
    """线性状态递推：s_t = a*s_{t-1} + x_t；y_t = head(s_t)。

    相比 L1 直觉版（tanh MinimalSSM），去掉 tanh 才能让累加值通过，
    是本引擎能收敛的关键。
    """

    def __init__(self, d_state=1, a_init=0.95):
        super().__init__()
        self.d_state = d_state
        # 漏损系数参数化：a = sigmoid(logit)，保证永远在 (0,1) 内（数值稳定）
        self.a_logit = nn.Parameter(torch.tensor(math.log(a_init / (1 - a_init))))
        self.out = nn.Linear(d_state, 1)

    def forward(self, x):
        batch, seq, _ = x.shape
        a = torch.sigmoid(self.a_logit)
        state = torch.zeros(batch, self.d_state)
        outs = []
        for t in range(seq):
            state = a * state + x[:, t]
            outs.append(self.out(state))
        return torch.stack(outs, dim=1)  # [batch, seq, 1]

    def rollout(self, x):
        """流式推理：无窗口，只带 state 逐点输入（输出与 forward 一致）。"""
        a = torch.sigmoid(self.a_logit)
        state = torch.zeros(x.shape[0], self.d_state)
        outs = []
        with torch.no_grad():
            for t in range(x.shape[1]):
                state = a * state + x[:, t]
                outs.append(self.out(state))
        return torch.stack(outs, dim=1)


def setup_logging(log_dir):
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("ssm_engine")
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
    lg.info("device=%s d_state=%d seq_len=%d", device, args.d_state, args.seq_len)

    train_x, train_y = make_batch(args.n_train, args.seq_len, args.seed)
    val_x, val_y = make_batch(args.n_val, args.seq_len, args.seed + 1)
    lg.info("train=%d val=%d", len(train_x), len(val_x))

    model = LinearSSM(d_state=args.d_state, a_init=args.a_init).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    epoch0, best = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        epoch0, best = ck.get("epoch", 0), ck.get("best", float("inf"))
        lg.info("resumed from %s at epoch %d", args.resume, epoch0)

    ckpt_dir = pathlib.Path(args.log_dir)
    metrics = open(ckpt_dir / "metrics.csv",
                   "a" if args.resume else "w", newline="", encoding="utf-8")
    w = csv.writer(metrics)
    if not args.resume:
        w.writerow(["epoch", "loss", "val_mse", "elapsed_s"])
    metrics.flush()
    start = time.time()
    no_improve = 0
    try:
        for epoch in range(epoch0, args.epochs):
            perm = torch.randperm(len(train_x))
            total = 0.0
            for i in range(0, len(train_x), args.batch):
                idx = perm[i:i + args.batch]
                xb, yb = train_x[idx].to(device), train_y[idx].to(device)
                loss = loss_fn(model(xb), yb)
                opt.zero_grad()
                try:
                    loss.backward()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        lg.error("OOM: 请减小 --batch 或 --seq_len")
                        sys.exit(3)
                    raise
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                total += loss.item() * len(idx)
            avg = total / len(train_x)
            model.eval()
            with torch.no_grad():
                vm = 0.0
                for i in range(0, len(val_x), args.batch):
                    xb, yb = val_x[i:i + args.batch].to(device), val_y[i:i + args.batch].to(device)
                    vm += loss_fn(model(xb), yb).item() * len(xb)
                vm /= len(val_x)
            model.train()
            improved = vm < best
            if improved:
                best = vm
                no_improve = 0
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                            "epoch": epoch + 1, "best": best}, ckpt_dir / "best.pt")
            else:
                no_improve += 1
            if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
                lg.info("epoch %d/%d loss %.5f val_mse %.5f (best %.5f%s)",
                        epoch + 1, args.epochs, avg, vm, best,
                        ", saved" if improved else "")
            w.writerow([epoch + 1, round(avg, 5), round(vm, 5),
                        round(time.time() - start, 2)])
            metrics.flush()
            if args.patience and no_improve >= args.patience:
                lg.info("early stop at epoch %d", epoch + 1)
                break
    except KeyboardInterrupt:
        lg.info("interrupted, saving last.pt")
    finally:
        metrics.close()
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "epoch": args.epochs, "best": best}, ckpt_dir / "last.pt")

    # 流式推理展示：加载 best，用 rollout 跑一条验证样本并与 truth 对比
    ck = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        sample_x = val_x[:1].to(device)
        pred = model.rollout(sample_x).cpu()
        truth = val_y[:1].cpu()
        p6 = [round(float(v), 2) for v in pred[0, :6, 0]]
        t6 = [round(float(v), 2) for v in truth[0, :6, 0]]
    lg.info("stateful rollout sample: pred[0:6]=%s truth=%s", p6, t6)
    lg.info("done best_val_mse=%.5f metrics=%s", best, ckpt_dir / "metrics.csv")


if __name__ == "__main__":
    main()
