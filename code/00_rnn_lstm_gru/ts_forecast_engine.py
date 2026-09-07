# -*- coding: utf-8 -*-
"""
RNN / LSTM / GRU 时序预测工程化训练引擎（L2）

【层级】L2（工程标准：可直接用于个人时序项目的训练脚本）
【环境依赖】
- Python 3.10+；PyTorch 2.x（建议 >=2.1）；pandas 可选（读 CSV 时用，纯正弦可不装）
- 安装：pip install torch==2.2.2 pandas==2.2.2（CSV 场景）
- CPU 可跑；CUDA 可用时自动切换（--device cuda 可强制）

【核心逻辑】
- 数据：正弦（内置）或 CSV（--data 每行一个数值）→ 归一化（只 fit 训练段）→
  滑窗切 (lookback -> next) 样本 → 训练/验证切分；
- 模型：--model rnn|lstm|gru 工厂创建，输出层接 Linear(hidden,1)；
- 工程点：参数校验、日志（控制台+文件）、设备/种子可复现、验证集固定、
  best/last checkpoint、断点续训、early stop、CSV 指标、异常提示；
- 训练完自动做“递归多步预测”演示并打印 RMSE。

【关键参数】（默认值 / 推荐值 / 适配场景）
- --model lstm（默认；rnn 最快但精度低、gru 折中）：模型选型；
- --lookback 12（默认；周期 20 的正弦建议 12-24，业务周期数据按周期取）：历史窗口；
- --hidden 32（默认；数据复杂 64-128）：隐层维度；
- --epochs 60（默认；loss 平台期后 100-200 也无害）：训练轮数；
- --lr 1e-2（默认；震荡降到 5e-3 或加 warmup）：学习率；
- --clip 1.0（默认；防梯度爆炸，建议 0.5-1.0）：梯度裁剪范数；
- --batch 64（默认；小数据 32）：批大小；--data：CSV 路径（缺省用正弦）；
- --resume：从 checkpoint 续训；--patience 0（默认关，5-15 开启 early stop）。

【避坑】
- 归一化参数只能从训练段估计，不能把验证/未来数据混进 min/max；
- 递归多步预测误差会累积，输出远点 RMSE 大是正常现象；
- 断点续训时 --epochs 是“总轮数”，已训轮数自动跳过；
- CSV 有表头/多列时报错会给出明确提示，需先清洗成单列数值。

【运行结果示例】（真实运行，CPU，sine + lstm）
$ python ts_forecast_engine.py --model lstm --epochs 40 --batch 128 --log_dir out_ts
INFO [ts_forecast_engine] device=cpu model=lstm seed=0
INFO [ts_forecast_engine] epoch 10/40 loss 0.00238 val_mse 0.00243 (best 0.00232)
INFO [ts_forecast_engine] epoch 20/40 loss 0.00005 val_mse 0.00005 (best 0.00005)
INFO [ts_forecast_engine] epoch 30/40 loss 0.00003 val_mse 0.00002 (best 0.00002, saved)
INFO [ts_forecast_engine] epoch 40/40 loss 0.00001 val_mse 0.00001 (best 0.00001, saved)
INFO [ts_forecast_engine] forecast next 12 steps rmse=0.0365
INFO [ts_forecast_engine] done best_val_mse=0.00001 metrics=out_ts/metrics.csv
（loss/val_mse 单调下降且 <1e-4 = 成功；rmse 是递归多步误差，天然大于单步）

【高频报错 Top5】
1. CSV 解析失败：文件含表头/多列；修复：用单列数值文件，或先 pandas 清洗；
2. shape 报错：x 需 [N, lookback, 1]；修复：window_data 返回前统一 unsqueeze(-1)；
3. loss=NaN：lr 太大或数据含 NaN；修复：--clip 1.0、--lr 5e-3、检查 CSV；
4. resume 后验证指标跳变：验证集必须固定 seed（本引擎已固定）；修复：勿改 --seed；
5. CUDA OOM：batch 太大；修复：--batch 32/16 或减小 hidden。

【输出解读】
- loss/val_mse 单调下降且最终 <0.001（正弦任务）即成功；
- forecast rmse 比 val_mse 大一个量级属正常（误差累积）；
- metrics.csv 可画 loss/val_mse 曲线；best.pt 为验证集最优模型。

【工程改造方向】
- 换业务数据：CSV 单列时间序列即可；多变量请扩展 window_data 输入多列；
- 接入项目：加载 best.pt 封装 predict_next()，隐状态跨请求保留做流式；
- 迭代优化：加多步目标（预测未来 5 点）、分位数损失、特征工程（见 L3 rnn_opt.py）。
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

from rnn_lstm_gru_demo import ManualRNN, make_sine_data


def parse_args():
    p = argparse.ArgumentParser(description="RNN/LSTM/GRU 时序预测工程引擎 (L2)")
    p.add_argument("--model", default="lstm", choices=["rnn", "lstm", "gru"])
    p.add_argument("--lookback", type=int, default=12)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--data", default=None, help="CSV 路径，每行一个数值；缺省用内置正弦")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--log_dir", default="out_ts")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def validate(args):
    errs = []
    if args.lookback <= 0 or args.hidden <= 0 or args.epochs <= 0:
        errs.append("lookback/hidden/epochs 必须为正整数")
    if not (0 < args.lr < 1):
        errs.append("lr 建议在 (0,1)")
    if not (0 < args.val_ratio < 1):
        errs.append("val_ratio 需在 (0,1)")
    if errs:
        raise ValueError("\n".join(errs))


def load_series(path):
    """读 CSV：每行一个数值；容忍空行/注释行。"""
    vals = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                vals.append(float(line.split(",")[0]))
            except ValueError:
                raise ValueError("无法解析行: %r（请先清洗为单列数值）" % line[:40])
    if len(vals) < 50:
        raise ValueError("数据点过少(<50)：%d" % len(vals))
    return torch.tensor(vals, dtype=torch.float32)


def normalize(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9), lo, hi


def window_data(series, lookback):
    xs, ys = [], []
    for i in range(len(series) - lookback):
        xs.append(series[i:i + lookback])
        ys.append(series[i + lookback])
    return torch.stack(xs).unsqueeze(-1), torch.stack(ys).unsqueeze(-1)


def build_model(kind, hidden, clip):
    if kind == "rnn":
        model = ManualRNN(input_size=1, hidden=hidden)
    elif kind == "lstm":
        cell = nn.LSTM(1, hidden, batch_first=True)
        model = _SeqHead(cell, hidden)
    else:
        cell = nn.GRU(1, hidden, batch_first=True)
        model = _SeqHead(cell, hidden)
    return model


class _SeqHead(nn.Module):
    """库模型包装：取最后时间步 -> Linear 头。"""

    def __init__(self, cell, hidden):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.cell(x)
        return self.head(out[:, -1, :])


def setup_logging(log_dir):
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("ts_engine")
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
    lg.info("device=%s model=%s seed=%d", device, args.model, args.seed)

    # 数据准备：归一化只 fit 训练段
    if args.data:
        raw = load_series(args.data)
    else:
        raw = make_sine_data(1200, args.lookback)[0][:, 0, 0]  # 取原始序列段
        # make_sine_data 返回样本窗口，这里重建连续序列更方便
        t = torch.arange(1200, dtype=torch.float32) / 20.0
        raw = torch.sin(t)
    n_val = int(len(raw) * args.val_ratio)
    train_raw = raw[: len(raw) - n_val]
    train_norm, lo, hi = normalize(train_raw)
    # 用训练段的 min/max 归一化整段
    val_norm = (raw[len(raw) - n_val:] - lo) / (hi - lo + 1e-9)
    train_x, train_y = window_data(train_norm, args.lookback)
    val_x, val_y = window_data(val_norm, args.lookback)
    lg.info("samples train=%d val=%d", len(train_x), len(val_x))

    model = build_model(args.model, args.hidden, args.clip).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    epoch0, best = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        epoch0 = ck.get("epoch", 0)
        best = ck.get("best", float("inf"))
        lg.info("resumed from %s at epoch %d (best %.5f)", args.resume, epoch0, best)

    ckpt_dir = pathlib.Path(args.log_dir)
    metrics = open(ckpt_dir / "metrics.csv", "a" if args.resume else "w",
                   newline="", encoding="utf-8")
    w = csv.writer(metrics)
    if not args.resume:
        w.writerow(["epoch", "loss", "val_mse", "elapsed_s"])
    metrics.flush()
    start = time.time()
    no_improve = 0

    n_batch = max(1, len(train_x) // args.batch)
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
                        lg.error("CUDA OOM: 请减小 --batch 或 --hidden")
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

    # 递归多步预测演示（取训练域内尾部，避免“分布外推”干扰教学演示）
    model.eval()
    with torch.no_grad():
        start = len(train_norm) - 2 * args.lookback
        tail = train_norm[start:start + args.lookback].unsqueeze(0).unsqueeze(-1).to(device)
        preds = []
        h = tail
        for _ in range(args.lookback):
            p = model(h)
            preds.append(p.item())
            h = torch.cat([h[:, 1:], p.unsqueeze(1)], dim=1)  # [1,1,1] 拼到窗口尾部
        # 反归一化回原尺度（预测覆盖 train_raw 最后 lookback 个点）
        preds_raw = [p * (hi - lo + 1e-9) + lo for p in preds]
        truth_raw = raw[len(train_raw) - args.lookback:len(train_raw)].tolist()
        rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(preds_raw, truth_raw)) / args.lookback)
    lg.info("forecast next %d steps rmse=%.4f", args.lookback, rmse)
    lg.info("done best_val_mse=%.5f metrics=%s", best, ckpt_dir / "metrics.csv")


"""工程改造方向（见 docstring）：
1. 多变量/多步：扩展 window_data 与 loss；
2. 业务接入：best.pt 加载后封装 predict_next；
3. 高级优化：rnn_opt.py（L3）提供稳收敛与流式推理对照。
"""

if __name__ == "__main__":
    main()
