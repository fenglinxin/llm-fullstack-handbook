# -*- coding: utf-8 -*-
"""
世界模型工程化训练引擎（L2）：旋转环境多步推演

【层级】L2（工程标准：可直接用于个人状态转移建模任务的训练脚本）
【环境依赖】
- Python 3.10+；PyTorch 2.x（建议 >=2.1）
- 安装：pip install torch==2.2.2；CPU 可跑，CUDA 自动可用
【核心逻辑】
- 环境：相位匀速旋转，状态 s=[sin θ, cos θ]，动作 a=角速度，s' 解析可算；
- 模型：DynamicsNet（残差 MLP next = s + f(s,a)）；
- 指标：单步 val MSE + 多步 rollout 误差（与解析真值比，验证误差累积）；
- 工程点：参数校验、日志（控制台+文件）、seed/device、train/val 固定切分、
  best/last checkpoint、断点续训、early stop、CSV 指标。
【关键参数】（默认值 / 推荐值 / 适配场景）
- --hidden 64（默认；复杂环境 128-256）：MLP 宽度；
- --n_train 2000/--n_val 400：样本数；--steps 300（默认）；
- --lr 1e-2（默认；1e-3~1e-2）；--batch 256；
- --rollout_steps 20（默认）：多步推演评测长度；
- --resume：续训；--patience 0（默认关）。
【避坑】
- 状态必须是周期友好的表示（sin/cos），别直接回归裸角度 θ；
- 多步 rollout 误差随步数增长是必然的，评测固定 rollout_steps 才能对比；
- 训练/验证同环境不同随机种子即可，无需额外领域外数据；
- rollout 首步用真值、后续全用模型预测（closed-loop）才是真推演。
【运行结果示例】（真实运行，CPU）
$ python world_model_engine.py --steps 300 --log_dir out_wm
INFO [world_model_engine] device=cpu train=2000 val=400 rollout_steps=20
INFO [world_model_engine] step 200 loss 0.00017 val_mse 0.00015 rollout_err@20=0.0274 (best 0.00015, saved)
INFO [world_model_engine] step 300 loss 0.00011 val_mse 0.00009 rollout_err@20=0.0389 (best 0.00009, saved)
INFO [world_model_engine] done best_val_mse=0.00009 metrics=out_wm/metrics.csv
（val_mse<1e-4、rollout_err@20 在 0.02-0.05 量级 = 单步与多步推演都工作）
【高频报错 Top5】
1. rollout 轨迹发散：动作/状态未归一化；修复：状态 [-1,1]、动作限幅；
2. loss 不降：lr 太大或数据泄漏（训练里混进验证）；修复：lr 1e-2、固定切分；
3. 角度回归 NaN：把 θ 直接当目标；修复：只回归 sin/cos（本引擎已如此）；
4. resume 后指标跳变：验证集不固定；修复：不要改 --seed；
5. 单步 loss 很小但 rollout 差：误差累积正常，训练用噪声注入可缓解（L3）。
【输出解读】
- val_mse<1e-4 + rollout err 远小于 1 = 引擎工作正常；
- rollout err 约 5-20 倍于单步 err 属正常，优化手段见 world_model_opt.py。
【工程改造方向】
- 换真实环境：提供 (state, action, next_state) 三元组即可；
- 视觉世界模型：状态换成 VAE latent，模型不变；
- 规划：把 rollout 封装成可微分规划器做 CEM/交叉熵搜索。
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

from world_model_demo import DynamicsNet, rollout


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_val", type=int, default=400)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--rollout_steps", type=int, default=20)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--log_dir", default="out_wm")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def make_env_data(n, seed):
    """旋转环境数据：θ_{t+1}=θ_t+ω，状态用 sin/cos。"""
    g = torch.Generator()
    g.manual_seed(seed)
    theta = torch.rand(n, generator=g) * 6.283
    omega = torch.rand(n, generator=g) * 0.5 + 0.2
    s = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
    a = omega.unsqueeze(-1)
    ns = torch.stack([torch.sin(theta + omega), torch.cos(theta + omega)], dim=-1)
    return s, a, ns


def rollout_err(model, steps, device):
    """从 s0=[0,1]、a=0.3 出发 rollout，与解析真值比平均误差。"""
    s0 = torch.tensor([[0.0, 1.0]], device=device)
    a0 = torch.tensor([[0.3]], device=device)
    traj = rollout(model, s0, a0, steps=steps)
    t = torch.arange(steps + 1).float() * 0.3
    truth = torch.stack([torch.sin(t), torch.cos(t)], dim=-1)
    return (traj[:, 0] - truth).pow(2).mean().sqrt().item()


def setup_logging(log_dir):
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("wm_engine")
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
    if args.steps <= 0 or args.n_train <= 0:
        print("[config error] steps/n_train 必须为正", file=sys.stderr)
        sys.exit(2)
    lg = setup_logging(args.log_dir)
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    s, a, ns = make_env_data(args.n_train, args.seed)
    vs, va, vns = make_env_data(args.n_val, args.seed + 1)
    lg.info("device=%s train=%d val=%d rollout_steps=%d",
            device, len(s), len(vs), args.rollout_steps)

    model = DynamicsNet(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    step0, best = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step0, best = ck.get("step", 0), ck.get("best", float("inf"))
        lg.info("resumed from %s at step %d", args.resume, step0)

    ckpt_dir = pathlib.Path(args.log_dir)
    metrics = open(ckpt_dir / "metrics.csv",
                   "a" if args.resume else "w", newline="", encoding="utf-8")
    w = csv.writer(metrics)
    if not args.resume:
        w.writerow(["step", "loss", "val_mse", "rollout_err", "elapsed_s"])
    metrics.flush()
    start = time.time()
    no_improve = 0
    try:
        for step in range(step0, args.steps):
            idx = torch.randint(0, len(s), (args.batch,))
            sb, ab, nb = s[idx].to(device), a[idx].to(device), ns[idx].to(device)
            loss = loss_fn(model(sb, ab), nb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % 50 == 0 or step + 1 == args.steps:
                model.eval()
                with torch.no_grad():
                    vm = loss_fn(model(vs.to(device), va.to(device)),
                                 vns.to(device)).item()
                    re = rollout_err(model, args.rollout_steps, device)
                model.train()
                improved = vm < best
                if improved:
                    best = vm
                    no_improve = 0
                    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                                "step": step + 1, "best": best}, ckpt_dir / "best.pt")
                else:
                    no_improve += 1
                lg.info("step %d loss %.5f val_mse %.5f rollout_err@%d=%.4f (best %.5f%s)",
                        step + 1, loss.item(), vm, args.rollout_steps, re, best,
                        ", saved" if improved else "")
                w.writerow([step + 1, round(loss.item(), 5), round(vm, 5), round(re, 4),
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
    lg.info("done best_val_mse=%.5f metrics=%s", best, ckpt_dir / "metrics.csv")


if __name__ == "__main__":
    main()
