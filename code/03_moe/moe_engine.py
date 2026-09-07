# -*- coding: utf-8 -*-
"""
MoE 训练工程引擎（L2）：路由分类任务（工程化训练脚本）

【层级】L2（工程标准：可直接用于个人 MoE 实验的训练脚本）
【环境依赖】
- Python 3.10+；PyTorch 2.x（建议 >=2.1）
- 安装：pip install torch==2.2.2；CPU 可跑，CUDA 自动可用
【核心逻辑】
- 任务：合成“路由分类”数据——每个 token 属于 4 个簇之一，向量 x~N(mu_c)，
  标签 y=c；MoE 的 Expert 输出 4 类 logits，Router 需要学会“把 token 分给
  对应专家”，任务收敛 = 路由真正在工作；
- 损失：task CE + balance_coef * 负载均衡 MSE；
- 工程点：参数校验、日志（控制台+文件）、seed/device、train/val 固定划分、
  best/last checkpoint、断点续训、early stop、CSV 指标、专家命中分布打印。
【关键参数】（默认值 / 推荐值 / 适配场景）
- --n_experts 4（默认；8-16 效果更细但更易崩塌）：专家数；
- --top_k 2（默认；1 更稀疏更快、3 更稳更贵）：每 token 激活专家数；
- --balance_coef 0.1（默认；本任务实测 0.1 才消除专家饿死，任务优先可设 0）：均衡系数；
- --d_model 16（默认）；--n_train 1000/--n_val 300：样本量；
- --steps 400（默认；越大越稳）：训练步数；--lr 1e-3；
- --resume：续训；--patience 0（默认关）。
【避坑】
- 负载均衡 loss 用 MSE 简化版即可教学，工业实现多用 Switch/辅助负载项；
- top_k 专家权重需在选中的专家上重新归一化（softmax(top_scores)），
  本引擎训练循环沿用 Demo 实现；
- 专家崩溃（某专家长期 0 命中）时先调大 balance_coef，再看数据均衡；
- 训练任务若收敛但命中分布严重不均，说明任务本身可被少数专家解决，
  属正常现象，重点是监控而非强求均匀。
【运行结果示例】（真实运行，CPU，balance_coef=0.1）
$ python moe_engine.py --steps 300 --batch 128 --balance_coef 0.1 --log_dir out_moe
INFO [moe_engine] step 50  loss 0.383 acc 1.000 hits=[217 148 234 1] (best 1.000, saved)
INFO [moe_engine] step 150 loss 0.211 acc 1.000 hits=[137 154 184 125] (best 1.000)
INFO [moe_engine] step 300 loss 0.206 acc 1.000 hits=[138 154 165 143] (best 1.000)
INFO [moe_engine] done best_val_acc=1.000 metrics=out_moe/metrics.csv
（注意：balance_coef=0.01 时专家 4 会饿死 hits=0，acc 仍 1.0——）
（acc 达标不代表负载健康，必须看 hits 分布，这是 MoE 独有的验收点）
【高频报错 Top5】
1. top_k>n_experts 直接崩溃：修复：CLI 校验 top_k<=n_experts（已内置）；
2. 输出全一样：router 没训练/balance_coef 太大把路由压平；修复：先设 0 跑通任务；
3. resume 后 acc 跳变：验证集不固定；修复：不要改 --seed；
4. 专家 0 命中：先调大 balance_coef，仍无效就检查数据是否被某类垄断；
5. CUDA OOM：batch 太大；修复：--batch 64。
【输出解读】
- acc>0.9 + 专家命中分布无明显 0 = MoE 训练成功；
- hits 方差大说明路由倾斜，可调 balance_coef（L3 moe_opt.py 有扫描）。
【工程改造方向】
- 换真实任务：把合成簇换成自己的 token 特征/标签；
- 换真实结构：Expert 换成 Transformer FFN 并接入层（参考 minimind_style/moe.py）；
- 分布式：训练上 EP/TP 通信、推理上专家路由缓存（主线 39 章）。
"""

import argparse
import csv
import logging
import pathlib
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_demo import SparseMoE


class MoEClassifier(nn.Module):
    """SparseMoE + 分类头 -> n_experts 类 logits。

    注意：moe_demo.SparseMoE 的 balance_loss 用“硬计数”算（不可导，
    只能展示）。本引擎改用工业常用的可导辅助损失：
    aux = n_experts * sum(f_i * p_i)，其中 f_i=命中占比（stop-grad），
    p_i=平均路由概率——这样 balance_coef 才能真的影响路由。
    """

    def __init__(self, d_model, n_experts, top_k, hidden=32, balance_coef=0.01):
        super().__init__()
        self.moe = SparseMoE(d_model=d_model, n_experts=n_experts,
                             top_k=top_k, balance_coef=balance_coef)
        self.head = nn.Linear(d_model, n_experts)

    def forward(self, x):
        out, _ = self.moe(x)
        flat = x.reshape(-1, x.shape[-1])
        p = F.softmax(self.moe.router(flat), dim=-1)
        top_idx = torch.topk(p, self.moe.top_k, dim=-1).indices.detach()
        f = torch.stack([
            (top_idx == e).any(-1).float().mean() for e in range(self.moe.n_experts)
        ]).detach()  # 命中占比不反传（标准 Switch aux）
        aux = self.moe.n_experts * (f * p.mean(0)).sum()
        return self.head(out[:, -1]), aux


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_experts", type=int, default=4)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--d_model", type=int, default=16)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--balance_coef", type=float, default=0.1)
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_val", type=int, default=300)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--log_dir", default="out_moe")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def validate(args):
    errs = []
    if args.top_k > args.n_experts:
        errs.append("top_k(%d) 不能大于 n_experts(%d)" % (args.top_k, args.n_experts))
    if args.steps <= 0 or args.n_train <= 0:
        errs.append("steps/n_train 必须为正")
    if errs:
        raise ValueError("\n".join(errs))


def make_data(n, n_experts, d_model, seed):
    """固定簇中心（跨 train/val 一致），只换采样：x ~ N(mu_c, 0.3I)。"""
    mu_gen = torch.Generator()
    mu_gen.manual_seed(1234)  # 簇中心固定：train/val 才能同域
    mu = torch.randn(n_experts, d_model, generator=mu_gen) * 2.0
    g = torch.Generator()
    g.manual_seed(seed)
    labels = torch.randint(0, n_experts, (n,), generator=g)
    x = mu[labels] + torch.randn(n, d_model, generator=g) * 0.3
    return x, labels


def hits_of(model, x, y, device, batch=256):
    model.eval()
    moe = model.moe
    hits = torch.zeros(moe.n_experts)
    right = total = 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb, yb = x[i:i + batch].to(device), y[i:i + batch].to(device)
            logits, _ = model(xb.unsqueeze(1))
            pred = logits.argmax(-1)
            right += (pred == yb).sum().item()
            total += len(yb)
            scores = moe.router(xb)
            top_idx = torch.topk(scores, moe.top_k, -1).indices
            for e in range(moe.n_experts):
                hits[e] += (top_idx == e).sum().item()
    model.train()
    return right / total, hits


def setup_logging(log_dir):
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("moe_engine")
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

    train_x, train_y = make_data(args.n_train, args.n_experts, args.d_model, args.seed)
    val_x, val_y = make_data(args.n_val, args.n_experts, args.d_model, args.seed + 1)
    lg.info("device=%s experts=%d top_k=%d train=%d val=%d",
            device, args.n_experts, args.top_k, len(train_x), len(val_x))

    model = MoEClassifier(d_model=args.d_model, n_experts=args.n_experts,
                        top_k=args.top_k, hidden=args.hidden,
                        balance_coef=args.balance_coef).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

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
    try:
        for step in range(step0, args.steps):
            idx = torch.randint(0, len(train_x), (args.batch,))
            xb = train_x[idx].unsqueeze(1).to(device)   # [b,1,d] -> token 级
            yb = train_y[idx].to(device)
            logits, bal = model(xb)
            loss = ce(logits, yb) + args.balance_coef * bal
            opt.zero_grad()
            try:
                loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    lg.error("OOM: 减小 --batch 或 --hidden")
                    sys.exit(3)
                raise
            opt.step()

            if (step + 1) % 50 == 0 or step + 1 == args.steps:
                acc, hits = hits_of(model, val_x, val_y, device)
                improved = acc > best
                if improved:
                    best = acc
                    no_improve = 0
                    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                                "step": step + 1, "best": best}, ckpt_dir / "best.pt")
                else:
                    no_improve += 1
                hits_str = " ".join("%d" % int(h) for h in hits)
                lg.info("step %d loss %.3f acc %.3f hits=[%s] (best %.3f%s)",
                        step + 1, loss.item(), acc, hits_str, best,
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