# -*- coding: utf-8 -*-
"""
Transformer 落地代码 L2：序列反转任务工程化训练引擎

【层级】L2（工程标准：可直接用于个人项目/二次开发的训练脚本）
【环境依赖】
- Python 3.10+；PyTorch 2.x（建议 >=2.1，2.0 亦可运行）
- 安装：pip install torch==2.2.2；GPU 版按 PyTorch 官网 CUDA 对应版本安装
- CPU 可跑通全部功能；CUDA 可用时自动切 GPU（--device cuda 可强制）

【核心逻辑】
- 复用 train_seq2seq_mini.py 的 MiniSeq2Seq 与 make_batch（反转任务）；
- 工程化点：参数校验、结构化日志（控制台+文件）、随机种子、设备自动选择、
  评估集固定、最优模型保存、断点续训、early stop、CSV 指标导出、异常友好提示；
- 训练主循环：随机 batch 迭代 total_steps 步，每 eval_every 步评估一次。

【关键参数】（默认值 / 推荐值 / 适配场景）
- --steps 300（默认；加大模型后 500-1000）：总训练步数；
- --batch 32（默认；显存小用 16，追求稳定用 64）：批大小；
- --lr 1e-3（默认；loss 震荡降到 5e-4）：学习率；
- --d_model 32（默认；追求精度 64-128）：模型宽度；
- --n_heads 4（默认；必须整除 d_model）：头数；
- --max_src 6（默认；任务难度，8-10 更难更慢）：源序列长度；
- --max_len 20（默认；必须 >= max_src+2）：位置编码长度；
- --eval_every 25（默认）：评估间隔；--patience 0（默认关，5-10 开启 early stop）；
- --resume：从 checkpoint 续训；--log_dir：日志/模型/CSV 输出目录。

【避坑】
- d_model 必须能被 n_heads 整除，否则启动即报错（已做参数校验）；
- 断点续训时 --steps 是“总步数”，已训练的步数会自动跳过；
- GPU OOM 会捕获并提示减小 batch，而不是裸堆栈；
- early stop 只保存验证集 accuracy 最高的 checkpoint。

【运行结果示例】（真实运行，CPU）
$ python seq2seq_engine.py --steps 300 --eval_every 50 --log_dir out_engine
INFO [seq2seq_engine] step 50  loss 2.5486 val_acc 0.1667 (best 0.1667, saved)
INFO [seq2seq_engine] step 150 loss 1.3421 val_acc 0.3958 (best 0.3958, saved)
INFO [seq2seq_engine] step 250 loss 0.6810 val_acc 0.6250 (best 0.6250, saved)
INFO [seq2seq_engine] step 300 loss 0.5075 val_acc 0.8646 (best 0.8646, saved)
INFO [seq2seq_engine] done. best_val_acc=0.8646 metrics=out_engine/metrics.csv
（loss 单调下降、val_acc 最终 >0.8 = 训练成功）

【高频报错 Top5】
1. d_model 与 n_heads 不整除：本引擎启动时 ValueError 提示，改 --n_heads；
2. checkpoint 版本不兼容（size mismatch）：换模型结构后旧 ckpt 不能直接续训；
   修复：重新训练或仅加载 model 前缀（load_state_dict strict=False 自查）；
3. CUDA OOM：日志提示 Reduce batch size，改 --batch 16 或 --max_src 5；
4. Windows 下日志中文乱码：控制台 GBK 时用 PYTHONIOENCODING=utf-8；
5. resume 后 loss 不降：确认 --steps 是总步数，续训从 checkpoint 步数继续而非重头。

【输出解读】
- loss 单调下降、val_acc 上升即正常；
- 每个 eval 步打印 best 与是否保存新 ckpt；
- metrics.csv 可用 Excel/pandas 打开画 loss/acc 曲线。

【工程改造方向】
- 换数据：把 make_batch 替换成自己的 Dataset（文本/图像 token 序列）；
- 接入项目：把 checkpoint 加载封装成服务端推理类，或导出 onnx；
- 迭代优化：加深层数、加 warmup、梯度裁剪、EMA 权重（见 attention_opt.py L3）。
"""

import argparse
import csv
import json
import logging
import pathlib
import random
import sys
import time

import torch

from train_seq2seq_mini import MiniSeq2Seq, make_batch


def parse_args():
    p = argparse.ArgumentParser(description="序列反转任务工程化训练引擎 (L2)")
    p.add_argument("--vocab", type=int, default=20)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--max_len", type=int, default=20)
    p.add_argument("--max_src", type=int, default=6)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval_n", type=int, default=16)
    p.add_argument("--eval_every", type=int, default=25)
    p.add_argument("--patience", type=int, default=0, help="early stop 步数，0=关闭")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--log_dir", default="out_engine")
    p.add_argument("--resume", default=None, help="checkpoint 路径，续训")
    return p.parse_args()


def validate(args):
    """启动前参数校验：宁可启动失败，不可中途崩溃。"""
    errs = []
    if args.d_model % args.n_heads != 0:
        errs.append("d_model(%d) 必须能被 n_heads(%d) 整除" % (args.d_model, args.n_heads))
    if args.max_len < args.max_src + 2:
        errs.append("max_len(%d) 至少需要 max_src+2=%d" % (args.max_len, args.max_src + 2))
    if args.steps <= 0 or args.batch <= 0:
        errs.append("steps/batch 必须为正整数")
    if not (0 < args.lr < 1):
        errs.append("lr 建议在 (0,1) 区间")
    if errs:
        raise ValueError("\n".join(errs))


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_logging(log_dir):
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("seq2seq_engine")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(levelname)s [%(module)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(pathlib.Path(log_dir) / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def make_eval_set(seed, n, max_src, vocab):
    """固定评估集：同一 seed 永远同一批样本，保证对比公平。"""
    g = torch.Generator()
    g.manual_seed(seed)
    src = torch.randint(2, vocab, (n, max_src), generator=g)
    tgt = torch.flip(src, dims=[1])
    return src, tgt


def evaluate(model, src, tgt, device):
    model.eval()
    with torch.no_grad():
        pred = model.generate(src.to(device), max_len=tgt.shape[1])[:, 1:]
        acc = (pred.cpu() == tgt).float().mean().item()
    model.train()
    return acc


def save_checkpoint(path, model, opt, step, best_acc, cfg):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "best_acc": best_acc,
        "cfg": vars(cfg),
    }, path)


def load_checkpoint(path, model, opt, device):
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # 兼容 PyTorch <2.0 无 weights_only 参数
        ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model"])
    if opt is not None and "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"])
    return ck.get("step", 0), ck.get("best_acc", 0.0)


def main():
    args = parse_args()
    try:
        validate(args)
    except ValueError as e:
        print("[config error]", e, file=sys.stderr)
        sys.exit(2)

    logger = setup_logging(args.log_dir)
    device = pick_device(args.device)
    logger.info("device=%s seed=%d", device, args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model = MiniSeq2Seq(vocab=args.vocab, d_model=args.d_model,
                        n_heads=args.n_heads, max_len=args.max_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    eval_src, eval_tgt = make_eval_set(args.seed + 1, args.eval_n, args.max_src, args.vocab)
    step0, best_acc = 0, 0.0
    if args.resume:
        step0, best_acc = load_checkpoint(args.resume, model, opt, device)
        logger.info("resumed from %s at step %d (best %.4f)", args.resume, step0, best_acc)

    ckpt_dir = pathlib.Path(args.log_dir)
    csv_path = ckpt_dir / "metrics.csv"
    resume_mode = args.resume is not None and csv_path.exists()
    csv_file = open(csv_path, "a" if resume_mode else "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not resume_mode:
        writer.writerow(["step", "loss", "val_acc", "elapsed_s"])
    csv_file.flush()

    start = time.time()
    no_improve = 0
    try:
        for step in range(step0, args.steps):
            src, tgt_in, tgt = make_batch(args.batch, max_src=args.max_src, vocab=args.vocab)
            src, tgt_in, tgt = src.to(device), tgt_in.to(device), tgt.to(device)
            logits = model(src, tgt_in)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
            opt.zero_grad()
            try:
                loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("CUDA OOM: 请减小 --batch 或 --max_src")
                    sys.exit(3)
                raise
            opt.step()

            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                acc = evaluate(model, eval_src, eval_tgt, device)
                improved = acc > best_acc
                if improved:
                    best_acc = acc
                    save_checkpoint(ckpt_dir / "best.pt", model, opt, step + 1, best_acc, args)
                    no_improve = 0
                else:
                    no_improve += 1
                logger.info("step %d loss %.4f val_acc %.4f (best %.4f%s)",
                            step + 1, loss.item(), acc, best_acc,
                            ", saved" if improved else "")
                writer.writerow([step + 1, round(loss.item(), 4), round(acc, 4),
                                 round(time.time() - start, 2)])
                csv_file.flush()
                if args.patience and no_improve >= args.patience:
                    logger.info("early stop at step %d", step + 1)
                    break
    except KeyboardInterrupt:
        save_checkpoint(ckpt_dir / "interrupted.pt", model, opt, step + 1, best_acc, args)
        logger.info("interrupted, checkpoint saved")
    finally:
        csv_file.close()

    save_checkpoint(ckpt_dir / "last.pt", model, opt, args.steps, best_acc, args)
    logger.info("done. best_val_acc=%.4f metrics=%s", best_acc, csv_path)


"""
工程改造方向（正文见 docstring）：
1. 换数据集：重写 make_batch 返回 (src, tgt_in, tgt)；
2. 接推理服务：best.pt 加载后封装 predict()；
3. 后续优化：warmup/梯度裁剪/EMA 可参考 L3 attention_opt.py。
"""

if __name__ == "__main__":
    main()
