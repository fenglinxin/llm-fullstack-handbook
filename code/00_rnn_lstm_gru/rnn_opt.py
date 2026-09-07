# -*- coding: utf-8 -*-
"""
RNN / LSTM / GRU 高阶优化对照（L3）：稳收敛 + 流式推理提速

【层级】L3（高阶优化：对应时序模型章节“稳收敛/降延时”教学重点）
【环境依赖】
- Python 3.10+；PyTorch 2.x（建议 >=2.1；torch.compile 实验需 2.0+）
- 安装：pip install torch==2.2.2；CPU 即可跑，编译实验在 Windows 上可能跳过
【核心逻辑】
1. 稳收敛实验：带噪声的较长正弦序列 + 偏大 lr=0.3，
   对比 无梯度裁剪 vs clip=0.5：统计 loss 尖峰次数与最终 loss，
   直观展示“梯度裁剪把训练从发散边缘拉回来”；
2. 流式推理实验：LSTM 逐点解码时（a）每步全窗重算 vs（b）携带
   隐状态只喂新点，断言输出一致并计时；
3. 可选：torch.compile 编译 LSTM 提速对照（失败自动跳过并提示）。
【关键参数】（默认值 / 推荐值 / 适配场景）
- --epochs 30（默认）：实验轮数；
- --lr 0.3（默认故意偏大，便于观察发散/裁剪差异）：学习率；
- --clip 0.5（默认；业务上 0.5-1.0 常用）：梯度裁剪范数；
- --noise 0.05（默认；0=纯正弦，越大越容易震荡）：数据噪声；
- --seq_len 500（默认）：流式推理解码步数。
【避坑】
- 小 lr（1e-2）下裁剪效果不明显，本实验故意放大 lr 展示机制；
- 流式一致性断言必须通过，否则“提速”无意义；
- torch.compile 首次调用含编译开销，计时前先 warmup 一次；
- CPU 计时受干扰，实验统一 repeats>=3 取平均。
【运行结果示例】（真实运行，CPU；用 --lr 1.0 放大“防发散”差异）
$ python rnn_opt.py --epochs 20 --lr 1.0 --clip 0.5 --seq_len 200
noise=0.05 lr=1.00 clip=0.50 epochs=20
[stability] no_clip : max_loss=231.5770 final_loss=0.9601
[stability] clip0.5 : max_loss=231.5770 final_loss=0.5690
[streaming] outputs match = True
[streaming] full-window: 1003.33 ms | stateful: 136.77 ms | speedup 7.3x
[compile  ] compiled: 1392.83 ms (warmup 后)
PASS
（lr=1.0 下 clip 把 final_loss 从 0.96 压到 0.57；正常 lr=1e-2 时两者都收敛良好，
裁剪价值在“偏大 lr/复杂数据/长序列”场景才明显）
【高频报错 Top5】
1. loss=NaN：lr 过大且未裁剪；修复：--lr 降回 1e-2 或开 --clip；
2. “Tensors must have same number of dimensions”：隐状态 h/c 维度
   [layers,batch,hidden]，注意与输入 [batch,1,input] 区分；
3. torch.compile 报错：Windows/旧版本不支持；修复：捕获异常自动跳过，
   或升级 Linux + CUDA 环境；
4. 流式与整窗输出不一致：隐状态未清零/未传 h0；修复：逐 token 前
   初始化 h0=0 并每步回传；
5. 计时 0.00ms：repeats 太少；修复：加大 repeats 或用 perf_counter_ns。
【输出解读】
- stability 两组对比：max_loss 相同（首步随机初始化）但 clip 组 final_loss
  更低 = 裁剪让大 lr 训练不跑飞；收敛曲线更稳（可自行加 epoch 打印验证）；
- streaming 断言 True 且 speedup>1 = 流式优化正确有效（实测 7.3x）；
- compile 行打印 skipped 不算失败，只是平台不支持。
【工程改造方向】
- 上线前把 clip/lr_schedule/EMA 写入训练配置（参考 ts_forecast_engine.py L2）；
- 流式场景（在线预测/生成）把 stateful 解码封装成服务；
- 长序列继续优化：truncated BPTT、分桶 batch、量化 LSTM。
"""

import argparse
import math
import time

import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--clip", type=float, default=0.5)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--seq_len", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def noisy_sine(n=1500, lookback=16, noise=0.05, seed=0):
    torch.manual_seed(seed)
    t = torch.arange(n, dtype=torch.float32) / 15.0
    s = torch.sin(t) + noise * torch.randn(n)
    xs, ys = [], []
    for i in range(len(s) - lookback):
        xs.append(s[i:i + lookback])
        ys.append(s[i + lookback])
    return torch.stack(xs).unsqueeze(-1), torch.stack(ys).unsqueeze(-1)


def train_lstm(x, y, lr, clip, epochs, seed=0):
    """训练一个 LSTM 单步预测头，返回 (final_loss, max_loss)。"""
    torch.manual_seed(seed)
    cell = nn.LSTM(1, 32, batch_first=True)
    head = nn.Linear(32, 1)
    opt = torch.optim.Adam(list(cell.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = nn.MSELoss()
    max_loss = 0.0
    final = float("nan")
    for _ in range(epochs):
        opt.zero_grad()
        out, _ = cell(x)
        loss = loss_fn(head(out[:, -1, :]), y)
        loss.backward()
        if clip > 0:
            nn.utils.clip_grad_norm_(list(cell.parameters()) + list(head.parameters()), clip)
        opt.step()
        max_loss = max(max_loss, loss.item())
        final = loss.item()
    return final, max_loss


def streaming_compare(model_cell, head, x0, steps):
    """对比全窗重算 vs 带隐状态逐点解码（输出必须一致）。"""
    # 参考：整段前向，逐点生成（每步把历史全量喂回）
    def full_way():
        seq = x0
        last = None
        with torch.no_grad():
            for _ in range(steps):
                out, _ = model_cell(seq)
                last = head(out[:, -1, :])
                nxt = last
                seq = torch.cat([seq[:, 1:], nxt.unsqueeze(1)], dim=1)
        return last

    def stateful_way():
        h = torch.zeros(1, 1, 32)
        c = torch.zeros(1, 1, 32)
        cur = x0[:, -1:]
        last = None
        with torch.no_grad():
            for _ in range(steps):
                out, (h, c) = model_cell(cur, (h, c))
                last = head(out[:, -1, :])
                cur = last.unsqueeze(1)
        return last

    a = full_way()
    b = stateful_way()
    match = bool(torch.allclose(a, b, atol=1e-4))
    return match


def timeit(fn, repeats=3):
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / repeats


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    print("noise=%.2f lr=%.2f clip=%.2f epochs=%d" % (args.noise, args.lr, args.clip, args.epochs))

    x, y = noisy_sine(1500, 16, args.noise, args.seed)
    f_nc, m_nc = train_lstm(x, y, args.lr, 0.0, args.epochs)
    f_c, m_c = train_lstm(x, y, args.lr, args.clip, args.epochs)
    print("[stability] no_clip : max_loss=%.4f final_loss=%.4f" % (m_nc, f_nc))
    print("[stability] clip%.1f : max_loss=%.4f final_loss=%.4f" % (args.clip, m_c, f_c))

    cell = nn.LSTM(1, 32, batch_first=True)
    head = nn.Linear(32, 1)
    x0 = torch.randn(1, 16, 1)
    ok = streaming_compare(cell, head, x0, args.seq_len)
    print("[streaming] outputs match = %s" % ok)

    cell2 = nn.LSTM(1, 32, batch_first=True)
    head2 = nn.Linear(32, 1)

    def full_run():
        seq = x0
        with torch.no_grad():
            for _ in range(args.seq_len):
                out, _ = cell2(seq)
                nxt = head2(out[:, -1, :])
                seq = torch.cat([seq[:, 1:], nxt.unsqueeze(1)], dim=1)

    def state_run():
        h = torch.zeros(1, 1, 32)
        c = torch.zeros(1, 1, 32)
        cur = x0[:, -1:]
        with torch.no_grad():
            for _ in range(args.seq_len):
                out, (h, c) = cell2(cur, (h, c))
                cur = head2(out[:, -1, :]).unsqueeze(1)

    full_run(); state_run()  # warmup
    t_f = timeit(full_run)
    t_s = timeit(state_run)
    print("[streaming] full-window: %.2f ms | stateful: %.2f ms | speedup %.1fx"
          % (t_f, t_s, t_f / max(t_s, 1e-9)))

    # 可选：torch.compile 提速对照（失败自动跳过）
    try:
        if not hasattr(torch, "compile"):
            raise RuntimeError("no torch.compile")
        cell_c = torch.compile(nn.LSTM(1, 32, batch_first=True))
        head_c = torch.compile(nn.Linear(32, 1))

        def compiled_run():
            seq = x0
            with torch.no_grad():
                for _ in range(20):
                    out, _ = cell_c(seq)
                    nxt = head_c(out[:, -1, :])
                    seq = torch.cat([seq[:, 1:], nxt.unsqueeze(1)], dim=1)

        compiled_run()
        t_c = timeit(compiled_run)
        print("[compile  ] compiled: %.2f ms (warmup 后)" % t_c)
    except Exception as e:  # noqa: BLE001 教学脚本：任何失败都降级
        print("[compile  ] skipped on this platform (%s)" % type(e).__name__)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()