# -*- coding: utf-8 -*-
"""
第11章 增量预训练方案：领域语料继续预训练（一键脚本）

【层级】L1（一键运行：python continue_pretrain.py，CPU 1-3 分钟）
【环境依赖】Python 3.10+；PyTorch 2.x（pip install torch==2.2.2）；复用 minimind_style 代码
【核心逻辑】加载已有 out/pretrain.pt（缺失自动跑 150 步基座）；把含新词表的
领域语料（重复 15 次让新 token 学够）拼到语料后；词表扩展时拷贝旧 embedding/head
权重、新行随机初始化；小 lr 继续训练，保存 out/pretrain_v2.pt。
【关键参数】--steps 300（默认）；--lr 2e-4（增量训练必须远小于从零训练）；
--dim 64（需与基座一致才能续训）。
【避坑】增量 lr 过大/步数过多=灾难遗忘；词表变化必须先做权重扩展再训练；
领域语料太少时新 token 学不动，重复 N 次是常见工程技巧。
【运行结果示例】
$ python continue_pretrain.py --steps 300
[vocab extend] 新增 15 个 token 的 embedding/head 已初始化
step 0 loss 8.6072 | 公司知识库：回句环连保律。
step 120 loss 2.8109 | 公司知识库：内部文档系统统一在哪里照
step 299 loss 1.0766 | 公司知识库：内部文档系统统一走权限网
saved -> .../minimind_style/out/pretrain_v2.pt
【高频报错 Top5】
1. size mismatch：未做词表扩展直接 load；修复：本脚本已实现扩展逻辑；
2. 新词仍是 UNK：领域语料重复次数不够；修复：重复 15-30 次；
3. 通用能力丢失：lr 太大；修复：--lr 1e-4~2e-4；
4. 基座缺失：脚本自动预训练 150 步兜底；5. CPU 慢：--dim 32 验证。
【输出解读】看到领域 prompt 生成里出现“知识库”等新词 = 增量生效；
通用 prompt 仍能输出语料句子 = 遗忘可控。
【工程改造方向】接真实领域语料（ch04-10 数据工程输出）；增量+SFT 两段式；
评测遗忘（ch19 overfit_diagnoser 扩展）。
"""

import argparse
import pathlib
import random
import subprocess
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent / "minimind_style"
sys.path.insert(0, str(ROOT))

EXTRA = [
    "问：内部文档在哪里查？答：请登录公司知识库系统查看。",
    "公司知识库：内部文档系统统一走权限网关。",
    "公司知识库：工单超过二十四小时未处理会自动升级。",
]


def ensure_base_ckpt():
    ck = ROOT / "out" / "pretrain.pt"
    if not ck.exists():
        print("[bootstrap] 未找到基座 checkpoint，自动预训练 150 步...")
        subprocess.run([sys.executable, "pretrain.py", "--steps", "150"],
                       cwd=str(ROOT), check=True)
    return ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from tokenizer import CharTokenizer
    from model import TinyConfig, TinyGPT
    import pretrain as P

    ck = ensure_base_ckpt()
    data = torch.load(str(ck), map_location="cpu", weights_only=True)
    base_cfg = TinyConfig(**data["cfg"])
    corpus = (ROOT / "data" / "corpus.txt").read_text(encoding="utf-8")
    new_corpus = corpus + ("\n".join(EXTRA) + "\n") * 15
    tok = CharTokenizer()
    tok.fit([new_corpus])
    ids = tok.encode(new_corpus)
    windows = P.make_windows(ids, args.seq)
    cfg = TinyConfig(vocab_size=len(tok.vocab), dim=base_cfg.dim,
                     n_layers=base_cfg.n_layers, n_heads=base_cfg.n_heads,
                     max_seq=args.seq)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print("vocab base=%d continue=%d windows=%d" % (base_cfg.vocab_size, len(tok.vocab), len(windows)))

    # 词表扩展：旧权重拷贝，新行保持初始化
    model = TinyGPT(cfg)
    if base_cfg.vocab_size == cfg.vocab_size:
        model.load_state_dict(data["model"])
    else:
        base_model = TinyGPT(base_cfg)
        base_model.load_state_dict(data["model"])
        sd = {k: v.clone() for k, v in base_model.state_dict().items()
              if k not in ("token_emb.weight", "lm_head.weight")}
        model.load_state_dict(sd, strict=False)
        n_old = base_cfg.vocab_size
        with torch.no_grad():
            model.token_emb.weight[:n_old].copy_(base_model.token_emb.weight)
            model.lm_head.weight[:n_old].copy_(base_model.lm_head.weight)
        print("[vocab extend] 新增 %d 个 token 的 embedding/head 已初始化" % (cfg.vocab_size - n_old))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for step in range(args.steps):
        batch = [random.choice(windows) for _ in range(16)]
        x = torch.tensor(batch)
        logits = model(x[:, :-1])
        loss = loss_fn(logits.reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 60 == 0 or step == args.steps - 1:
            gen = P.sample_text(model, tok, "cpu", prompt="公司知识库：", max_new=12)
            print("step %d loss %.4f | %s" % (step, loss.item(), gen[:34]))
    out = ROOT / "out" / "pretrain_v2.pt"
    torch.save({"model": model.state_dict(),
                "cfg": {"vocab_size": cfg.vocab_size, "dim": cfg.dim,
                        "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                        "max_seq": cfg.max_seq}}, out)
    tok.save(ROOT / "out" / "vocab_v2.json")
    print("saved ->", out)


if __name__ == "__main__":
    main()
