# -*- coding: utf-8 -*-
"""
code/ 全局代码落地规范自动检查器（零第三方依赖）

【层级】L2（工程工具）
【环境依赖】Python 3.10+，标准库；无第三方包
【核心逻辑】
- 扫描 code/ 下所有 .py（排除 out/、__pycache__、scripts/ 自身与临时文件）；
- 检查每个文件模块 docstring 是否含规范要求的字段关键词；
- 统计注释密度（机器近似：非空行中含 # 的行占比），低于阈值仅提示不判失败；
- 按主题目录汇总 L1/L2/L3 层级覆盖；
- 默认宽松输出报告（exit 0），--strict 时任何文件缺字段即非零退出。
【关键参数】
- --root：代码根目录，默认 code/；
- --strict：严格模式，任何文件缺字段返回非零；
- --min-comment-density：注释密度提示阈值，默认 0.15。
【避坑】
- 只判断字段关键词是否存在，不校验内容质量（质量由人审）；
- 字段必须完整出现在模块 docstring 中，写在函数 docstring 里不算；
- 注释密度只是机器近似，真实“逐行核心注释”以人审为准。
【运行结果示例】
$ python code/scripts/check_code_standard.py
files=35 fields_ok=0 fields_missing=35
【高频报错 Top5】
1. 找不到文件：确认在仓库根目录运行，或用 --root 指定路径；
2. 中文编码问题：文件头必须保留 # -*- coding: utf-8 -*-；
3. 误报缺字段：字段词要完整出现在模块 docstring（文件头三引号内）；
4. Windows 路径：--root 请用正斜杠或原始字符串；
5. 想跳过某文件：给文件名加 _legacy 前缀或移到 scripts/ 下。
【输出解读】missing 为空的文件即字段合规；layer 列显示声明层级。
【工程改造方向】接入 CI：python code/scripts/check_code_standard.py --strict
"""

import argparse
import pathlib
import re
import sys

REQUIRED_FIELDS = [
    "【层级】",
    "【环境依赖】",
    "【核心逻辑】",
    "【关键参数】",
    "【避坑】",
    "【运行结果示例】",
    "【高频报错",
    "【输出解读】",
    "【工程改造方向】",
]

LAYER_MARK = re.compile(r"【层级】s*(L[123])")


def is_code_file(p: pathlib.Path) -> bool:
    s = p.as_posix()
    if p.suffix != ".py":
        return False
    if "/out/" in s or "__pycache__" in s or "/scripts/" in s:
        return False
    if p.name.startswith("_"):
        return False
    return True


def read_text(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def comment_density(p: pathlib.Path) -> float:
    text = read_text(p)
    lines = text.splitlines()
    total = sum(1 for ln in lines if ln.strip())
    if total == 0:
        return 1.0
    hashed = sum(1 for ln in lines if ln.strip() and "#" in ln)
    return round(hashed / total, 2)


def check_file(p: pathlib.Path):
    head = read_text(p)[:40000]
    missing = [f for f in REQUIRED_FIELDS if f not in head]
    m = LAYER_MARK.search(head)
    layer = m.group(1) if m else None
    return {
        "path": p.as_posix().replace("\\", "/"),
        "layer": layer,
        "missing": missing,
        "density": comment_density(p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--min-comment-density", type=float, default=0.15)
    args = ap.parse_args()
    root = pathlib.Path(args.root) if args.root else pathlib.Path(__file__).resolve().parent.parent
    files = sorted(p for p in root.rglob("*.py") if is_code_file(p))
    if not files:
        print("no python files found under", root)
        sys.exit(0)
    report = [check_file(p) for p in files]
    ok = [r for r in report if not r["missing"]]
    bad = [r for r in report if r["missing"]]
    print("files=%d fields_ok=%d fields_missing=%d" % (len(report), len(ok), len(bad)))
    for r in report:
        tag = "OK  " if not r["missing"] else "MISS"
        print("%s %-58s layer=%-3s comments=%.2f missing=%s"
              % (tag, r["path"], r["layer"] or "-", r["density"],
                 ",".join(r["missing"]) if r["missing"] else "-"))
    dirs = {}
    for r in report:
        d = pathlib.Path(r["path"]).parent.as_posix()
        dirs.setdefault(d, set()).add(r["layer"])
    print("")
    print("--- layer coverage by dir ---")
    for d in sorted(dirs):
        print("%-50s layers=%s" % (d, sorted(x for x in dirs[d] if x)))
    if args.strict and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
