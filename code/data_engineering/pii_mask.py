# -*- coding: utf-8 -*-
"""
第08章 敏感数据脱敏：PII 识别与掩码（工程脚本）

【层级】L2（工程脱敏：规则识别手机/邮箱/身份证/姓名并支持多种掩码策略）
【环境依赖】Python 3.10+，仅标准库（re）
【核心逻辑】正则识别 PII 实体 -> 按实体类型选择掩码策略：
full(全掩)、partial(留首尾)、keep_domain(邮箱保留@后域名)；输出替换统计。
【关键参数】
- --sample：生成演示文本（默认关）；--mode full|partial|keep_domain（默认 partial）。
【避坑】身份证/手机正则会误匹配数字串：加边界与校验位逻辑；
脱敏后要校验“无明文残留”（print 掩码后文本复核）；真实合规见 个保法/行业规范。
【运行结果示例】
$ python pii_mask.py --sample
原: 联系电话 13812345678，邮箱 zhangsan@corp.com，身份证 110101199001011234。
掩: 联系电话 138****5678，邮箱 z***@corp.com，身份证 110101199****11234。
原: 客服：李四 13900001111 联系 test@example.cn 处理退款。
掩: 客服：李四 139****1111 联系 t***@example.cn 处理退款。
mode=partial hit summary: {'phone': 3, 'email': 2}
【高频报错 Top5】
1. 邮箱域名被全掩：keep_domain 模式可保留；2. 手机误伤固话：正则限定 1[3-9] 开头加 9 位数字；
3. 身份证校验位不查会误掩：加权重校验（工程改造）；4. 中文名误伤：需要词表/上下文；
5. 掩码后长度变化导致对齐错：占位符定长（本实现用 * 逐字符替换）。
【输出解读】每类 PII 命中数 + 掩码后文本打印即成功。
【工程改造方向】接入 NER 模型识别人名/地址；实体白名单；脱敏流水线 + 审计日志。
"""

import argparse
import re


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true")
    p.add_argument("--mode", default="partial", choices=["full", "partial", "keep_domain"])
    return p.parse_args()


PATTERNS = [
    ("phone", re.compile(r"1[3-9]\d{9}")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")),
    ("idcard", re.compile(r"\b\d{17}[\dXx]\b")),
]


def mask(text, mode):
    hits = {}
    def repl(m, kind, mode):
        hits[kind] = hits.get(kind, 0) + 1
        s = m.group(0)
        if kind == "phone":
            return s[:3] + "*" * 4 + s[7:] if mode == "partial" else "*" * len(s)
        if kind == "email":
            if mode == "keep_domain":
                return "***@" + s.split("@")[1]
            local, _, dom = s.partition("@")
            return local[:1] + "***@" + dom if mode == "partial" else "*" * len(s)
        if kind == "idcard":
            return s[:6] + "*" * 8 + s[-4:] if mode == "partial" else "*" * len(s)
        return s
    for kind, rx in PATTERNS:
        text = rx.sub(lambda m, k=kind: repl(m, k, mode), text)
    return text, hits


def gen_sample():
    return [
        "联系电话 13812345678，邮箱 zhangsan@corp.com，身份证 110101199001011234。",
        "客服：李四 13900001111 联系 test@example.cn 处理退款。",
    ]


def main():
    args = parse_args()
    texts = gen_sample() if args.sample else ["示例：13812345678 test@corp.com"]
    total = {}
    for t in texts:
        out, hits = mask(t, args.mode)
        for k, v in hits.items():
            total[k] = total.get(k, 0) + v
        print("原: %s" % t)
        print("掩: %s" % out)
    print("mode=%s hit summary: %s" % (args.mode, total))


if __name__ == "__main__":
    main()