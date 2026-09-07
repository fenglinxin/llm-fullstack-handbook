# -*- coding: utf-8 -*-
"""
第45章 线上问题闭环排查：日志分析 + 告警规则（一键脚本）

【层级】L2（可观测工具：规则告警引擎，可接真实日志流）
【环境依赖】Python 3.10+，标准库
【核心逻辑】模拟 200 条服务日志（info/warn/error+latency），规则：
error 占比>10% 或 p95 延迟>300ms 触发告警；输出级别分布、p95 与告警清单。
【关键参数】--n 200（默认）；--err_rate 0.08（默认，模拟输入的错误率）。
【避坑】真实监控用 Prometheus/OpenTelemetry；告警要配抑制与恢复通知；
延迟统计用分位数不要用均值。
【运行结果示例】
$ python monitor_demo.py
lines=200 error_rate=14.5% p95=356ms
alerts: [error_rate=14.5% > 10%, p95_latency=356ms > 300ms]（--err_rate 0.15）
【高频报错 Top5】
1. 日志格式不统一：先归一化解析；2. p95 算错：排序后取 95% 位置；
3. 告警风暴：加静默窗口；4. 漏检：规则要覆盖错误码/超时；5. 时区：统一 UTC。
【输出解读】alerts 列表为空且 error%<阈值 = 服务健康；有告警=按手册处理。
【工程改造方向】接真实日志（filebeat）+ 告警 webhook；应急手册自动化（runbook）。
"""

import argparse
import random
import statistics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--err_rate", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    random.seed(a.seed)
    lines = []
    for i in range(a.n):
        r = random.random()
        lv = "info" if r > a.err_rate + 0.05 else ("warn" if r > a.err_rate else "error")
        lat = random.uniform(10, 60) if lv == "info" else random.uniform(80, 500)
        lines.append((lv, lat, "req_%d" % i))
    err = sum(1 for lv, _, _ in lines if lv == "error")
    lats = sorted(lat for _, lat, _ in lines)
    p95 = lats[int(len(lats) * 0.95)]
    alerts = []
    if err / len(lines) > 0.10:
        alerts.append("error_rate=%.1f%% > 10%%" % (100 * err / len(lines)))
    if p95 > 300:
        alerts.append("p95_latency=%.0fms > 300ms" % p95)
    print("lines=%d error_rate=%.1f%% p95=%.0fms" % (len(lines), 100 * err / len(lines), p95))
    print("alerts:", alerts if alerts else "none（服务健康）")


if __name__ == "__main__":
    main()
