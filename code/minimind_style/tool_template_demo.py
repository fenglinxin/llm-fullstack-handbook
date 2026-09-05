# -*- coding: utf-8 -*-
"""
工具调用 / 思考模板演示（P3：不改模型结构，先把“数据长什么样”讲清楚）

【定位】MiniMind 演进路线中的“工具 + 思考”阶段，本质是**模板工程**：
把 CoT 思考、工具调用、工具结果都编码成可训练的文本格式。
本文件用纯 Python 演示模板与两轮工具调用闭环，无需训练即可运行。

【环境依赖】Python 3.8+，仅标准库。

【运行】
python tool_template_demo.py --demo cot     # 思考模板样例
python tool_template_demo.py --demo tool    # 工具调用两轮闭环
python tool_template_demo.py --demo check   # 检查字符级 tokenizer 能否覆盖模板

【核心概念】
1. CoT 模板：把“思考过程”显式写进训练文本，让模型学会先推理再回答；
2. 工具模板：模型输出 <tool_call>{JSON}</tool_call>，外部执行器解析并返回
   <tool_result>，模型再基于结果生成最终答案；
3. 模板字符（<>、引号、花括号）若不在词表里，字符级 tokenizer 会变成 UNK，
   真实项目请用 BPE/ByteLevel 分词（本文件 check 模式会实测演示）。
"""

import argparse
import json


def build_cot_sample(question, think, answer):
    """构造一条带思考的 SFT 样本（对应主线 17 章模板一致性）。"""
    return {
        "q": question,
        "a": "<think>" + think + "</think><answer>" + answer + "</answer>",
    }


def parse_tool_call(text):
    """从模型输出中提取 <tool_call>{...}</tool_call> 并解析成 dict。"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def run_tool(name, args):
    """模拟工具执行器：真实项目里这里是代码/数据库/API 的封装。"""
    if name == "get_weather":
        return "上海，晴，26℃"
    if name == "get_time":
        return "2026-01-06 14:30"
    if name == "calc":
        return str(eval(args.get("expr", "0")))  # 教学示例：真实系统勿直接 eval
    return "未知工具"


def demo_cot():
    print("=== CoT 思考模板样例 ===")
    sample = build_cot_sample(
        "打印机连不上怎么办？",
        "先判断故障类型：连接问题。按“电源→网络→重启→工单”顺序排查。",
        "先检查电源和网络，然后重启打印机，并提交工单。",
    )
    print(json.dumps(sample, ensure_ascii=False, indent=2))
    print()
    print("训练时该样例与普通 SFT 完全一样：回答部分算 loss，指令部分 mask 掉。")
    print("区别只在文本格式里多了 <think>...</think> 标签——这就是'学会思考'的起点。")
    print()


def demo_tool():
    print("=== 工具调用两轮闭环（模板层） ===")
    # 第一轮：模型应输出工具调用
    model_out_1 = (
        "<think>用户问天气，需要先调用天气工具。</think>"
        '<tool_call>{"name": "get_weather", "args": {"city": "上海"}}</tool_call>'
    )
    print("[1] 模型输出:", model_out_1)
    call = parse_tool_call(model_out_1)
    print("[2] 解析结果:", call)
    result = run_tool(call["name"], call["args"]) if call else "解析失败"
    print("[3] 工具结果:", result)

    # 第二轮：把工具结果拼回上下文，模型生成最终回答
    prompt2 = (
        "<tool_result>" + result + "</tool_result>"
        "根据工具结果回答用户：上海今天天气怎么样？"
    )
    model_out_2 = "<answer>上海今天晴，26℃，适合出门。</answer>"
    print("[4] 二轮输入:", prompt2)
    print("[5] 最终回答:", model_out_2)
    print()
    print("真实训练数据 = 把[1][2][3][4]的文本按角色拼成一个多轮样本。")
    print("两轮模板缺一不可：不把工具结果喂回去，模型就无法'基于事实'作答。")
    print()


def demo_check():
    print("=== 字符级 tokenizer 覆盖检查 ===")
    import pathlib
    from tokenizer import CharTokenizer
    tok_path = pathlib.Path(__file__).resolve().parent / "out" / "vocab.json"
    if not tok_path.exists():
        print("未找到 out/vocab.json，请先运行 pretrain.py 生成词表。")
        return
    tok = CharTokenizer.load(tok_path)
    samples = [
        "<think>先检查电源</think>",
        '<tool_call>{"name": "get_weather"}</tool_call>',
        "问：打印机连不上怎么办？答：",
    ]
    for s in samples:
        ids = tok.encode(s)
        unk = sum(1 for i in ids if i == tok.unk_id)
        print("覆盖率 %.0f%% | UNK=%d | %s" % (100 * (1 - unk / len(ids)), unk, s))
    print()
    print("结论：中文问答字符大多在词表里；<>{} 引号等模板字符大概率是 UNK。")
    print("所以字符级分词只适合'最小闭环'教学，工具/思考模板请上 BPE 分词。")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", choices=["cot", "tool", "check", "all"], default="all")
    args = p.parse_args()
    if args.demo in ("cot", "all"):
        demo_cot()
    if args.demo in ("tool", "all"):
        demo_tool()
    if args.demo in ("check", "all"):
        demo_check()


"""
进阶改造 Prompt
1. 把 think 标签换成真实的 CoT 数据（先让大模型生成，再蒸馏给迷你模型）；
2. 给工具增加 schema 校验（参数类型/必填），失败时返回 <tool_error>；
3. 用本模板把 sft_data.py 扩展成带工具的多轮 jsonl，跑一遍 train_sft.py；
4. 尝试用已训练好的微型模型生成 <tool_call>，观察它对模板字符的复现能力；
5. 思考：eval 别用于工具参数、工具白名单、权限隔离如何落到工程里（主线 25/40 章）。
"""

if __name__ == "__main__":
    main()
