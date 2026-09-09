"""评测脚本：跑分任务完成率 / 轨迹正确率 / 平均耗时 / 估算成本。

用法（在项目根目录）：
    python -m eval.run_eval              # 跑全部用例
    python -m eval.run_eval --limit 5    # 只跑前 5 个
    python -m eval.run_eval --case refund-threshold-high

原理：
- 在进程内直接调用 LangGraph 图（不经 HTTP，减少环境干扰）；
- 用日志钩子收集每次工具调用（名称/参数）和路由意图；
- 关键词检查答案；工单仓库检查是否真的转人工及触发场景；
- token 用量按千问公开价估算成本（结果仅供参考）。
"""
import argparse
import ast
import asyncio
import json
import logging
import re
import sys
import time
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage

# 让控制台能正常打印中文
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
CASES_FILE = BASE_DIR / "eval" / "cases.json"
REPORT_FILE = BASE_DIR / "eval" / "report.json"

logger = logging.getLogger("customer-service")
# 关键：不设 INFO 级别的话，工具的 INFO 日志会在到达钩子前被过滤掉
logger.setLevel(logging.INFO)

# 千问 plus 估算单价（元 / 1K tokens），仅用于成本估算
PRICE_INPUT_YUAN_PER_1K = 0.0008
PRICE_OUTPUT_YUAN_PER_1K = 0.002


class CaptureHandler(logging.Handler):
    """临时日志钩子：记录评测期间打出的所有日志行。"""

    def __init__(self):
        super().__init__()
        self.setLevel(logging.INFO)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _parse_intent(lines: list[str]) -> str | None:
    """从日志里取最后一次路由意图。"""
    matches = [
        re.search(r"路由节点: intent=(\w+)", line) for line in lines
    ]
    for m in reversed(matches):
        if m:
            return m.group(1)
    return None


def _parse_tool_calls(lines: list[str]) -> list[dict]:
    """把日志里的工具调用还原成 [{name, args}] 列表。"""
    calls: list[dict] = []
    current: dict | None = None
    for line in lines:
        m = re.search(r">>> 调用工具: (\w+)", line)
        if m:
            current = {"name": m.group(1), "args": {}}
            calls.append(current)
            continue
        m = re.search(r">>> 工具参数: (.+)$", line)
        if m and current is not None:
            try:
                current["args"] = ast.literal_eval(m.group(1))
            except Exception:
                current["args"] = {}
    return calls


def _tool_required_satisfied(required: dict, actual: list[dict]) -> tuple[bool, str]:
    """检查某个必需工具是否被调过（名称一致 + 关键参数匹配）。"""
    name = required["name"]
    args = required.get("args", {})
    candidates = [c for c in actual if c["name"] == name]
    if not candidates:
        return False, f"未调用工具 {name}（实际调用: {[c['name'] for c in actual]}）"
    if args:
        for key, value in args.items():
            if not any(c["args"].get(key) == value for c in candidates):
                return False, f"工具 {name} 的参数 {key}={value} 未匹配（实际参数: {[c['args'] for c in candidates]}）"
    return True, ""


def _messages_to_chat(history: list[dict[str, str]]):
    """与 app/main.py 相同的历史转消息逻辑。"""
    from langchain_core.messages import AIMessage

    messages = []
    for m in history:
        if m["role"] == "assistant":
            messages.append(AIMessage(content=m["content"]))
        else:
            messages.append(HumanMessage(content=m["content"]))
    return messages


def _read_tokens(message) -> tuple:
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        return usage.get("input_tokens"), usage.get("output_tokens")
    if usage is not None:
        return getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)
    return None, None


async def _run_one_case(case: dict) -> dict:
    """跑一个用例，返回评分所需全部信息。"""
    from app.graph import agent_graph
    from app.memory import (
        add_message,
        get_attempts,
        get_messages,
        increment_attempts,
        list_handoff_tickets,
        reset_attempts,
    )

    session_id = "eval-" + case["id"] + "-" + uuid.uuid4().hex[:6]
    reset_attempts(session_id)

    handler = CaptureHandler()
    logger.addHandler(handler)
    latencies: list[float] = []
    total_input = 0
    total_output = 0
    last_answer = ""
    try:
        for user_message in case["messages"]:
            history = get_messages(session_id)
            messages = _messages_to_chat(history)
            messages.append(HumanMessage(content=user_message))
            attempts = get_attempts(session_id)

            start = time.perf_counter()
            output = await agent_graph.ainvoke(
                {
                    "messages": messages,
                    "trace": [],
                    "session_id": session_id,
                    "attempts": attempts,
                }
            )
            latencies.append(time.perf_counter() - start)

            final_message = output["messages"][-1]
            last_answer = final_message.content
            in_tokens, out_tokens = _read_tokens(final_message)
            if in_tokens:
                total_input += in_tokens
            if out_tokens:
                total_output += out_tokens

            add_message(session_id, "user", user_message)
            add_message(session_id, "assistant", last_answer)

            intent = output.get("intent", "")
            if intent == "clarify" and not output.get("handoff_ticket_id"):
                increment_attempts(session_id)
            else:
                reset_attempts(session_id)
    finally:
        logger.removeHandler(handler)

    lines = handler.lines
    actual_intent = _parse_intent(lines)
    actual_tools = _parse_tool_calls(lines)

    # 是否真的产生了本会话的工单（含触发场景）
    new_ticket = None
    for ticket in list_handoff_tickets():
        if ticket.get("session_id") == session_id:
            new_ticket = ticket
            break

    return {
        "case": case,
        "answer": last_answer,
        "latencies": latencies,
        "total_input": total_input,
        "total_output": total_output,
        "actual_intent": actual_intent,
        "actual_tools": actual_tools,
        "new_ticket": new_ticket,
    }


def _grade(result: dict) -> dict:
    """按用例期望逐项打分，返回 {pass, reasons, detail}。"""
    case = result["case"]
    expect = case["expect"]
    reasons: list[str] = []

    if "intent" in expect and result["actual_intent"] != expect["intent"]:
        reasons.append(
            f"意图不符：期望 {expect['intent']}，实际 {result['actual_intent']}"
        )

    required = expect.get("tools_required", [])
    trajectory_ok = True
    for item in required:
        ok, msg = _tool_required_satisfied(item, result["actual_tools"])
        if not ok:
            trajectory_ok = False
            reasons.append(msg)

    answer = result["answer"] or ""
    for word in expect.get("must_contain", []):
        if word not in answer:
            reasons.append(f"答案缺少关键词：{word}")
    for word in expect.get("must_not_contain", []):
        if word in answer:
            reasons.append(f"答案不应出现：{word}")

    needs_human = result["new_ticket"] is not None
    if expect.get("needs_human") is not None and needs_human != expect["needs_human"]:
        reasons.append(f"转人工不符：期望 {expect['needs_human']}，实际 {needs_human}")
    trigger = expect.get("trigger")
    if trigger and result["new_ticket"] and result["new_ticket"].get("trigger") != trigger:
        reasons.append(
            f"触发场景不符：期望 {trigger}，实际 {result['new_ticket'].get('trigger')}"
        )

    passed = not reasons
    return {
        "passed": passed,
        "trajectory_ok": trajectory_ok and not any("工具" in r or "未调用" in r for r in reasons),
        "reasons": reasons,
    }


def _fmt_case(case: dict) -> str:
    return f"{case['id']}（{case.get('description', '')}）"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个用例")
    parser.add_argument("--case", type=str, default=None, help="只跑指定 id 的用例")
    args = parser.parse_args()

    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
    elif args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("没有可跑的用例")

    results = []  # 保存每条用例完整运行结果的列表
    total_latency = 0.0  # 总耗时（秒，浮点数）
    total_turns = 0  # 总对话轮次
    total_input = 0  # 输入token总数
    total_output = 0  # 输出token总数
    task_pass = 0  # 任务通过的用例数量（任务级通过率）
    trajectory_pass = 0  # 执行轨迹通过数量（过程/步骤是否正确）

    for index, case in enumerate(cases, start=1):
        print(f"\n[{index}/{len(cases)}] 运行: {_fmt_case(case)}")
        result = await _run_one_case(case)
        grade = _grade(result)
        results.append({**result, **grade})

        total_latency += sum(result["latencies"])
        total_turns += len(result["latencies"])
        total_input += result["total_input"]
        total_output += result["total_output"]
        if grade["passed"]:
            task_pass += 1
        if grade["trajectory_ok"]:
            trajectory_pass += 1

        status = "PASS" if grade["passed"] else "FAIL"
        print(f"    结果: {status} | 意图={result['actual_intent']} | "
              f"工具={[t['name'] for t in result['actual_tools']]}")
        if grade["reasons"]:
            for r in grade["reasons"]:
                print(f"    - {r}")

    n = len(cases)
    cost_yuan = (
        total_input / 1000 * PRICE_INPUT_YUAN_PER_1K
        + total_output / 1000 * PRICE_OUTPUT_YUAN_PER_1K
    )
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "task_pass_rate": round(task_pass / n, 4),
            "trajectory_pass_rate": round(trajectory_pass / n, 4),
            "avg_latency_seconds": round(total_latency / total_turns, 2) if total_turns else 0,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "estimated_cost_yuan": round(cost_yuan, 4),
        },
        "cases": [
            {
                "id": r["case"]["id"],
                "description": r["case"].get("description", ""),
                "passed": r["passed"],
                "trajectory_ok": r["trajectory_ok"],
                "reasons": r["reasons"],
                "actual_intent": r["actual_intent"],
                "actual_tools": r["actual_tools"],
                "avg_latency": round(sum(r["latencies"]) / len(r["latencies"]), 2)
                if r["latencies"] else 0,
                "answer_excerpt": (r["answer"] or "")[:120],
            }
            for r in results
        ],
    }
    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 50)
    print("评测报告")
    print("=" * 50)
    print(f"任务完成率:      {task_pass}/{n} = {report['metrics']['task_pass_rate']:.1%}")
    print(f"轨迹正确率:      {trajectory_pass}/{n} = {report['metrics']['trajectory_pass_rate']:.1%}")
    print(f"平均耗时(每轮):  {report['metrics']['avg_latency_seconds']} 秒")
    print(f"估算成本:        ¥{report['metrics']['estimated_cost_yuan']}")
    print(f"报告已保存:      {REPORT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
