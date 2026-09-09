"""评测脚本：任务完成率 / 轨迹正确率 / 平均耗时 / 估算成本。

用法（在项目根目录）：
    python -m eval.run_eval                 # 每个用例跑 1 轮
    python -m eval.run_eval --rounds 3      # 每个用例跑 3 轮（看稳定性）
    python -m eval.run_eval --limit 5       # 只跑前 5 个用例
    python -m eval.run_eval --case order-not-found

为什么要多轮：模型有随机性，单轮 100% 可能是运气。
--rounds 3 会给出每个用例的“通过率”，把时好时坏的用例暴露出来。
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

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
CASES_FILE = BASE_DIR / "eval" / "cases.json"
REPORT_FILE = BASE_DIR / "eval" / "report.json"

logger = logging.getLogger("customer-service")
logger.setLevel(logging.INFO)

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
    matches = [re.search(r"路由节点: intent=(\w+)", line) for line in lines]
    for m in reversed(matches):
        if m:
            return m.group(1)
    return None


def _parse_tool_calls(lines: list[str]) -> list[dict]:
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
    name = required["name"]
    args = required.get("args", {})
    candidates = [c for c in actual if c["name"] == name]
    if not candidates:
        return False, f"未调用工具 {name}（实际调用: {[c['name'] for c in actual]}）"
    if args:
        for key, value in args.items():
            if not any(c["args"].get(key) == value for c in candidates):
                return False, (
                    f"工具 {name} 的参数 {key}={value} 未匹配"
                    f"（实际参数: {[c['args'] for c in candidates]}）"
                )
    return True, ""


def _messages_to_chat(history: list[dict[str, str]]):
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
    """跑一个用例一次，返回评分所需信息。"""
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
        "actual_intent": _parse_intent(lines),
        "actual_tools": _parse_tool_calls(lines),
        "new_ticket": new_ticket,
    }


def _grade(result: dict) -> dict:
    case = result["case"]
    expect = case["expect"]
    reasons: list[str] = []

    if "intent" in expect and result["actual_intent"] != expect["intent"]:
        reasons.append(f"意图不符：期望 {expect['intent']}，实际 {result['actual_intent']}")

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

    return {
        "passed": not reasons,
        "trajectory_ok": trajectory_ok,
        "reasons": reasons,
    }


def _fmt_case(case: dict) -> str:
    return f"{case['id']}（{case.get('description', '')}）"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1, help="每个用例跑几轮（默认 1）")
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

    all_results: list[dict] = []
    for case in cases:
        print(f"\n运行: {_fmt_case(case)} x{args.rounds} 轮")
        for round_index in range(1, args.rounds + 1):
            result = await _run_one_case(case)
            grade = _grade(result)
            status = "PASS" if grade["passed"] else "FAIL"
            print(f"  轮{round_index}: {status} | 意图={result['actual_intent']} | "
                  f"工具={[t['name'] for t in result['actual_tools']]}")
            all_results.append({**result, **grade, "round": round_index})

    n_cases = len(cases)
    n_runs = len(all_results)
    total_latency = 0.0
    total_turns = 0
    total_input = 0
    total_output = 0
    task_pass_runs = 0
    trajectory_pass_runs = 0
    for r in all_results:
        total_latency += sum(r["latencies"])
        total_turns += len(r["latencies"])
        total_input += r["total_input"]
        total_output += r["total_output"]
        task_pass_runs += 1 if r["passed"] else 0
        trajectory_pass_runs += 1 if r["trajectory_ok"] else 0

    cost_yuan = (
        total_input / 1000 * PRICE_INPUT_YUAN_PER_1K
        + total_output / 1000 * PRICE_OUTPUT_YUAN_PER_1K
    )
    case_summaries = []
    for case in cases:
        related = [r for r in all_results if r["case"]["id"] == case["id"]]
        pass_count = sum(1 for r in related if r["passed"])
        reasons = []
        seen = set()
        for r in related:
            for reason in r["reasons"]:
                if reason not in seen:
                    seen.add(reason)
                    reasons.append(reason)
        case_summaries.append(
            {
                "id": case["id"],
                "description": case.get("description", ""),
                "rounds": args.rounds,
                "pass_count": pass_count,
                "pass_rate": round(pass_count / args.rounds, 4),
                "reasons": reasons,
                "answer_excerpt": (related[-1]["answer"] or "")[:120],
            }
        )

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rounds": args.rounds,
        "metrics": {
            "task_pass_rate": round(task_pass_runs / n_runs, 4),
            "trajectory_pass_rate": round(trajectory_pass_runs / n_runs, 4),
            "avg_latency_seconds": round(total_latency / total_turns, 2) if total_turns else 0,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "estimated_cost_yuan": round(cost_yuan, 4),
        },
        "cases": case_summaries,
    }
    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 50)
    print(f"评测报告（{n_cases} 用例 x {args.rounds} 轮 = {n_runs} 次运行）")
    print("=" * 50)
    print(f"任务完成率:      {task_pass_runs}/{n_runs} = {report['metrics']['task_pass_rate']:.1%}")
    print(f"轨迹正确率:      {trajectory_pass_runs}/{n_runs} = {report['metrics']['trajectory_pass_rate']:.1%}")
    print(f"平均耗时(每轮):  {report['metrics']['avg_latency_seconds']} 秒")
    print(f"估算成本:        ¥{report['metrics']['estimated_cost_yuan']}")
    print("不稳定用例（通过率 < 100%）：")
    flaky = [c for c in case_summaries if c["pass_rate"] < 1.0]
    if flaky:
        for c in flaky:
            print(f"  - {c['id']}: {c['pass_count']}/{c['rounds']}，原因: {c['reasons']}")
    else:
        print("  无")
    print(f"报告已保存: {REPORT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
