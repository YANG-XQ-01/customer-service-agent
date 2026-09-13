"""运行时指标注册表（进程内，线程安全）。

记录：请求量/状态码、延迟分布、LLM 调用与 token、工具调用、转人工、限流命中。
注意：数据存在进程内存里，只反映当前 worker；多 worker 部署时应改为
Redis 计数器或接入 Prometheus（见 docs/walkthrough.md 的监控章节）。
"""
import threading
import time
from collections import Counter, deque

# 千问 plus 估算单价（元 / 1K tokens），仅用于成本估算
PRICE_INPUT_YUAN_PER_1K = 0.0008
PRICE_OUTPUT_YUAN_PER_1K = 0.002

_lock = threading.Lock()
_start_time = time.time()

_requests: Counter = Counter()          # (path, status) -> 次数
_recent_requests: deque = deque(maxlen=500)   # (ts, path, status, ms)
_latencies: deque = deque(maxlen=2000)  # (ts, ms)

_llm_calls = 0
_llm_input_tokens = 0
_llm_output_tokens = 0

_tool_calls: Counter = Counter()        # 工具名 -> 次数
_handoffs: Counter = Counter()          # 触发场景 -> 次数
_rate_limited: Counter = Counter()      # scope -> 次数
_errors = 0


def record_request(path: str, status: int, duration_ms: float) -> None:
    with _lock:
        _requests[(path, int(status))] += 1
        now = time.time()
        _latencies.append((now, duration_ms))
        _recent_requests.append((now, path, int(status), duration_ms))


def record_llm_call(input_tokens: int | None, output_tokens: int | None) -> None:
    global _llm_calls, _llm_input_tokens, _llm_output_tokens
    with _lock:
        _llm_calls += 1
        _llm_input_tokens += int(input_tokens or 0)
        _llm_output_tokens += int(output_tokens or 0)


def record_tool(name: str) -> None:
    with _lock:
        _tool_calls[name] += 1


def record_handoff(trigger: str) -> None:
    with _lock:
        _handoffs[trigger] += 1


def record_rate_limit(scope: str) -> None:
    with _lock:
        _rate_limited[scope] += 1


def record_error() -> None:
    global _errors
    with _lock:
        _errors += 1


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(ratio * (len(ordered) - 1))))
    return round(ordered[index], 1)


def snapshot() -> dict:
    """生成看板所需的完整快照。"""
    with _lock:
        now = time.time()
        window = [ms for ts, ms in _latencies if now - ts <= 300]
        recent = [r for r in _recent_requests if now - r[0] <= 60]
        total_requests = sum(_requests.values())
        status_counts = Counter()
        for (_, status), count in _requests.items():
            status_counts[status] += count
        window_60s = [ms for ts, ms in _latencies if now - ts <= 60]
        qps = round(len(recent) / 60, 2)
        errors_5xx = sum(c for (_, s), c in _requests.items() if s >= 500)
        client_errors = sum(c for (_, s), c in _requests.items() if 400 <= s < 500)
        cost = (
            _llm_input_tokens / 1000 * PRICE_INPUT_YUAN_PER_1K
            + _llm_output_tokens / 1000 * PRICE_OUTPUT_YUAN_PER_1K
        )
        return {
            "uptime_seconds": round(now - _start_time, 1),
            "requests": {
                "total": total_requests,
                "qps_last_60s": qps,
                "status_counts": {str(k): v for k, v in sorted(status_counts.items())},
                "errors_5xx": errors_5xx,
                "client_errors_4xx": client_errors,
            },
            "latency_ms": {
                "samples": len(window),
                "p50": _percentile(window_60s, 0.50),
                "p95": _percentile(window_60s, 0.95),
                "p99": _percentile(window_60s, 0.99),
                "max": round(max(window_60s), 1) if window_60s else 0.0,
            },
            "llm": {
                "calls": _llm_calls,
                "input_tokens": _llm_input_tokens,
                "output_tokens": _llm_output_tokens,
                "estimated_cost_yuan": round(cost, 4),
            },
            "tools": dict(_tool_calls.most_common()),
            "handoffs": dict(_handoffs),
            "rate_limited": dict(_rate_limited),
            "unhandled_errors": _errors,
        }

