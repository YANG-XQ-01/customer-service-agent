"""进程内存：会话消息 + 连续失败计数 + 转人工工单。

为什么先这么做：
- 零依赖、零配置，马上能跑，适合先理解“记忆”这件事本身；
- 代价（两个明确的限制，面试常考）：
  1. 服务重启，记忆就丢；
  2. 部署多个进程时，各进程各存各的，不共享。
  这两个限制到后面阶段会用 Redis / 数据库解决，这里先不引入。

数据结构：session_id -> 消息列表
每条消息：{"role": "user" 或 "assistant", "content": "文字内容"}
"""
from datetime import datetime
import uuid

from app import config

# 全局内存仓库。FastAPI 单进程内所有请求共享这一个字典。
_SESSIONS: dict[str, list[dict[str, str]]] = {}
# 每个会话“连续没能理解/解答”的次数（成功回答会清零）
_ATTEMPTS: dict[str, int] = {}
# 转人工工单仓库：ticket_id -> 工单详情（人工工作台从这里读）
_HANDOFF_TICKETS: dict[str, dict] = {}


def get_messages(session_id: str) -> list[dict[str, str]]:
    """取某个会话的历史消息；没有则返回空列表（并顺手建档）。"""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = []
    return _SESSIONS[session_id]


def add_message(session_id: str, role: str, content: str) -> None:
    """往会话追加一条消息，并只保留最近 N 轮，防止历史无限膨胀。"""
    history = get_messages(session_id)
    history.append({"role": role, "content": content})

    # 1 轮 = 1 条 user + 1 条 assistant，所以最多保留 2 * N 条
    max_messages = config.MEMORY_MAX_TURNS * 2
    if len(history) > max_messages:
        _SESSIONS[session_id] = history[-max_messages:]


def get_attempts(session_id: str) -> int:
    """读取该会话连续失败次数（用于判断是否该转人工）。"""
    return _ATTEMPTS.get(session_id, 0)


def increment_attempts(session_id: str) -> int:
    """连续失败 +1，返回新值。"""
    _ATTEMPTS[session_id] = get_attempts(session_id) + 1
    return _ATTEMPTS[session_id]


def reset_attempts(session_id: str) -> None:
    """成功回答或转人工后清零（只清连续计数，不删聊天记录）。"""
    _ATTEMPTS.pop(session_id, None)


def new_ticket_id() -> str:
    """生成唯一工单号：H + 时间戳 + 4 位随机。"""
    return "H" + datetime.now().strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:4].upper()


def create_handoff_ticket(ticket_id: str, **fields) -> dict:
    """登记一张转人工工单，并补上创建时间。"""
    ticket = {
        "ticket_id": ticket_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **fields,
    }
    _HANDOFF_TICKETS[ticket_id] = ticket
    return ticket


def get_handoff_ticket(ticket_id: str) -> dict | None:
    """按工单号取一张工单。"""
    return _HANDOFF_TICKETS.get(ticket_id)


def list_handoff_tickets() -> list[dict]:
    """人工工作台用：最新的工单排最前。"""
    return sorted(
        _HANDOFF_TICKETS.values(),
        key=lambda t: t.get("created_at", ""),
        reverse=True,
    )
