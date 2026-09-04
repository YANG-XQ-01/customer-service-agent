"""极简会话记忆：把每个会话的聊天记录存在进程内存里。

为什么先这么做：
- 零依赖、零配置，马上能跑，适合先理解“记忆”这件事本身；
- 代价（两个明确的限制，面试常考）：
  1. 服务重启，记忆就丢；
  2. 部署多个进程时，各进程各存各的，不共享。
  这两个限制到后面阶段会用 Redis / 数据库解决，这里先不引入。

数据结构：session_id -> 消息列表
每条消息：{"role": "user" 或 "assistant", "content": "文字内容"}
"""
from app import config

# 全局内存仓库。FastAPI 单进程内所有请求共享这一个字典。
_SESSIONS: dict[str, list[dict[str, str]]] = {}


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

