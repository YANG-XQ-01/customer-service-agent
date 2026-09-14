"""Redis 存储层：用户账号、登录令牌、聊天记录、连续失败计数。

为什么要外置到 Redis：
- 进程内存（memory.py）在服务重启或多 worker 部署时各自一份，
  会出现“刚聊过就忘”“连续失败计数失效”“工单看不到”等问题；
- Redis 让会话状态跨请求、跨进程、跨重启保持一致，并支持 TTL 自动过期。
"""
import hashlib
import json
import secrets
import time
import uuid

import redis

from app import config

# 统一 key 前缀，便于与其它业务共用 Redis 时隔离
_PREFIX = "cs"

_client = redis.Redis(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    db=config.REDIS_DB,
    decode_responses=True,      # 自动把Redis返回的bytes类型解码成字符串str
    socket_timeout=3,           # 读写Redis命令的超时时间：发送/接收数据超过3s就抛异常
    socket_connect_timeout=3,   # 连接Redis服务器的超时时间：建立连接超过3s就抛异常
    health_check_interval=30,   # 每30秒检查一次Redis服务器是否健康
)


def ping() -> bool:
    """健康检查用：Redis 是否可用。"""
    try:
        # redis-py 的 ping() 本身返回 True/False；这里只负责把异常转成 False
        return _client.ping()
    except Exception:
        return False


def _user_key(username: str) -> str:
    """生成用户信息对应的 Redis key"""
    return f"{_PREFIX}:user:{username.strip().lower()}"


def _token_key(token: str) -> str:
    """生成登录令牌对应的 Redis key"""
    return f"{_PREFIX}:token:{token}"


def _chat_key(user_id: str) -> str:
    """生成聊天记录对应的 Redis key"""
    return f"{_PREFIX}:chat:{user_id}"


def _attempts_key(user_id: str) -> str:
    """生成连续失败计数对应的 Redis key"""
    return f"{_PREFIX}:attempts:{user_id}"


def _hash_password(password: str, salt: str) -> str:
    """对密码进行加盐哈希处理,生产环境推荐使用 bcrypt / passlib，而不是裸 sha256，sha256 计算太快，暴力破解风险更高。"""
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def register_user(username: str, password: str) -> dict | None:
    """注册用户；用户名已存在返回 None。密码只存加盐哈希。

    并发安全：用 SET key value NX 做“按 key 的原子占位”——
    只有一个请求能把用户名写进去，其余请求拿到 False 直接返回 None。
    不用 HSET NX：一是 redis-py 的 hset() 不支持该参数，
    二是 HSET 的 NX 是按字段判断，可能出现“只补齐部分字段”的半成品记录。
    """
    key = _user_key(username)
    salt = secrets.token_hex(8)
    user_id = uuid.uuid4().hex[:12]
    record = {
        "user_id": user_id,
        "username": username.strip(),
        "salt": salt,
        "password_hash": _hash_password(password, salt),
    }
    created = _client.set(
        key, json.dumps(record, ensure_ascii=False), nx=True
    )
    if not created:
        return None
    return {"user_id": user_id, "username": username.strip()}


def _load_user(key: str) -> dict:
    """读取用户记录；兼容早期用 Hash 存储的旧数据。"""
    kind = _client.type(key)
    if kind == "string":
        raw = _client.get(key)
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}
    if kind == "hash":
        return _client.hgetall(key)
    return {}


def verify_login(username: str, password: str) -> dict | None:
    """校验账号密码；成功返回用户信息，失败返回 None。"""
    data = _load_user(_user_key(username))
    if not data:
        return None
    expected = data.get("password_hash", "")
    actual = _hash_password(password, data.get("salt", ""))
    if not secrets.compare_digest(expected, actual):
        return None
    return {"user_id": data["user_id"], "username": data.get("username", username)}


def create_token(user_id: str) -> str:
    """签发登录令牌（存 Redis，带 TTL）。"""
    token = secrets.token_urlsafe(24)
    _client.set(_token_key(token), user_id, ex=config.TOKEN_TTL)
    return token


def get_user_id_by_token(token: str) -> str | None:
    if not token:
        return None
    return _client.get(_token_key(token))


def revoke_token(token: str) -> None:
    if token:
        _client.delete(_token_key(token))


def add_message(user_id: str, role: str, content: str, max_messages: int = 200) -> None:
    """追加一条聊天记录；只保留最近 max_messages 条，并刷新 TTL。"""
    key = _chat_key(user_id)
    _client.rpush(key, json.dumps({"role": role, "content": content}, ensure_ascii=False))
    _client.ltrim(key, -max_messages, -1)
    _client.expire(key, config.CHAT_HISTORY_TTL)


def get_messages(user_id: str, limit: int = 50) -> list[dict]:
    """读取最近 limit 条聊天记录（旧 → 新）。"""
    raw = _client.lrange(_chat_key(user_id), -limit, -1)
    messages = []
    for item in raw:
        try:
            messages.append(json.loads(item))
        except json.JSONDecodeError:
            continue
    return messages


def clear_messages(user_id: str) -> None:
    _client.delete(_chat_key(user_id))


def get_attempts(user_id: str) -> int:
    return int(_client.get(_attempts_key(user_id)) or 0)


def increment_attempts(user_id: str) -> int:
    key = _attempts_key(user_id)
    value = _client.incr(key)
    _client.expire(key, config.CHAT_HISTORY_TTL)
    return int(value)


def reset_attempts(user_id: str) -> None:
    _client.delete(_attempts_key(user_id))


# ---------------- 限流与登录失败锁定 ----------------

def incr_window(key: str, window_seconds: int) -> int:
    """固定窗口计数：窗口内第一次访问时设置过期时间，返回当前次数。

    用于接口限流（如“每分钟最多 N 次”）和登录失败计数。
    """
    full_key = f"{_PREFIX}:rl:{key}"
    value = _client.incr(full_key)
    if value == 1 or _client.ttl(full_key) < 0:
        _client.expire(full_key, window_seconds)
    return int(value)


def get_window_count(key: str) -> int:
    return int(_client.get(f"{_PREFIX}:rl:{key}") or 0)


def clear_window(key: str) -> None:
    _client.delete(f"{_PREFIX}:rl:{key}")


def _login_fail_key(username: str) -> str:
    return f"login_fail:{username.strip().lower()}"


def get_login_failures(username: str) -> int:
    return get_window_count(_login_fail_key(username))


def record_login_failure(username: str, lock_seconds: int) -> int:
    """记录一次登录失败；返回累计失败次数。"""
    return incr_window(_login_fail_key(username), lock_seconds)


def clear_login_failures(username: str) -> None:
    clear_window(_login_fail_key(username))


# ---------------- 转人工工单（Redis 持久化） ----------------

def new_ticket_id() -> str:
    """生成唯一工单号：H + 时间戳 + 4 位随机。"""
    return "H" + time.strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:4].upper()


def _ticket_key(ticket_id: str) -> str:
    return f"{_PREFIX}:ticket:{ticket_id}"


_TICKET_INDEX = f"{_PREFIX}:tickets:index"


def create_handoff_ticket(ticket_id: str, **fields) -> dict:
    """写入工单并登记到索引（按时间倒序查询用）。"""
    ticket = {
        "ticket_id": ticket_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **fields,
    }
    _client.set(
        _ticket_key(ticket_id),
        json.dumps(ticket, ensure_ascii=False),
        ex=config.TICKET_TTL,
    )
    _client.zadd(_TICKET_INDEX, {ticket_id: time.time()})
    # 只用索引里保留最近 500 张，避免无限增长
    _client.zremrangebyrank(_TICKET_INDEX, 0, -501)
    return ticket


def get_handoff_ticket(ticket_id: str) -> dict | None:
    raw = _client.get(_ticket_key(ticket_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def list_handoff_tickets(limit: int = 100) -> list[dict]:
    """人工工作台用：按时间倒序返回最近工单，顺带清理过期索引。"""
    ids = _client.zrevrange(_TICKET_INDEX, 0, limit - 1)
    tickets = []
    stale = []
    for ticket_id in ids:
        ticket = get_handoff_ticket(ticket_id)
        if ticket is None:
            stale.append(ticket_id)
            continue
        tickets.append(ticket)
    if stale:
        _client.zrem(_TICKET_INDEX, *stale)
    return tickets
