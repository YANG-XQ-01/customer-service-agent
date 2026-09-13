"""Redis 存储层：用户账号、登录令牌、聊天记录、连续失败计数。

为什么要外置到 Redis：
- 进程内存（memory.py）在服务重启或多 worker 部署时各自一份，
  会出现“刚聊过就忘”“连续失败计数失效”“工单看不到”等问题；
- Redis 让会话状态跨请求、跨进程、跨重启保持一致，并支持 TTL 自动过期。
"""
import hashlib
import json
import secrets
import uuid

import redis

from app import config

# 统一 key 前缀，便于与其它业务共用 Redis 时隔离
_PREFIX = "cs"

_client = redis.Redis(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    db=config.REDIS_DB,
    decode_responses=True,
    socket_timeout=3,
    socket_connect_timeout=3,
    health_check_interval=30,
)


def ping() -> bool:
    """健康检查用：Redis 是否可用。"""
    try:
        return bool(_client.ping())
    except Exception:
        return False


def _user_key(username: str) -> str:
    return f"{_PREFIX}:user:{username.strip().lower()}"


def _token_key(token: str) -> str:
    return f"{_PREFIX}:token:{token}"


def _chat_key(user_id: str) -> str:
    return f"{_PREFIX}:chat:{user_id}"


def _attempts_key(user_id: str) -> str:
    return f"{_PREFIX}:attempts:{user_id}"


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def register_user(username: str, password: str) -> dict | None:
    """注册用户；用户名已存在返回 None。密码只存加盐哈希。"""
    key = _user_key(username)
    if _client.exists(key):
        return None
    salt = secrets.token_hex(8)
    user_id = uuid.uuid4().hex[:12]
    _client.hset(
        key,
        mapping={
            "user_id": user_id,
            "username": username.strip(),
            "salt": salt,
            "password_hash": _hash_password(password, salt),
        },
    )
    return {"user_id": user_id, "username": username.strip()}


def verify_login(username: str, password: str) -> dict | None:
    """校验账号密码；成功返回用户信息，失败返回 None。"""
    data = _client.hgetall(_user_key(username))
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

