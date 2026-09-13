"""FastAPI 入口：电商智能客服 Agent。

处理链路：
网页发消息 -> 根据登录令牌取会话历史（Redis）-> 交给 LangGraph 图
-> 路由节点判断意图 -> 派给对应专家节点（内部可调工具）
-> 产出最终回答 -> 写回会话历史 -> 返回给网页

用户体系：注册/登录后签发令牌，聊天记录与连续失败计数都存 Redis，
因此刷新页面、重启服务、多 worker 都不会丢历史。
未登录仍可用（历史存进程内存），便于评测脚本直接调用。
"""
import logging
import sys
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app import config, metrics, store
from app.graph import agent_graph
from app.memory import (
    add_message,
    get_attempts,
    get_messages,
    increment_attempts,
    reset_attempts,
)

# Windows 控制台默认编码可能是 GBK，先切成 UTF-8，保证日志中文可读
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("customer-service")
logger.info("人工审核退款阈值: ¥%s | Redis: %s:%s/%s",
            config.REFUND_THRESHOLD, config.REDIS_HOST, config.REDIS_PORT, config.REDIS_DB)

app = FastAPI(title="电商智能客服 Agent", version="0.4.0")


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    """统计每个 HTTP 请求的状态码与耗时，供监控看板使用。"""
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        metrics.record_request(request.url.path, 500, (time.perf_counter() - started) * 1000)
        metrics.record_error()
        raise
    metrics.record_request(
        request.url.path, response.status_code, (time.perf_counter() - started) * 1000
    )
    if response.status_code >= 500:
        metrics.record_error()
    return response


# ---------------- 请求/响应模型 ----------------

class AuthRequest(BaseModel):
    """注册与登录共用：用户名 + 密码。"""

    username: str = Field(..., min_length=2, max_length=32)
    password: str = Field(..., min_length=6, max_length=64)


class AuthResponse(BaseModel):
    token: str
    user_id: str
    username: str


class ChatRequest(BaseModel):
    """聊天请求：登录用户带 token；未登录可只带 session_id。"""

    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="", max_length=64)
    token: str = Field(default="", max_length=128)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    needs_human: bool = False
    handoff: dict | None = None


# ---------------- 页面与静态资源 ----------------

@app.get("/")
async def index():
    """首页：返回聊天网页。"""
    return FileResponse(config.BASE_DIR / "static" / "index.html")


app.mount(
    "/static",
    StaticFiles(directory=config.BASE_DIR / "static"),
    name="static",
)


@app.get("/human")
async def human_page():
    """人工客服工作台页面。"""
    return FileResponse(config.BASE_DIR / "static" / "human.html")


@app.get("/health")
async def health():
    """健康检查：附带 Redis 连通性。"""
    return {"status": "ok", "redis": store.ping()}


@app.get("/dashboard")
async def dashboard_page():
    """监控看板页面。"""
    return FileResponse(config.BASE_DIR / "static" / "dashboard.html")


def _mysql_ok() -> bool:
    """轻量探活：执行一次 SELECT 1。"""
    try:
        from sqlalchemy import text

        from app.db import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@app.get("/api/metrics")
async def api_metrics():
    """监控看板数据接口：运行指标 + 配置 + 依赖健康。"""
    return {
        "metrics": metrics.snapshot(),
        "deps": {
            "redis": store.ping(),
            "mysql": _mysql_ok(),
            "milvus_collection": config.MILVUS_COLLECTION,
        },
        "config": {
            "model": config.QWEN_CHAT_MODEL,
            "refund_threshold": config.REFUND_THRESHOLD,
            "rate_limit_chat_per_min": config.RATE_LIMIT_CHAT_PER_MIN,
            "rate_limit_auth_per_min": config.RATE_LIMIT_AUTH_PER_MIN,
            "login_max_failures": config.LOGIN_MAX_FAILURES,
        },
        "tickets": {
            "total": len(store.list_handoff_tickets(500)),
        },
    }


@app.get("/api/handoffs")
async def handoffs():
    """人工工作台接口：返回全部工单（最新在前）。"""
    return {"tickets": store.list_handoff_tickets()}


# ---------------- 限流与登录安全 ----------------

def _client_ip(request: Request) -> str:
    """取客户端 IP。

    注意：如果服务部署在 Nginx 等反向代理后面，需要改成读
    X-Forwarded-For 的第一段，并只信任可信代理写入的值。
    """
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(identity: str, scope: str, limit: int, window: int = 60) -> None:
    """固定窗口限流：超过限制抛 429，前端提示稍后再试。"""
    count = store.incr_window(f"{scope}:{identity}", window)
    if count > limit:
        metrics.record_rate_limit(scope)
        logger.warning("触发限流: scope=%s identity=%s count=%s", scope, identity, count)
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁，请稍后再试（{scope} 每分钟最多 {limit} 次）",
        )


# ---------------- 用户体系 ----------------

@app.post("/api/register", response_model=AuthResponse)
async def register(req: AuthRequest, request: Request):
    """注册并直接登录。用户名重复返回 409。"""
    _enforce_rate_limit(_client_ip(request), "auth", config.RATE_LIMIT_AUTH_PER_MIN)
    user = store.register_user(req.username, req.password)
    if user is None:
        raise HTTPException(status_code=409, detail="用户名已存在")
    token = store.create_token(user["user_id"])
    logger.info("新用户注册: %s (%s)", user["username"], user["user_id"])
    return AuthResponse(token=token, **user)


@app.post("/api/login", response_model=AuthResponse)
async def login(req: AuthRequest, request: Request):
    """登录成功后签发令牌（有效期由 TOKEN_TTL 控制）。"""
    _enforce_rate_limit(_client_ip(request), "auth", config.RATE_LIMIT_AUTH_PER_MIN)

    # 连续失败锁定：达到上限后，即使密码正确也要等锁定窗口过期
    failures = store.get_login_failures(req.username)
    if failures >= config.LOGIN_MAX_FAILURES:
        raise HTTPException(
            status_code=429,
            detail=f"登录失败次数过多，请 {config.LOGIN_LOCK_SECONDS // 60} 分钟后再试",
        )

    user = store.verify_login(req.username, req.password)
    if user is None:
        count = store.record_login_failure(req.username, config.LOGIN_LOCK_SECONDS)
        logger.warning("登录失败: %s（累计 %s 次）", req.username, count)
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    store.clear_login_failures(req.username)
    token = store.create_token(user["user_id"])
    logger.info("用户登录: %s (%s)", user["username"], user["user_id"])
    return AuthResponse(token=token, **user)


@app.post("/api/logout")
async def logout(token: str = ""):
    """退出登录：吊销令牌（聊天记录保留在 Redis）。"""
    store.revoke_token(token)
    return {"ok": True}


@app.get("/api/history")
async def history(token: str = "", limit: int = 50):
    """读取该用户的聊天记录，供页面刷新后回显。"""
    user_id = store.get_user_id_by_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    return {"user_id": user_id, "messages": store.get_messages(user_id, limit=limit)}


# ---------------- 聊天主流程 ----------------

def _to_chat_messages(history: list[dict[str, str]]):
    """把历史 dict 转成 LangChain 消息对象（只含 user/assistant）。"""
    messages = []
    for m in history:
        if m["role"] == "assistant":
            messages.append(AIMessage(content=m["content"]))
        else:
            messages.append(HumanMessage(content=m["content"]))
    return messages


def _read_token_usage(message: AIMessage) -> tuple:
    """兼容两种 token 用量返回格式（字典 / 对象）。"""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        return usage.get("input_tokens"), usage.get("output_tokens")
    if usage is not None:
        return getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)
    token_usage = getattr(message, "response_metadata", {}).get("token_usage", {})
    return token_usage.get("prompt_tokens"), token_usage.get("completion_tokens")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    # 1) 登录用户走 Redis；未登录回退到进程内存（评测脚本/匿名试用）
    user_id = store.get_user_id_by_token(req.token) if req.token else None
    if req.token and not user_id:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")

    # 限流：登录用户按用户 ID，匿名按 IP
    _enforce_rate_limit(
        user_id or f"ip:{_client_ip(request)}",
        "chat",
        config.RATE_LIMIT_CHAT_PER_MIN,
    )

    if user_id:
        session_id = user_id
        history = store.get_messages(user_id, limit=config.MEMORY_MAX_TURNS * 2)
        attempts = store.get_attempts(user_id)
    else:
        session_id = req.session_id.strip() or "anonymous"
        history = get_messages(session_id)
        attempts = get_attempts(session_id)

    initial_messages = _to_chat_messages(history)
    initial_messages.append(HumanMessage(content=user_message))

    logger.info("=== 收到消息 session=%s%s ===", session_id, "（登录用户）" if user_id else "")
    logger.info("用户: %s | 当前连续失败次数: %s", user_message, attempts)

    # 2) 跑 LangGraph 图：route -> 专家节点 -> END
    start = time.perf_counter()
    try:
        output = await agent_graph.ainvoke(
            {
                "messages": initial_messages,
                "trace": [],
                "session_id": session_id,
                "attempts": attempts,
            }
        )
    except Exception as exc:
        logger.exception("LangGraph 运行失败")
        raise HTTPException(status_code=502, detail=f"服务暂时不可用：{exc}")
    elapsed = time.perf_counter() - start

    final_message = output["messages"][-1]
    reply = final_message.content
    prompt_tokens, completion_tokens = _read_token_usage(final_message)
    metrics.record_llm_call(prompt_tokens, completion_tokens)

    logger.info(">>> 图节点轨迹: %s", " -> ".join(output.get("trace", [])))
    logger.info("总耗时: %.2fs | 输入tokens: %s | 输出tokens: %s",
                elapsed, prompt_tokens, completion_tokens)
    logger.info("助手: %s", reply)

    # 3) 写回会话历史（登录用户进 Redis，匿名进内存）
    if user_id:
        store.add_message(user_id, "user", user_message)
        store.add_message(user_id, "assistant", reply)
    else:
        add_message(session_id, "user", user_message)
        add_message(session_id, "assistant", reply)

    # 4) 更新连续失败计数
    intent = output.get("intent", "")
    is_clarify_turn = intent == "clarify" and not output.get("handoff_ticket_id")
    if user_id:
        if is_clarify_turn:
            attempts = store.increment_attempts(user_id)
            logger.info("连续失败次数更新为: %s", attempts)
        else:
            store.reset_attempts(user_id)
    else:
        if is_clarify_turn:
            attempts = increment_attempts(session_id)
            logger.info("连续失败次数更新为: %s", attempts)
        else:
            reset_attempts(session_id)

    # 5) 工单摘要回传前端
    needs_human = False
    handoff_info = None
    ticket_id = output.get("handoff_ticket_id")
    if ticket_id:
        ticket = store.get_handoff_ticket(ticket_id)
        if ticket:
            needs_human = True
            handoff_info = {
                "ticket_id": ticket["ticket_id"],
                "trigger": ticket["trigger"],
                "reason": ticket["reason"],
            }
            logger.info(">>> 本次已转人工: %s (%s)", ticket_id, ticket["trigger"])

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        needs_human=needs_human,
        handoff=handoff_info,
    )
