"""FastAPI 入口：接收消息、调用大模型、返回回答。

本阶段只有一条主链路（这也是后面所有阶段的地基）：
网页发消息 -> /api/chat 收到 -> 取出该会话历史并拼上新消息
-> 交给千问 -> 把回答存回历史 -> 返回给网页
"""
import logging
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config
from app.llm import create_chat_model
from app.memory import add_message, get_messages

# Windows 控制台默认编码可能是 GBK，先把标准输出/错误切成 UTF-8，
# 否则日志里的中文会变成乱码（演示时你要求日志必须可读）
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# 日志：每条请求都会打印，这就是你以后看“完整轨迹”的地方
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("customer-service")

app = FastAPI(title="电商智能客服 Agent", version="0.1.0")

# 模型对象只创建一次，避免每次请求都重新“握手”建连接
_chat_model = create_chat_model() if config.DASHSCOPE_API_KEY else None
if _chat_model is None:
    logger.warning("未检测到 DASHSCOPE_API_KEY，请检查项目根目录的 .env 文件")


class ChatRequest(BaseModel):
    """前端发来的请求体。session_id 用来区分不同用户/会话。"""

    session_id: str = Field(..., min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    """返回给前端的响应体。"""

    session_id: str
    reply: str


@app.get("/")
async def index():
    """首页：返回聊天网页。"""
    return FileResponse(config.BASE_DIR / "static" / "index.html")


# 把 static 目录挂到 /static 路径，网页里的 CSS/JS 才能被浏览器加载
app.mount(
    "/static",
    StaticFiles(directory=config.BASE_DIR / "static"),
    name="static",
)


@app.get("/health")
async def health():
    """健康检查：部署阶段会用到，现在顺手加上。"""
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id.strip()
    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    # 模型没配好就直接给 503，让网页/客户端看到明确原因
    if _chat_model is None:
        raise HTTPException(status_code=503, detail="服务未配置模型密钥，请检查 .env")

    # 1) 取出该会话的历史，再拼上用户新消息
    history = get_messages(session_id)
    messages_for_model = [
        {"role": m["role"], "content": m["content"]} for m in history
    ]
    messages_for_model.insert(0, {"role": "system",
                                  "content": "你是星辰数码商城的客服小星，回答简洁、礼貌，用中文，不确定的事情不要编。"})
    messages_for_model.append({"role": "user", "content": user_message})

    logger.info("=== 收到消息 session=%s ===", session_id)
    logger.info("用户: %s", user_message)

    # 2) 调用千问，并记录耗时
    start = time.perf_counter()
    try:
        response = await _chat_model.ainvoke(messages_for_model)
    except Exception as exc:  # 网络、密钥、限流都可能在这里抛错
        logger.exception("调用大模型失败")
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}")
    elapsed = time.perf_counter() - start

    reply = response.content
    # 读取 token 用量：不同版本/模型的返回格式不一样，写成兼容读取
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        # 实测千问兼容接口把用量放在字典里
        prompt_tokens = usage.get("input_tokens")
        completion_tokens = usage.get("output_tokens")
    elif usage is not None:
        # 部分 langchain 版本返回的是对象，用属性读取
        prompt_tokens = getattr(usage, "input_tokens", None)
        completion_tokens = getattr(usage, "output_tokens", None)
    else:
        # 兼容旧格式：token_usage 藏在 response_metadata 里
        token_usage = getattr(response, "response_metadata", {}).get("token_usage", {})
        prompt_tokens = token_usage.get("prompt_tokens")
        completion_tokens = token_usage.get("completion_tokens")

    logger.info(
        "耗时: %.2fs | 输入tokens: %s | 输出tokens: %s",
        elapsed,
        prompt_tokens,
        completion_tokens,
    )
    logger.info("助手: %s", reply)

    # 3) 把“用户句 + 助手句”一起存回历史，下一轮才能记住
    add_message(session_id, "user", user_message)
    add_message(session_id, "assistant", reply)

    return ChatResponse(session_id=session_id, reply=reply)
