"""FastAPI 入口：电商客服 Agent（阶段 3：LangGraph 多智能体）。

处理链路：
网页发消息 -> 取出会话历史 -> 交给 LangGraph 图
-> 路由节点判断意图 -> 派给对应专家节点（内部可调工具）
-> 产出最终回答 -> 存回历史 -> 返回给网页
"""
import logging
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app import config
from app.graph import agent_graph
from app.memory import add_message, get_messages

# Windows 控制台默认编码可能是 GBK，先切成 UTF-8，保证日志中文可读
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("customer-service")

app = FastAPI(title="电商智能客服 Agent", version="0.3.0")


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


# 把 static 目录挂到 /static 路径
app.mount(
    "/static",
    StaticFiles(directory=config.BASE_DIR / "static"),
    name="static",
)


@app.get("/health")
async def health():
    """健康检查。"""
    return {"status": "ok"}


def _to_chat_messages(history: list[dict[str, str]]):
    """把内存里的历史 dict 转成 LangChain 消息对象。

    只转 user/assistant 两种角色；工具中间过程不存历史，
    模型需要时会重新调工具。
    """
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
async def chat(req: ChatRequest):
    session_id = req.session_id.strip()
    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    # 1) 取出历史并拼上新消息，作为图的初始状态
    history = get_messages(session_id)
    initial_messages = _to_chat_messages(history)
    initial_messages.append(HumanMessage(content=user_message))

    logger.info("=== 收到消息 session=%s ===", session_id)
    logger.info("用户: %s", user_message)

    # 2) 让整张图跑起来：route -> 专家节点 -> END
    start = time.perf_counter()
    try:
        output = await agent_graph.ainvoke(
            {
                "messages": initial_messages,
                "trace": [],
            }
        )
    except Exception as exc:
        logger.exception("LangGraph 运行失败")
        raise HTTPException(status_code=502, detail=f"服务暂时不可用：{exc}")
    elapsed = time.perf_counter() - start

    # 3) 取最终回答（图里专家节点追加的最后一条消息）
    final_message = output["messages"][-1]
    reply = final_message.content
    prompt_tokens, completion_tokens = _read_token_usage(final_message)

    logger.info(">>> 图节点轨迹: %s", " -> ".join(output.get("trace", [])))
    logger.info("总耗时: %.2fs | 输入tokens: %s | 输出tokens: %s",
                elapsed, prompt_tokens, completion_tokens)
    logger.info("助手: %s", reply)

    # 4) 只把“用户句 + 最终回答”存回历史
    add_message(session_id, "user", user_message)
    add_message(session_id, "assistant", reply)

    return ChatResponse(session_id=session_id, reply=reply)

