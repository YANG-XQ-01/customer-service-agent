"""FastAPI 入口：电商客服 Agent（阶段 2：工具调用 + RAG）。

处理链路：
网页发消息 -> 取出会话历史 -> 交给“会调工具的 Agent”
-> Agent 判断要不要调工具（查订单/物流/售后/知识库）
-> 拿结果组织回答 -> 存回历史 -> 返回给网页
"""
import logging
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app import config
from app.llm import create_chat_model
from app.memory import add_message, get_messages
from app.tools import TOOLS

# Windows 控制台默认编码可能是 GBK，先切成 UTF-8，保证日志中文可读
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("customer-service")

app = FastAPI(title="电商智能客服 Agent", version="0.2.0")

# 模型对象只创建一次
_chat_model = create_chat_model() if config.DASHSCOPE_API_KEY else None

# 客服人设 + 使用规则。注意：规则写在这里，而不是祈祷模型自觉。
SYSTEM_PROMPT = """你是星辰数码商城的客服“小星”。你的任务是用简洁、礼貌的中文帮用户解决订单、物流、售后和商品咨询问题。

必须遵守的规则：
1. 查订单/物流/售后必须先调用对应工具，绝不能凭记忆编造订单信息。
2. 用户没给订单号时，先向用户索要完整订单号，绝不自己编一个。
3. 工具返回“未找到”时，如实告诉用户查不到，请其核对订单号。
4. 商品参数、售后政策、物流规则类问题，先调用知识库检索工具，基于检索结果回答。
5. 回答要引用知识来源（如“根据退货政策”），不确定就说不确定，不编造。
6. 需要用户确认的事项（退款金额、处理方案）要列清楚，一次说清。"""


class ToolCallLogger(BaseCallbackHandler):
    """回调日志器：每次工具被调用时自动打印完整轨迹。

    LangChain 的工具调用会触发 on_tool_start / on_tool_end 事件，
    我们在这里记录：工具名、传入参数、返回结果、耗时。
    这就是“完整工具调用日志”的实现。
    """

    def __init__(self) -> None:
        self._starts: dict[str, float] = {}

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs) -> None:
        self._starts[str(run_id)] = time.perf_counter()
        logger.info(">>> 调用工具: %s", serialized.get("name"))
        logger.info(">>> 工具参数: %s", input_str)

    def on_tool_end(self, output, *, run_id, **kwargs) -> None:
        elapsed = time.perf_counter() - self._starts.pop(str(run_id), time.perf_counter())
        logger.info(">>> 工具返回(%.2f秒): %s", elapsed, output)

    def on_tool_error(self, error, *, run_id, **kwargs) -> None:
        logger.error(">>> 工具出错: %s", error)


# 组装“会调工具的 Agent”：高层 Agent，模型自主决定调哪个工具。
# 只有配好了密钥才创建，否则接口返回 503 并提示检查 .env。
_agent = None
if _chat_model is not None:
    _agent = create_agent(
        model=_chat_model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
else:
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
    避免把大量工具噪音塞进下一轮，模型需要时会重新调工具。
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
    if _agent is None:
        raise HTTPException(status_code=503, detail="服务未配置模型密钥，请检查 .env")

    # 1) 取出历史并拼上新消息（历史里只有 user/assistant 对话）
    history = get_messages(session_id)
    messages_for_agent = _to_chat_messages(history)
    messages_for_agent.append(HumanMessage(content=user_message))

    logger.info("=== 收到消息 session=%s ===", session_id)
    logger.info("用户: %s", user_message)

    # 2) 交给 Agent 处理。回调里会打印每一步工具调用
    start = time.perf_counter()
    try:
        output = await _agent.ainvoke(
            {"messages": messages_for_agent},
            config={"callbacks": [ToolCallLogger()]},
        )
    except Exception as exc:
        logger.exception("Agent 调用失败")
        raise HTTPException(status_code=502, detail=f"Agent 调用失败：{exc}")
    elapsed = time.perf_counter() - start

    final_message = output["messages"][-1]
    reply = final_message.content
    prompt_tokens, completion_tokens = _read_token_usage(final_message)

    logger.info("总耗时: %.2fs | 输入tokens: %s | 输出tokens: %s",
                elapsed, prompt_tokens, completion_tokens)
    logger.info("助手: %s", reply)

    # 3) 只把“用户句 + 最终回答”存回历史
    add_message(session_id, "user", user_message)
    add_message(session_id, "assistant", reply)

    return ChatResponse(session_id=session_id, reply=reply)

