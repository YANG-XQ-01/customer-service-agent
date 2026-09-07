"""路由节点：判断用户意图 + 抽取订单号。"""
import logging
import re
import time
from typing import Literal, Optional

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from app import config
from app.agents.prompts import ROUTER_SYSTEM
from app.llm import get_chat_model
from app.state import AgentState

logger = logging.getLogger("customer-service")


class IntentResult(BaseModel):
    """路由模型必须输出的结构化结果（枚举约束 + 订单号）。"""

    intent: Literal["order", "after_sale", "knowledge", "chat", "clarify"]
    order_no: Optional[str] = Field(
        default=None, description="消息或最近对话中出现的 10~12 位完整订单号"
    )


# 结构化输出：让模型必须返回上面的 JSON，而不是自由发挥
_structured = get_chat_model().with_structured_output(IntentResult)


def _normalize_order_no(raw: str | None) -> str | None:
    """只保留干净的 10~12 位数字，过滤掉模型可能带出的杂字符。"""
    if not raw:
        return None
    match = re.search(r"\d{10,12}", raw)
    return match.group(0) if match else None


async def route_node(state: AgentState) -> dict:
    """输入：状态里的对话消息；输出：intent + extracted（不产生回答）。"""
    start = time.perf_counter()
    # 只拿最近 8 条消息给路由看，够判断了，还能省 token
    recent = state["messages"][-8:]
    router_messages = [SystemMessage(content=ROUTER_SYSTEM)] + recent

    result = await _structured.ainvoke(router_messages)
    intent = result.intent.strip().lower()
    order_no = _normalize_order_no(result.order_no)

    logger.info(">>> 路由节点: intent=%s order_no=%s (%.2fs)",
                intent, order_no, time.perf_counter() - start)
    return {
        "intent": intent,
        "extracted": {"order_no": order_no},
        "trace": [f"route -> {intent}"],
    }

