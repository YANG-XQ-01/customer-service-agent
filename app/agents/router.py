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

    intent: Literal[
        "order", "after_sale", "knowledge", "chat", "clarify", "handoff"
    ]
    order_no: Optional[str] = Field(
        default=None, description="消息或最近对话中出现的 10~12 位完整订单号"
    )


# 结构化输出：让模型必须返回上面的 JSON，而不是自由发挥
_structured = get_chat_model().with_structured_output(IntentResult)


def _normalize_order_no(raw: str | None) -> str | None:
    r"""从模型输出里提取干净的 10~12 位订单号。

    用环视 (?<!\d) / (?!\d) 而不是 \\b：
    Python 正则里 \\w 包含汉字，\\b 会把“订单20260901001”这种
    贴着中文的订单号挡掉；环视只排除数字，中文场景也适用，
    同时避免从 13 位以上的超长数字串里误截前 10~12 位。
    """
    if not raw:
        return None
    match = re.search(r"(?<!\d)\d{10,12}(?!\d)", raw)
    return match.group(0) if match else None


async def route_node(state: AgentState) -> dict:
    """输入：状态里的对话消息；输出：intent + extracted（不产生回答）。"""
    start = time.perf_counter()
    # 只拿最近 8 条消息给路由看，够判断了，还能省 token
    recent = state["messages"][-8:]
    router_messages = [SystemMessage(content=ROUTER_SYSTEM)] + recent

    try:
        result = await _structured.ainvoke(router_messages)
        intent = result.intent.strip().lower()
        raw_order_no = result.order_no
    except Exception as exc:
        # 模型输出不合枚举/JSON 解析失败时，不让请求 502，
        # 降级成 clarify 走澄清/兜底节点（图里永远有路可走）
        logger.warning(">>> 路由结构化输出解析失败，降级为 clarify: %s", exc)
        intent = "clarify"
        raw_order_no = None
    order_no = _normalize_order_no(raw_order_no)

    logger.info(">>> 路由节点: intent=%s order_no=%s (%.2fs)",
                intent, order_no, time.perf_counter() - start)
    return {
        "intent": intent,
        # 合并而不是覆盖：保留之前抽到的槽位（如商品名），
        # 为“澄清后再路由”这类循环结构做准备；用 .get 防首轮 KeyError
        "extracted": {**(state.get("extracted") or {}), "order_no": order_no},
        "trace": [f"route -> {intent}"],
    }
