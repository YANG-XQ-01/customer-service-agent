"""转人工节点：三种触发场景汇合到这里，统一打包工单上下文。"""
import logging
import time

from langchain_core.messages import AIMessage, AnyMessage

from app.memory import create_handoff_ticket, new_ticket_id
from app.state import AgentState

logger = logging.getLogger("customer-service")


def _trigger_info(state: AgentState) -> tuple[str, str]:
    """根据进入方式判断触发场景，返回 (trigger, reason)。"""
    intent = state.get("intent", "")
    if intent == "handoff":
        return "user_request", "用户明确要求转人工客服"
    if state.get("handoff_requested"):
        return "refund_threshold", state.get("handoff_reason") or "退款/售后需人工审核"
    return "repeated_failures", "连续多次未能理解或解决用户问题"


def _message_to_dict(message: AnyMessage) -> dict:
    """把 LangChain 消息转成 {role, content}，方便人工侧阅读。"""
    role = "assistant" if isinstance(message, AIMessage) else "user"
    content = getattr(message, "content", "")
    # 少数情况 content 是列表（多模态），拼成文本展示
    if isinstance(content, list):
        content = " ".join(str(part) for part in content)
    return {"role": role, "content": str(content)}


async def handoff_node(state: AgentState) -> dict:
    """生成工单：把完整上下文（对话、意图、槽位、触发原因）打包给人工。"""
    start = time.perf_counter()
    trigger, reason = _trigger_info(state)
    ticket_id = state.get("handoff_ticket_id") or new_ticket_id()

    # 人工最需要看的：最近 10 轮对话
    history = [
        _message_to_dict(m) for m in state.get("messages", [])
    ][-10:]

    create_handoff_ticket(
        ticket_id=ticket_id,
        session_id=state.get("session_id", ""),
        trigger=trigger,
        reason=reason,
        intent=state.get("intent", ""),
        extracted=state.get("extracted", {}),
        messages=history,
    )
    elapsed = time.perf_counter() - start
    logger.info(">>> 转人工工单 %s | 触发=%s | 原因=%s | 对话%s条 (%.2fs)",
                ticket_id, trigger, reason, len(history), elapsed)

    answer = (
        f"好的，已为您转接人工客服。\n"
        f"工单号：{ticket_id}\n"
        f"转接原因：{reason}\n"
        "人工客服会带着我们的对话记录继续为您处理，请稍候。"
    )
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "handoff_ticket_id": ticket_id,
        "trace": [f"handoff -> {ticket_id}"],
    }

