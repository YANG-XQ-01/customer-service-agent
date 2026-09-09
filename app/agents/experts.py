"""四个业务节点：订单专家 / 售后专家 / 知识专家 / 闲聊 / 澄清兜底。

订单、售后、知识三个节点内部是“受限工具集的 Agent”（create_agent），
每个只拿到本职的 1~3 个工具，选错概率远小于阶段 2 的“全能 Agent”。
闲聊和兜底不需要工具，直接调模型即可——不是每个节点都要是 Agent。
"""
import logging
import re
import time

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from app import config
from app.agents.prompts import (
    AFTER_SALE_SYSTEM,
    CHAT_SYSTEM,
    CLARIFY_SYSTEM,
    KNOWLEDGE_SYSTEM,
    ORDER_SYSTEM,
)
from app.llm import get_chat_model
from app.state import AgentState
from app.tools import (
    query_after_sale_by_no,
    query_logistics_by_no,
    query_order_by_no,
    request_human_handoff,
    search_service_knowledge,
)

logger = logging.getLogger("customer-service")


def _build_agent(tools: list, system_prompt: str):
    """每个专家共用一个模型实例，但工具集和提示词不同。"""
    return create_agent(
        model=get_chat_model(),
        tools=tools,
        system_prompt=system_prompt,
    )


# 模块加载时一次性建好，避免每次请求重新组装
ORDER_AGENT = _build_agent(
    [query_order_by_no, query_logistics_by_no], ORDER_SYSTEM
)
AFTER_SALE_AGENT = _build_agent(
    [
        query_order_by_no,
        query_after_sale_by_no,
        search_service_knowledge,
        request_human_handoff,
    ],
    AFTER_SALE_SYSTEM,
)
KNOWLEDGE_AGENT = _build_agent(
    [search_service_knowledge], KNOWLEDGE_SYSTEM
)


async def _run_agent_node(agent, state: AgentState, node_name: str) -> dict:
    """公共封装：跑 Agent、记日志、产出 answer 和追加的最终消息。"""
    start = time.perf_counter()
    logger.info(">>> 进入节点: %s", node_name)
    result = await agent.ainvoke(
        {"messages": state["messages"]},
        config={"recursion_limit": 15},
    )
    final_message = result["messages"][-1]
    elapsed = time.perf_counter() - start
    logger.info(">>> 退出节点: %s (%.2fs)", node_name, elapsed)
    return {
        "answer": final_message.content,
        "messages": [final_message],
        "trace": [f"{node_name} {elapsed:.2f}s"],
    }


async def order_agent_node(state: AgentState) -> dict:
    return await _run_agent_node(ORDER_AGENT, state, "订单专家")


def _extract_handoff_marker(messages) -> tuple | None:
    """在 Agent 的消息里找 request_human_handoff 工具留下的标记。

    标记格式：HANDOFF_REQUESTED|工单号|原因
    找到了就返回 (工单号, 原因)，由转人工节点统一收尾、打包上下文。
    """
    prefix = "HANDOFF_REQUESTED|"
    for message in messages:
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.startswith(prefix):
            parts = content.split("|", 2)
            if len(parts) >= 3:
                return parts[1], parts[2]
    return None


def _find_over_threshold_order(messages) -> tuple | None:
    """代码层兜底：检查售后 Agent 是否查到了超阈值订单。

    背景：模型有时查到 ¥2999 却“只说不调工具”。提示词拦不住时，
    这里直接看 query_order_by_no 的工具返回，金额超阈值就强制转人工。
    返回 (order_no, amount) 或 None。
    """
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "name", "") != "query_order_by_no":
            continue
        content = str(message.content)
        amount_match = re.search(r"实付金额：¥([\d.]+)", content)
        order_match = re.search(r"订单号：([A-Za-z0-9]+)", content)
        if amount_match and float(amount_match.group(1)) > config.REFUND_THRESHOLD:
            order_no = order_match.group(1) if order_match else "未知"
            return order_no, amount_match.group(1)
    return None


async def after_sale_agent_node(state: AgentState) -> dict:
    """售后专家：比其它专家多一道“转人工”检查。

    如果 Agent 因为退款金额超阈值调了 request_human_handoff，
    就不把它的普通回答当最终答案，而是打上转人工标记，
    让条件边把流程改道到 handoff_node 收尾。
    """
    start = time.perf_counter()
    logger.info(">>> 进入节点: 售后专家")
    result = await AFTER_SALE_AGENT.ainvoke(
        {"messages": state["messages"]},
        config={"recursion_limit": 15},
    )
    elapsed = time.perf_counter() - start

    marker = _extract_handoff_marker(result["messages"])
    if marker is not None:
        ticket_id, reason = marker
        logger.info(">>> 售后专家触发转人工: ticket=%s reason=%s (%.2fs)",
                    ticket_id, reason, elapsed)
        return {
            "handoff_requested": True,
            "handoff_ticket_id": ticket_id,
            "handoff_reason": reason,
            "messages": [],  # 不追加普通回答，最终话术由转人工节点生成
            "trace": [f"售后专家 {elapsed:.2f}s (触发转人工)"],
        }

    # 代码层兜底：模型没自觉调转人工工具，但订单金额确实超阈值
    over_threshold = _find_over_threshold_order(result["messages"])
    if over_threshold is not None:
        order_no, amount = over_threshold
        reason = (
            f"订单{order_no}实付金额¥{amount}超过人工审核阈值"
            f"¥{config.REFUND_THRESHOLD}，需人工审核"
        )
        logger.info(">>> 售后专家代码层强制转人工: %s (%.2fs)", reason, elapsed)
        return {
            "handoff_requested": True,
            "handoff_reason": reason,
            "messages": [],
            "trace": [f"售后专家 {elapsed:.2f}s (金额超阈值，代码强制转人工)"],
        }

    final_message = result["messages"][-1]
    logger.info(">>> 退出节点: 售后专家 (%.2fs)", elapsed)
    return {
        "answer": final_message.content,
        "messages": [final_message],
        "trace": [f"售后专家 {elapsed:.2f}s"],
    }


async def knowledge_agent_node(state: AgentState) -> dict:
    return await _run_agent_node(KNOWLEDGE_AGENT, state, "知识专家")


async def _run_plain_node(state: AgentState, system_prompt: str, node_name: str) -> dict:
    """闲聊/兜底：无工具，直接让模型回复。"""
    start = time.perf_counter()
    logger.info(">>> 进入节点: %s", node_name)
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = await get_chat_model().ainvoke(messages)
    elapsed = time.perf_counter() - start
    logger.info(">>> 退出节点: %s (%.2fs)", node_name, elapsed)
    return {
        "answer": response.content,
        "messages": [response],
        "trace": [f"{node_name} {elapsed:.2f}s"],
    }


async def chat_node(state: AgentState) -> dict:
    return await _run_plain_node(state, CHAT_SYSTEM, "闲聊节点")


async def clarify_node(state: AgentState) -> dict:
    return await _run_plain_node(state, CLARIFY_SYSTEM, "澄清/兜底节点")
