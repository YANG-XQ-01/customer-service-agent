"""四个业务节点：订单专家 / 售后专家 / 知识专家 / 闲聊 / 澄清兜底。

订单、售后、知识三个节点内部是“受限工具集的 Agent”（create_agent），
每个只拿到本职的 1~3 个工具，选错概率远小于阶段 2 的“全能 Agent”。
闲聊和兜底不需要工具，直接调模型即可——不是每个节点都要是 Agent。
"""
import logging
import time

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, SystemMessage

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
    [query_order_by_no, query_after_sale_by_no, search_service_knowledge],
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


async def after_sale_agent_node(state: AgentState) -> dict:
    return await _run_agent_node(AFTER_SALE_AGENT, state, "售后专家")


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

