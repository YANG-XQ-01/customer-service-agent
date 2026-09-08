"""组装 LangGraph 流程图。

节点：route（路由）→ 各专家/闲聊/兜底
边：普通边（START→route、各业务节点→END）
    条件边（route 按 intent 决定走哪个分支）
"""
from langgraph.graph import END, START, StateGraph

from app.agents.experts import (
    after_sale_agent_node,
    chat_node,
    clarify_node,
    knowledge_agent_node,
    order_agent_node,
)
from app.agents.handoff import handoff_node
from app.agents.router import route_node
from app.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    # 1) 登记所有节点
    graph.add_node("route", route_node)
    graph.add_node("order_agent", order_agent_node)
    graph.add_node("after_sale_agent", after_sale_agent_node)
    graph.add_node("knowledge_agent", knowledge_agent_node)
    graph.add_node("chat_node", chat_node)
    graph.add_node("clarify_node", clarify_node)
    graph.add_node("handoff_node", handoff_node)

    # 2) 起始边：图一启动先进路由
    graph.add_edge(START, "route")

    # 3) 条件边：路由输出哪个 intent，就走哪个分支。
    #    clarify 是特殊分支：连续失败次数 >= 2 时改道转人工
    def _router_exit(state):
        intent = state["intent"]
        if intent == "clarify" and state.get("attempts", 0) >= 2:
            return "handoff"
        return intent

    graph.add_conditional_edges(
        "route",
        _router_exit,
        {
            "order": "order_agent",
            "after_sale": "after_sale_agent",
            "knowledge": "knowledge_agent",
            "chat": "chat_node",
            "clarify": "clarify_node",
            "handoff": "handoff_node",
        },
    )

    # 4) 售后专家可能因超阈值触发转人工：改道 handoff_node 收尾
    def _after_sale_exit(state):
        return "handoff" if state.get("handoff_requested") else "done"

    graph.add_conditional_edges(
        "after_sale_agent",
        _after_sale_exit,
        {"handoff": "handoff_node", "done": END},
    )

    # 5) 其余业务节点完成后结束
    for node in [
        "order_agent",
        "knowledge_agent",
        "chat_node",
        "clarify_node",
        "handoff_node",
    ]:
        graph.add_edge(node, END)

    return graph.compile()


# 全局编译好的图，服务启动时加载一次
agent_graph = build_graph()
