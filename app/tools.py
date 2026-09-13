"""客服 Agent 的工具清单。

每个工具 = 一个 Python 函数 + @tool 装饰器。
工具的“名字 + 描述 + 参数说明”会被原样发给大模型，
模型就是靠这些信息决定“要不要用、什么时候用、传什么参数”。
所以描述必须具体、说人话——这是工具能被正确调用的关键。
"""
import logging
import time

from langchain_core.tools import tool

from app import config
from app.db import SessionLocal
from app import metrics
from app.models import AfterSale, LogisticsEvent, Order, OrderItem
from app.store import new_ticket_id

logger = logging.getLogger("customer-service")


def _log_tool_call(name: str, params: str, result: str, elapsed: float) -> None:
    """工具内统一打印调用日志：名称 / 参数 / 返回 / 耗时。

    为什么在工具里打而不是用回调：阶段 3 每个专家节点内部还有一层
    Agent 循环，回调容易漏挂；在工具函数里打日志最稳妥，永远不会漏。
    """
    logger.info(">>> 调用工具: %s", name)
    logger.info(">>> 工具参数: %s", params)
    logger.info(">>> 工具返回(%.2f秒): %s", elapsed, result)
    metrics.record_tool(name)


def _query_order_rows(order_no: str):
    """内部公共查询：订单 + 商品明细。查不到返回 None。"""
    db = SessionLocal()
    try:
        order = db.query(Order).filter(Order.order_no == order_no).first()
        if order is None:
            return None
        items = (
            db.query(OrderItem)
            .filter(OrderItem.order_id == order.id)
            .all()
        )
        return order, items
    finally:
        db.close()


@tool
def query_order_by_no(order_no: str) -> str:
    """查询订单信息：输入完整订单号，返回订单状态、下单时间、金额和商品清单。"""
    start = time.perf_counter()
    result = _query_order_rows(order_no)
    if result is None:
        text = f"未找到订单 {order_no}，请核实订单号后重试"
    else:
        order, items = result
        item_text = "、".join(
            f"{it.product_name} x{it.quantity}" for it in items
        )
        text = (
            f"订单号：{order.order_no}\n"
            f"状态：{order.status}\n"
            f"下单时间：{order.created_at:%Y-%m-%d %H:%M}\n"
            f"商品：{item_text}\n"
            f"实付金额：¥{order.total_amount}\n"
            f"快递：{order.carrier or '未发货'} {order.tracking_no or ''}"
        )
    _log_tool_call("query_order_by_no", f"{{'order_no': '{order_no}'}}",
                   text, time.perf_counter() - start)
    return text


@tool
def query_logistics_by_no(order_no: str) -> str:
    """查询物流轨迹：输入完整订单号，返回该订单每一段物流更新。"""
    start = time.perf_counter()
    db = SessionLocal()
    try:
        events = (
            db.query(LogisticsEvent)
            .filter(LogisticsEvent.order_no == order_no)
            .order_by(LogisticsEvent.event_at)
            .all()
        )
    finally:
        db.close()

    if _query_order_rows(order_no) is None:
        text = f"未找到订单 {order_no}，请核实订单号后重试"
    elif not events:
        text = f"订单 {order_no} 暂无物流轨迹（可能尚未发货）"
    else:
        lines = [f"{e.event_at:%m-%d %H:%M} {e.description}" for e in events]
        text = f"订单 {order_no} 物流轨迹：\n" + "\n".join(lines)
    _log_tool_call("query_logistics_by_no", f"{{'order_no': '{order_no}'}}",
                   text, time.perf_counter() - start)
    return text


@tool
def query_after_sale_by_no(order_no: str) -> str:
    """查询售后单进度：输入完整订单号，返回退货/退款/换货申请的状态。"""
    start = time.perf_counter()
    db = SessionLocal()
    try:
        records = (
            db.query(AfterSale)
            .filter(AfterSale.order_no == order_no)
            .order_by(AfterSale.created_at.desc())
            .all()
        )
    finally:
        db.close()

    if not records:
        text = f"订单 {order_no} 没有售后申请记录"
    else:
        lines = []
        for r in records:
            amount = f"¥{r.refund_amount}" if r.refund_amount is not None else "待定"
            lines.append(
                f"- {r.service_type}申请：状态 {r.status}，"
                f"金额 {amount}，原因：{r.reason or '未填写'}"
            )
        text = f"订单 {order_no} 的售后记录：\n" + "\n".join(lines)
    _log_tool_call("query_after_sale_by_no", f"{{'order_no': '{order_no}'}}",
                   text, time.perf_counter() - start)
    return text


@tool
def search_service_knowledge(query: str) -> str:
    """检索客服知识库：商品参数、退换货政策、退款时效、运费险、物流规则等，
    输入用户的问题原文，返回相关知识点。回答政策/商品问题前必须先调用本工具。"""
    from app.rag.retriever import search_top_k  # 延迟导入，避免循环依赖

    start = time.perf_counter()
    text = search_top_k(query, k=3)
    _log_tool_call("search_service_knowledge", f"{{'query': '{query}'}}",
                   text, time.perf_counter() - start)
    return text


@tool
def request_human_handoff(reason: str) -> str:
    """把对话转给人工客服：当退款/退货金额超过人工审核阈值，
    或出现需要人工审核的情况时调用。参数 reason 用一句话写清用户诉求。
    调用后停止自行处理，由转人工节点收尾。"""
    start = time.perf_counter()
    ticket_id = new_ticket_id()
    text = f"HANDOFF_REQUESTED|{ticket_id}|{reason}"
    _log_tool_call("request_human_handoff", f"{{'reason': '{reason}'}}",
                   text, time.perf_counter() - start)
    return text


# 把配置里的阈值写进工具描述（模型能看到的“说明书”），
# 改 .env 后重启服务即生效
request_human_handoff.description = (
    f"把对话转给人工客服：当用户要求的退款/退货金额超过 "
    f"¥{config.REFUND_THRESHOLD} 元，或出现需要人工审核的情况时调用。"
    "参数 reason 用一句话写清用户诉求。调用后停止自行处理，由转人工节点收尾。"
)

# 保留总清单（给需要全部工具的场合用；阶段 3 各专家只取子集）
TOOLS = [
    query_order_by_no,
    query_logistics_by_no,
    query_after_sale_by_no,
    search_service_knowledge,
    request_human_handoff,
]
