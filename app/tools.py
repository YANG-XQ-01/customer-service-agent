"""客服 Agent 的工具清单。

每个工具 = 一个 Python 函数 + @tool 装饰器。
工具的“名字 + 描述 + 参数说明”会被原样发给大模型，
模型就是靠这些信息决定“要不要用、什么时候用、传什么参数”。
所以描述必须具体、说人话——这是工具能被正确调用的关键。
"""
from langchain_core.tools import tool

from app.db import SessionLocal
from app.models import AfterSale, LogisticsEvent, Order, OrderItem


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
    result = _query_order_rows(order_no)
    if result is None:
        return f"未找到订单 {order_no}，请核实订单号后重试"
    order, items = result
    item_text = "、".join(
        f"{it.product_name} x{it.quantity}" for it in items
    )
    return (
        f"订单号：{order.order_no}\n"
        f"状态：{order.status}\n"
        f"下单时间：{order.created_at:%Y-%m-%d %H:%M}\n"
        f"商品：{item_text}\n"
        f"实付金额：¥{order.total_amount}\n"
        f"快递：{order.carrier or '未发货'} {order.tracking_no or ''}"
    )


@tool
def query_logistics_by_no(order_no: str) -> str:
    """查询物流轨迹：输入完整订单号，返回该订单每一段物流更新。"""
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

    if not events:
        return f"订单 {order_no} 暂无物流轨迹（可能尚未发货）"
    lines = [f"{e.event_at:%m-%d %H:%M} {e.description}" for e in events]
    return f"订单 {order_no} 物流轨迹：\n" + "\n".join(lines)


@tool
def query_after_sale_by_no(order_no: str) -> str:
    """查询售后单进度：输入完整订单号，返回退货/退款/换货申请的状态。"""
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
        return f"订单 {order_no} 没有售后申请记录"
    lines = []
    for r in records:
        amount = f"¥{r.refund_amount}" if r.refund_amount is not None else "待定"
        lines.append(
            f"- {r.service_type}申请：状态 {r.status}，"
            f"金额 {amount}，原因：{r.reason or '未填写'}"
        )
    return f"订单 {order_no} 的售后记录：\n" + "\n".join(lines)


@tool
def search_service_knowledge(query: str) -> str:
    """检索客服知识库：商品参数、退换货政策、退款时效、运费险、物流规则等，
    输入用户的问题原文，返回相关知识点。回答政策/商品问题前必须先调用本工具。"""
    from app.rag.retriever import search_top_k  # 延迟导入，避免循环依赖

    return search_top_k(query, k=3)


# Agent 会拿到这份清单，逐个“阅读”工具说明
TOOLS = [
    query_order_by_no,
    query_logistics_by_no,
    query_after_sale_by_no,
    search_service_knowledge,
]

