"""MySQL 表结构（ORM 模型）。

每个类 = 一张表，每个类属性 = 一列。
结构化的“事实数据”（订单、售后单）必须放这里，用 SQL 精确查，
而不是塞进向量库让模型猜——这是防幻觉的第一道防线。
"""
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """所有模型的公共基类。"""


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(20))


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    order_no: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(20))  # 待付款/待发货/运输中/已签收/已关闭
    total_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    carrier: Mapped[str | None] = mapped_column(String(20))  # 快递公司
    tracking_no: Mapped[str | None] = mapped_column(String(64))  # 快递单号
    created_at: Mapped[datetime] = mapped_column(DateTime)

    user: Mapped["User"] = relationship()
    items: Mapped[list["OrderItem"]] = relationship(back_populates="order")


class OrderItem(Base):
    """订单里的每一件商品（一张订单可以买多件）。"""

    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), nullable=False)
    product_name: Mapped[str] = mapped_column(String(100))
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2))

    order: Mapped["Order"] = relationship(back_populates="items")


class LogisticsEvent(Base):
    """物流轨迹：一个订单对应多条时间节点。"""

    __tablename__ = "logistics_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_no: Mapped[str] = mapped_column(String(32), index=True)
    event_at: Mapped[datetime] = mapped_column(DateTime)
    description: Mapped[str] = mapped_column(String(200))


class AfterSale(Base):
    """售后单：退货 / 退款 / 换货。"""

    __tablename__ = "after_sales"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_no: Mapped[str] = mapped_column(String(32), index=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    service_type: Mapped[str] = mapped_column(String(20))  # 退货/退款/换货
    status: Mapped[str] = mapped_column(String(20))  # 处理中/已同意/已拒绝/已完成
    refund_amount: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime)

