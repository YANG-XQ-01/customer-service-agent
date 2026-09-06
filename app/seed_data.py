"""初始化演示数据：建库建表 + 灌入假用户/订单/售后单。

运行方式（在项目根目录）：
    python -m app.seed_data

注意：这是开发演示专用脚本，每次运行会先清空本项目 4 张表再重建，
只影响 customer_service 库，不会碰任何其它数据库。
"""
from datetime import datetime

import pymysql
from sqlalchemy import create_engine

from app import config
from app.db import SessionLocal
from app.models import AfterSale, Base, LogisticsEvent, Order, OrderItem, User


def ensure_database() -> None:
    """如果 customer_service 库还不存在，先用 pymysql 直连建库。"""
    conn = pymysql.connect(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{config.MYSQL_DB}` "
                "DEFAULT CHARACTER SET utf8mb4"
            )
        conn.commit()
    finally:
        conn.close()


def seed() -> None:
    ensure_database()

    # 开发演示脚本：清空重建，保证每次跑出来的数据一致
    engine = create_engine(config.DATABASE_URL)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    db = SessionLocal()
    try:
        # ---- 用户 ----
        xiaoming = User(name="小明", phone="13800000001")
        xiaohong = User(name="小红", phone="13800000002")
        aqiang = User(name="阿强", phone="13800000003")
        db.add_all([xiaoming, xiaohong, aqiang])
        db.flush()  # 先拿到自增 id

        # ---- 小明：运输中的蓝牙耳机订单（20260901001，测试样本第 1 条用）----
        o1 = Order(
            user_id=xiaoming.id,
            order_no="20260901001",
            status="运输中",
            total_amount=499,
            carrier="顺丰速运",
            tracking_no="SF1234567890",
            created_at=datetime(2026, 8, 30, 10, 15),
        )
        o1.items = [
            OrderItem(product_name="星音蓝牙耳机 X9", quantity=1, price=499),
        ]

        # ---- 小明：上个月已签收的机械键盘订单（对应售后单，测试样本第 3/11 条）----
        o2 = Order(
            user_id=xiaoming.id,
            order_no="20260901002",
            status="已签收",
            total_amount=399,
            carrier="中通快递",
            tracking_no="ZT88886666",
            created_at=datetime(2026, 7, 20, 14, 30),
        )
        o2.items = [
            OrderItem(product_name="星辰机械键盘 K87", quantity=1, price=399),
        ]

        # ---- 小明：待发货订单 ----
        o3 = Order(
            user_id=xiaoming.id,
            order_no="20260901003",
            status="待发货",
            total_amount=59,
            carrier=None,
            tracking_no=None,
            created_at=datetime(2026, 9, 3, 9, 0),
        )
        o3.items = [
            OrderItem(product_name="桌面理线器套装", quantity=1, price=59),
        ]

        # ---- 小红：运输中的冰箱订单 ----
        o4 = Order(
            user_id=xiaohong.id,
            order_no="20260901004",
            status="运输中",
            total_amount=2999,
            carrier="京东物流",
            tracking_no="JD2026090001",
            created_at=datetime(2026, 9, 1, 20, 0),
        )
        o4.items = [
            OrderItem(product_name="星辰智能冰箱 B500", quantity=1, price=2999),
        ]

        db.add_all([o1, o2, o3, o4])
        db.flush()

        # ---- 物流轨迹 ----
        db.add_all(
            [
                LogisticsEvent(
                    order_no="20260901001",
                    event_at=datetime(2026, 8, 30, 18, 0),
                    description="商家已发货，顺丰已揽收",
                ),
                LogisticsEvent(
                    order_no="20260901001",
                    event_at=datetime(2026, 8, 31, 8, 20),
                    description="快件到达杭州转运中心",
                ),
                LogisticsEvent(
                    order_no="20260901001",
                    event_at=datetime(2026, 9, 1, 15, 45),
                    description="快件到达上海转运中心，运输中",
                ),
                LogisticsEvent(
                    order_no="20260901004",
                    event_at=datetime(2026, 9, 2, 9, 10),
                    description="商家已发货，京东物流已揽收",
                ),
            ]
        )

        # ---- 售后单：键盘退货申请（处理中）----
        db.add(
            AfterSale(
                order_no="20260901002",
                user_id=xiaoming.id,
                service_type="退货",
                status="处理中",
                refund_amount=399,
                reason="键盘左键连击，疑似质量问题",
                created_at=datetime(2026, 9, 2, 11, 0),
            )
        )
        db.commit()
    finally:
        db.close()

    print(
        "种子数据完成：3 个用户、4 个订单、4 条物流轨迹、1 张售后单"
    )


if __name__ == "__main__":
    seed()

