"""数据库引擎与会话管理（SQLAlchemy 2.0）。

engine：连接池，整个进程共享一个；
SessionLocal：每次请求用它开一个“会话”，用完必须关闭（见 get_db）。
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import config

# pool_pre_ping=True：拿连接前先探活，MySQL 断过重连也不会报错
engine = create_engine(
    config.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    """FastAPI 依赖：请求结束时自动归还/关闭数据库会话。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

