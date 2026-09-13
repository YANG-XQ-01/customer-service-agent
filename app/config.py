"""读取 .env 配置文件，集中管理所有配置项。

为什么单独建这个文件：
1. 密钥、模型名、地址这类东西如果散落在代码里，改一处就要全文搜索；
2. 集中到一个模块后，其它文件只要写 `from app.config import ...` 就能拿到，
   以后换模型、换端口，都只改 .env，不动业务代码。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录 = 本文件（app/config.py）的上一级
BASE_DIR = Path(__file__).resolve().parent.parent

# 把 .env 里的键值对加载进环境变量（已存在的环境变量优先，不会被覆盖）
load_dotenv(BASE_DIR / ".env")

# ---- 千问大模型 ----
# 你的密钥，只在 .env 里，.env 已被 .gitignore 排除，绝不会进 git
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen-plus")
# 阿里云百炼的 OpenAI 兼容接口：让 LangChain 用 OpenAI 协议去调千问
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 随机性：0 最稳定，1 最发散；客服场景后面会调低，先留成可配置
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# ---- 会话记忆 ----
# 每个会话最多记住多少轮（1 轮 = 用户 1 句 + 助手 1 句）
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "10"))

# ---- 售后规则 ----
# 退款金额超过该阈值必须转人工审核（元）
REFUND_THRESHOLD = int(os.getenv("REFUND_THRESHOLD", "500"))

# ---- Redis（用户体系 / 会话与聊天记录缓存）----
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
# 聊天记录与登录令牌的过期时间（秒），默认 7 天
CHAT_HISTORY_TTL = int(os.getenv("CHAT_HISTORY_TTL", str(7 * 24 * 3600)))
TOKEN_TTL = int(os.getenv("TOKEN_TTL", str(7 * 24 * 3600)))
# 工单保留时间（秒），默认 30 天
TICKET_TTL = int(os.getenv("TICKET_TTL", str(30 * 24 * 3600)))

# ---- 限流与登录安全 ----
# 每个用户/IP 每分钟允许的聊天请求数
RATE_LIMIT_CHAT_PER_MIN = int(os.getenv("RATE_LIMIT_CHAT_PER_MIN", "20"))
# 每个 IP 每分钟允许的注册/登录请求数
RATE_LIMIT_AUTH_PER_MIN = int(os.getenv("RATE_LIMIT_AUTH_PER_MIN", "10"))
# 连续登录失败达到该次数后临时锁定
LOGIN_MAX_FAILURES = int(os.getenv("LOGIN_MAX_FAILURES", "5"))
LOGIN_LOCK_SECONDS = int(os.getenv("LOGIN_LOCK_SECONDS", "300"))

# ---- MySQL（订单 / 售后数据）----
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "customer_service")

# SQLAlchemy 连接串。charset=utf8mb4 是中文不乱码的关键
DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
)

# ---- Milvus 知识向量库 ----
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
# 本项目专用的集合名，和旧项目的 ops_docs 互不干扰
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "customer_service_kb")
# 文本向量化模型（阿里云百炼）
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")

# ---- 服务与日志 ----
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
