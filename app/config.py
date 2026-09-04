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

# ---- 服务与日志 ----
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

