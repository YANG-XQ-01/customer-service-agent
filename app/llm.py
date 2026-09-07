"""创建“千问”大模型客户端。

这里用的是 LangChain 的 ChatOpenAI，但把地址指向阿里云百炼的
OpenAI 兼容接口：代码里写的是 OpenAI 协议，背后真正跑的是通义千问。

好处（面试点）：以后想换 OpenAI、DeepSeek 等任何兼容模型，
只需要改 config.py 里的地址和模型名，调用代码一行都不用动。
"""
from langchain_openai import ChatOpenAI

from app import config


def _build_chat_model() -> ChatOpenAI:
    """返回一个可被 LangChain 调用的聊天模型对象。"""
    return ChatOpenAI(
        model=config.QWEN_CHAT_MODEL,
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.QWEN_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
    )


# 全局复用一个模型实例：多个节点/Agent 共享，避免重复握手
_model: ChatOpenAI | None = None


def get_chat_model() -> ChatOpenAI:
    """惰性创建共享模型（第一次调用时才真正建连接）。"""
    global _model
    if _model is None:
        _model = _build_chat_model()
    return _model


# 兼容旧引用：main.py 之前用 create_chat_model() 创建
create_chat_model = _build_chat_model

