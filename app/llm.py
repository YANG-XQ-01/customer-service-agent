"""创建“千问”大模型客户端。

这里用的是 LangChain 的 ChatOpenAI，但把地址指向阿里云百炼的
OpenAI 兼容接口：代码里写的是 OpenAI 协议，背后真正跑的是通义千问。

好处（面试点）：以后想换 OpenAI、DeepSeek 等任何兼容模型，
只需要改 config.py 里的地址和模型名，调用代码一行都不用动。
"""
from langchain_openai import ChatOpenAI

from app import config


def create_chat_model() -> ChatOpenAI:
    """返回一个可被 LangChain 调用的聊天模型对象。"""
    return ChatOpenAI(
        model=config.QWEN_CHAT_MODEL,
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.QWEN_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
    )

