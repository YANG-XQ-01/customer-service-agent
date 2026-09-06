"""在线检索器：把用户问题向量化，去 Milvus 找最相关的知识片段。"""
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_milvus import Milvus

from app import config

# 用 DashScope 原生向量接口（与旧项目实测写法一致）。
# 注意：不要用 OpenAI 兼容通道调 embedding，参数格式不兼容。
_embeddings = DashScopeEmbeddings(
    model=config.QWEN_EMBEDDING_MODEL,
    dashscope_api_key=config.DASHSCOPE_API_KEY,
)

_store: Milvus | None = None


def _get_store() -> Milvus:
    """惰性创建向量库客户端：第一次检索时才连接 Milvus。"""
    global _store
    if _store is None:
        _store = Milvus(
            embedding_function=_embeddings,
            collection_name=config.MILVUS_COLLECTION,
            connection_args={
                "uri": f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}",
            },
            enable_dynamic_field=True,
        )
    return _store


def search_top_k(query: str, k: int = 3) -> str:
    """检索并格式化成给大模型的文本。

    每条都带上“来源”文件名——回答可以引用，也方便查证。
    """
    try:
        docs = _get_store().similarity_search(query, k=k)
    except Exception as exc:
        return f"知识库暂时不可用：{exc}"

    if not docs:
        return "知识库未检索到相关内容"

    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[片段{i}，来源:{source}]\n{doc.page_content}")
    return "\n\n".join(parts)
