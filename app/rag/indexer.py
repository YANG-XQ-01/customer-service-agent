"""把 data/knowledge/ 下的客服知识文档向量化，存入 Milvus。

运行方式（在项目根目录）：
    python -m app.rag.indexer

幂等：每次运行会先删除本项目的集合再重建，可重复执行，不会产生重复向量。
"""
from pathlib import Path

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config

KNOWLEDGE_DIR = config.BASE_DIR / "data" / "knowledge"


def load_documents() -> list[Document]:
    """读取所有 .md 知识文档，每篇带上文件名作为来源。"""
    docs = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(
                Document(page_content=text, metadata={"source": path.name})
            )
    return docs


def build() -> None:
    documents = load_documents()
    if not documents:
        raise SystemExit(f"没有找到知识文档：{KNOWLEDGE_DIR}")

    # 1) 切分：一篇长文档切成若干小段，检索粒度更准
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=40,
    )
    chunks = splitter.split_documents(documents)

    # 2) 向量化并入库（embedding 需要联网调用 text-embedding-v3）
    embeddings = DashScopeEmbeddings(
        model=config.QWEN_EMBEDDING_MODEL,
        dashscope_api_key=config.DASHSCOPE_API_KEY,
    )
    # drop_old=True：每次重建本集合，脚本可重复执行，不会产生重复数据
    Milvus.from_documents(
        chunks,
        embedding=embeddings,
        collection_name=config.MILVUS_COLLECTION,
        connection_args={
            "uri": f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}",
        },
        drop_old=True,
        enable_dynamic_field=True,
    )
    print(
        f"知识入库完成：{len(documents)} 篇文档 -> {len(chunks)} 个片段 "
        f"-> 集合 {config.MILVUS_COLLECTION}"
    )


if __name__ == "__main__":
    build()
