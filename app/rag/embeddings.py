"""轻量向量模型客户端：直接用 HTTP 调阿里云百炼原生 embedding 接口。

为什么不用 langchain_community 的 DashScopeEmbeddings：
本地开发环境装了它，但 Docker 镜像按 requirements.txt 安装时没有它，
容器一跑就暴露了“环境不一致”。这里用 httpx 直连原生接口，
少一个大依赖包，本机和 Docker 行为完全一致。
"""
import httpx
from langchain_core.embeddings import Embeddings


class DashScopeNativeEmbeddings(Embeddings):
    """兼容 LangChain Embeddings 接口的最小实现。"""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.url = (
            "https://dashscope.aliyuncs.com/api/v1/services/"
            "embeddings/text-embedding/text-embedding"
        )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        # 百炼批量接口单次上限 10 条，超出会返回 400 InvalidParameter，
        # 这里自动分批，最后把结果按原顺序拼回
        batch_size = 10
        results: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            results.extend(self._request(batch))
        return results

    def _request(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": {"texts": texts}}
        response = httpx.post(self.url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["output"]["embeddings"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_texts([text])[0]
