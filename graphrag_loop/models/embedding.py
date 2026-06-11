"""Embedding 接口 —— 文本向量化,本地 bge-m3(默认)或 DashScope API(可切换)。

由 .env 的 EMBEDDING_BACKEND 决定后端:
  local     → sentence-transformers 跑 bge-m3,零成本离线
  dashscope → 调通义 text-embedding-v3,不下模型但走网络计费
两种后端对外接口一致:encode(texts) -> List[向量]。
"""

from config.connections import EMBEDDING


class _LocalEmbedder:
    """本地 bge-m3(sentence-transformers)。首次使用会下载模型(约 2GB)。"""

    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        # 新版方法名 get_embedding_dimension;老版本回退
        if hasattr(self._model, "get_embedding_dimension"):
            self.dim = self._model.get_embedding_dimension()
        else:
            self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts, normalize=True):
        single = isinstance(texts, str)
        vecs = self._model.encode(
            [texts] if single else list(texts),
            normalize_embeddings=normalize,
        )
        return vecs[0].tolist() if single else [v.tolist() for v in vecs]


class _DashScopeEmbedder:
    """通义 text-embedding-v3。"""

    def __init__(self, model_name):
        from config.connections import DASHSCOPE
        import dashscope
        dashscope.api_key = DASHSCOPE.api_key
        self._dashscope = dashscope
        self._model = model_name
        self.dim = 1024  # text-embedding-v3 默认维度

    def encode(self, texts, normalize=True):
        single = isinstance(texts, str)
        inp = [texts] if single else list(texts)
        resp = self._dashscope.TextEmbedding.call(model=self._model, input=inp)
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope embedding 失败: {resp.message}")
        vecs = [item["embedding"] for item in resp.output["embeddings"]]
        return vecs[0] if single else vecs


def get_embedder():
    """工厂:按 .env 的 EMBEDDING_BACKEND 返回对应后端。"""
    if EMBEDDING.backend == "dashscope":
        return _DashScopeEmbedder(EMBEDDING.model_dashscope)
    return _LocalEmbedder(EMBEDDING.model_local)
