"""Qdrant 向量库封装 —— 语义检索层。

职责:建集合、upsert 向量、ANN top-k 检索。
Phase 1 用于"混合入口召回"的向量侧:把 query 向量化后检索最相似的实体/passage,
作为 BFS 的种子之一(与图侧软链接取并集)。

向量本身由 models/embedding.py 产出,本模块只管存与查,不耦合具体 embedding 后端。
集合维度在首次 ensure_collection 时按传入的 dim 确定。
"""

from config.connections import QDRANT


class VectorStore:
    def __init__(self, collection=None):
        from qdrant_client import QdrantClient
        # check_compatibility=False:client/server 小版本差异告警无碍功能,关掉
        self._client = QdrantClient(url=QDRANT.url, check_compatibility=False)
        self.collection = collection or QDRANT.collection

    def verify(self):
        """连通性自检:列出集合,连不上会抛异常。"""
        cols = self._client.get_collections().collections
        return f"Qdrant OK, {len(cols)} collections"

    def ensure_collection(self, dim, distance="Cosine"):
        """建集合(若不存在)。dim 必须与 embedder 输出维度一致。"""
        from qdrant_client.models import Distance, VectorParams
        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection in existing:
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=dim, distance=Distance[distance.upper()]
            ),
        )

    def upsert(self, ids, vectors, payloads):
        """写入向量。

        ids:每条的唯一 id(int/str);payloads:每条的元数据(如 {'name':..,'text':..})。
        """
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(id=i, vector=v, payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]
        self._client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector, top_k=5, type_filter=None):
        """ANN 检索,返回 [(payload, score), ...] 按相似度降序。

        type_filter:可选字符串(如 'entity' / 'event'),按 payload.type 过滤。
        双层 schema 用法:事实问只搜 entity 向量;情节/续写问只搜 event content 向量。
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        kwargs = dict(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        if type_filter:
            kwargs["query_filter"] = Filter(must=[FieldCondition(
                key="type", match=MatchValue(value=type_filter))])
        # qdrant-client 1.10+ 用 query_points 替代弃用的 search
        resp = self._client.query_points(**kwargs)
        return [(p.payload, p.score) for p in resp.points]
