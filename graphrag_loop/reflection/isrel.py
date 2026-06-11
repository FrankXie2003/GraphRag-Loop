"""S3 IsRel —— 用 Cross-Encoder(bge-reranker)给候选打相关性分,< τ_rel 剪掉。

替换 demo 里的词重叠 ranker(retrieval/ranker.py)。对外接口一致:
    score_batch(query, texts) -> List[float in 0..1]
所以 demo 的 beam.score_candidates 可以无改动地用它。

Cross-Encoder 同时看 (query, text) 两段做交互打分,比 bi-encoder 的余弦更准,
正是重排该用的模型。延迟换精度 —— 放在剪枝这种候选量已收窄的环节正合适。
"""

from config.connections import EMBEDDING


class RerankerScorer:
    name = "bge-reranker (cross-encoder)"

    def __init__(self, model_name=None):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name or EMBEDDING.reranker)

    def score(self, query, text):
        return self.score_batch(query, [text])[0]

    def score_batch(self, query, texts):
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        raw = self._model.predict(pairs)  # bge-reranker 输出 logit
        # sigmoid 归一化到 [0,1],便于和 τ_rel 阈值比较
        import math
        return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]


_singleton = None


def get_scorer():
    """惰性单例:重排模型加载较重,全程复用一个实例。"""
    global _singleton
    if _singleton is None:
        _singleton = RerankerScorer()
    return _singleton
