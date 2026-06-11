"""节点/路径相关性打分。

默认实现:词重叠(Jaccard-ish)打分 —— 零依赖、确定性,便于单测和肉眼验证。
可选实现:句向量 cosine(若安装了 sentence-transformers),更接近真实语义。

输出统一归一化到 [0, 1],供 beam 截断和 tau_rel 剪枝使用。
对应完整架构 reflection/isrel.py 的 Cross-Encoder / NLI 版本 —— 接口一致(query, text)->score。
"""

import re


def _tokenize(text):
    # 中英混合的极简分词:抽连续的中文字或英文单词。
    # 中文按单字切(玩具规模够用),英文按词切。
    tokens = re.findall(r"[a-zA-Z]+|[一-鿿]", text.lower())
    return set(tokens)


class WordOverlapRanker:
    """词重叠打分器:score = |query∩text| / |query|。

    确定性、无外部依赖。query 中的词在节点描述里命中越多分越高,
    这样'省/位于/行政'类信号节点会高于'景点/教师'类噪声节点。
    """

    name = "word-overlap"

    def score(self, query, text):
        q = _tokenize(query)
        if not q:
            return 0.0
        t = _tokenize(text)
        overlap = len(q & t)
        return overlap / len(q)

    def score_batch(self, query, texts):
        return [self.score(query, t) for t in texts]


class EmbeddingRanker:
    """可选:句向量 cosine 打分。仅在显式启用且装了 sentence-transformers 时使用。"""

    name = "embedding-cosine"

    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer  # 延迟导入,避免硬依赖
        self._model = SentenceTransformer(model_name)

    def score(self, query, text):
        return self.score_batch(query, [text])[0]

    def score_batch(self, query, texts):
        import numpy as np
        vecs = self._model.encode([query] + list(texts), normalize_embeddings=True)
        qv, tv = vecs[0], vecs[1:]
        sims = tv @ qv  # 已归一化,点积即 cosine
        # cosine ∈ [-1,1] → 映射到 [0,1]
        return [float((s + 1) / 2) for s in sims]


def get_ranker(use_embedding=False):
    """工厂:默认词重叠;use_embedding=True 且依赖可用时才用句向量。"""
    if use_embedding:
        try:
            return EmbeddingRanker()
        except Exception as e:  # 依赖缺失/加载失败 → 退回词重叠,demo 仍可跑
            print(f"[ranker] 句向量不可用({e}),回退到 word-overlap")
    return WordOverlapRanker()
