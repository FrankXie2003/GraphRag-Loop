"""L1b 混合入口召回 —— 把用户 query 转成 BFS 的起始种子节点。

两路并行,取并集:
  ① 整句向量召回:把整个 query 向量化,在图节点向量里找最相似的 top-k。
     好处:即使 query 没有明确实体词(如"那个送银子的人"),也能靠语义召回。
  ② 软链接:抽 query 里的实体提及,每个 mention 向量匹配若干候选节点(soft_linker)。
     好处:对明确实体更精准,多 mention 多入口。

两路结果去重,每个节点取其在各路中的最高分作为 seed_weight。
宁可多给几个种子(SEED_TOPK)让后续 loop 的剪枝去收敛,也不在入口赌单点精确。

这替换了 Phase 1 早期手写死的 seeds —— 现在用户给自然语言,系统自动找入口。
"""

from config.settings import PARAMS
from entry.soft_linker import soft_link


def hybrid_recall(query, embedder, vec_store, llm=None,
                  seed_topk=None, per_mention_k=3):
    """返回种子列表 [(name, weight), ...] 按 weight 降序,截到 seed_topk。"""
    seed_topk = seed_topk or PARAMS.SEED_TOPK

    scored = {}  # name -> 最高分

    def add(name, score):
        if name and (name not in scored or score > scored[name]):
            scored[name] = score

    # ① 整句向量召回
    q_vec = embedder.encode(query)
    for payload, score in vec_store.search(q_vec, top_k=seed_topk):
        add(payload.get("name"), score)

    # ② 软链接(每个 mention 的候选)
    for name, score in soft_link(query, embedder, vec_store,
                                 llm=llm, top_k=per_mention_k):
        add(name, score)

    seeds = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    return seeds[:seed_topk]
