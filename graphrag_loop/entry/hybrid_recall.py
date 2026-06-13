"""L1b 混合入口召回 —— 把用户 query 转成 BFS 的起始种子节点。

Phase 2 双层 schema 下,从单路召回升级为**双型分路并集**:

  ① 实体侧:整句向量召回 entity 向量 + 软链接(每个 mention 匹配 entity)
     好处:命中 query 里的人物/地点,作为 BFS 的"骨架"起点。
  ② 事件侧:整句向量召回 event 向量(用事件 content 检索)
     好处:对情节/续写类 query("中秋夜发生了什么"),直接命中 Event 节点的情节文本。

两路结果去重赋权:每个节点取在各路中的最高分。同名节点(理论不会出现,因为 Entity/Event
名字空间不同)以更高分为准。
两路混合的种子直接进 BFS,loop 既能从人物起跳、也能从事件起跳。

注:Phase 1 时只有 entity 一路;Phase 2 加 event 路是双层 schema 价值兑现的关键——
情节问题不再需要先从人物出发再走两跳到事件,可以直接命中事件起跳,缩短推理路径。
"""

from config.settings import PARAMS
from entry.soft_linker import soft_link


def hybrid_recall(query, embedder, vec_store, llm=None,
                  seed_topk=None, per_mention_k=3):
    """返回种子列表 [(name, weight, type), ...] 按 weight 降序,截到 seed_topk。

    type ∈ {'entity', 'event'},供 tracer 展示与下游可能的差异化处理使用。
    """
    seed_topk = seed_topk or PARAMS.SEED_TOPK

    # name -> (best_score, type)
    scored = {}

    def add(name, score, ntype):
        if not name:
            return
        if name not in scored or score > scored[name][0]:
            scored[name] = (score, ntype)

    q_vec = embedder.encode(query)

    # ① 实体路:整句召回 entity + 软链接
    for payload, s in vec_store.search(q_vec, top_k=seed_topk, type_filter="entity"):
        add(payload.get("name"), s, "entity")
    for name, s in soft_link(query, embedder, vec_store,
                             llm=llm, top_k=per_mention_k,
                             type_filter="entity"):
        add(name, s, "entity")

    # ② 事件路:整句召回 event(用事件 content 向量)
    for payload, s in vec_store.search(q_vec, top_k=seed_topk, type_filter="event"):
        add(payload.get("name"), s, "event")

    seeds = sorted(scored.items(), key=lambda kv: kv[1][0], reverse=True)
    return [(name, score, ntype) for name, (score, ntype) in seeds[:seed_topk]]
