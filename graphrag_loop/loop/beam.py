"""Beam 扩展与截断 —— 防"图爆炸"的核心宽度控制。

一轮 = expand(对 frontier 取邻居,跳过已访问边)
       → score(每个候选用 ranker 打分)
       → prune(score < tau_rel 砍掉)
       → truncate(剩下的取 top-N)

对应架构图 S1(扩展)+ S3(打分剪枝)+ S4(Beam 截断)。
S2 关系剪枝在 Phase 1 里合进 S3 的统一打分(把关系名纳入文本)。

逻辑契约与 demo/retrieval/beam.py 一致(Phase 0 已验证),
这里是 graphrag_loop 自包含的版本,不依赖 demo 包。
"""

from dataclasses import dataclass


@dataclass
class Candidate:
    """一个候选邻居,带产生它的边和打分,用于轨迹展示和后续证据收集。"""
    node: str
    relation: str
    direction: str   # 'out' / 'in'
    parent: str      # 从哪个 frontier 节点扩展来的
    score: float = 0.0

    @property
    def edge_key(self):
        # 无向去重:把端点排序,使 (A,rel,B) 与 (B,rel,A) 视作同一条边。
        a, b = sorted([self.parent, self.node])
        return (a, self.relation, b)


def expand(graph, frontier, visited_edges):
    """S1:对 frontier 每个节点取邻居,跳过已访问边。返回候选列表(未打分)。"""
    candidates = []
    seen_this_round = set()  # 防止本轮内多个父节点产生同一候选边的重复
    for node in frontier:
        for relation, neighbor, direction in graph.get_neighbors(node):
            cand = Candidate(node=neighbor, relation=relation,
                             direction=direction, parent=node)
            key = cand.edge_key
            if key in visited_edges or key in seen_this_round:
                continue  # 路径记忆:这条边走过了,跳过 → 避免环里打转
            seen_this_round.add(key)
            candidates.append(cand)
    return candidates


def score_candidates(ranker, query, candidates, graph):
    """S3 打分:用 [关系类型 + 邻居节点描述] 与 query 算相关性,写回 candidate.score。

    关键:必须把"关系类型"纳入打分文本,而不是只看邻居节点描述。
    否则像"总部位于"这种与问题强相关的边会被忽略(ToG 之所以先做 S2 关系剪枝
    正是这个原因)。这里把关系名拼进文本,等价于在单次打分里合并了 S2+S3。
    """
    if not candidates:
        return candidates
    texts = [f"{c.relation} {graph.describe(c.node)}" for c in candidates]
    scores = ranker.score_batch(query, texts)
    for c, s in zip(candidates, scores):
        c.score = s
    return candidates


def prune_and_truncate(candidates, tau_rel, beam_width,
                       mode="ratio", ratio=0.92, floor=0.30):
    """S3 剪枝 + S4 截断,再取 top-N。返回 (kept, dropped)。

    剪枝模式(缓解 τ_rel 对绝对分数敏感的问题):
      'absolute' —— score >= tau_rel(老办法,对打分器分布敏感)
      'ratio'    —— score >= max_score × ratio 且 score >= floor
                    跟绝对值脱钩:无论模型整体打分偏高/偏低,都按"相对本轮最优"剪。
                    floor 是绝对下限,防止全员低分时把垃圾也当成"相对高分"全留下。
      'gap'      —— 在排序后相邻分数差最大处切割(信号/噪声间通常有断崖)。
                    同样叠加 floor 兜底。
    """
    if not candidates:
        return [], []

    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    top = ranked[0].score

    if mode == "absolute":
        cut = lambda c: c.score >= tau_rel
    elif mode == "gap":
        # 找最大断崖位置 i,保留 [0..i];再叠加 floor 下限
        split = len(ranked)
        max_gap = -1.0
        for i in range(len(ranked) - 1):
            gap = ranked[i].score - ranked[i + 1].score
            if gap > max_gap:
                max_gap = gap
                split = i + 1
        keep_ids = set(id(c) for c in ranked[:split] if c.score >= floor)
        cut = lambda c: id(c) in keep_ids
    else:  # 'ratio'(默认)
        thresh = max(top * ratio, floor)
        cut = lambda c: c.score >= thresh

    survivors = [c for c in ranked if cut(c)]
    pruned_by_rule = [c for c in ranked if not cut(c)]

    kept = survivors[:beam_width]
    dropped_by_beam = survivors[beam_width:]
    dropped = pruned_by_rule + dropped_by_beam
    return kept, dropped
