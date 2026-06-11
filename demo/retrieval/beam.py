"""Beam 扩展与截断 —— 防"图爆炸"的核心宽度控制。

一轮 = expand(对 frontier 取邻居,跳过已访问边)
       → score(每个候选用 ranker 打分)
       → prune(score < tau_rel 砍掉)
       → truncate(剩下的取 top-N)

对应架构图 S1(扩展)+ S3(打分剪枝)+ S4(Beam 截断)。
S2 关系剪枝在 demo 里省略,合进 S3 的统一打分。
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


def prune_and_truncate(candidates, tau_rel, beam_width):
    """S3 剪枝 + S4 截断:砍掉 score<tau_rel,再按分数取 top-N。

    返回 (kept, dropped) 便于轨迹展示哪些被砍、为什么。
    """
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    survivors = [c for c in ranked if c.score >= tau_rel]
    pruned_by_threshold = [c for c in ranked if c.score < tau_rel]

    kept = survivors[:beam_width]
    dropped_by_beam = survivors[beam_width:]
    dropped = pruned_by_threshold + dropped_by_beam
    return kept, dropped
