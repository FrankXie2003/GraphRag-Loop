"""Demo 的 Agent 检索主循环 —— 本 demo 的核心。

控制流(对应架构图主循环,契约与 graphrag_loop/loop/controller.py 一致):

    for r in 1..D_max:
        S1 expand        扩展 frontier 邻居(跳过已访问边)
        S3 score+prune   打分,砍掉 score<τ_rel
        S4 truncate       取 top-N 作为新 frontier,并入证据子图
        S5 decide         估置信度,够了/到顶 → 停;否则下一轮

完整架构里替换的只是各步实现(真 Neo4j / 真重排 / 真 LLM),这套控制流不变。
"""

from dataclasses import dataclass, field

from retrieval.beam import expand, score_candidates, prune_and_truncate


@dataclass
class LoopState:
    """单次检索的循环状态。"""
    frontier: list                      # 当前层待扩展的节点
    visited_edges: set = field(default_factory=set)   # 路径记忆:已走过的边
    evidence_nodes: set = field(default_factory=set)  # 累积证据节点
    evidence_edges: list = field(default_factory=list)  # 证据子图的边(含来源)
    rounds: list = field(default_factory=list)        # 每轮快照,供轨迹打印
    confidence: float = 0.0
    stop_reason: str = ""


def run_loop(graph, ranker, policy, query, seeds, config, tracer=None):
    """执行检索主循环,返回填充好的 LoopState。"""
    state = LoopState(frontier=list(seeds))
    state.evidence_nodes.update(seeds)

    for r in range(1, config.D_MAX + 1):
        # --- S1 扩展 ---
        candidates = expand(graph, state.frontier, state.visited_edges)

        # --- S3 打分 ---
        candidates = score_candidates(ranker, query, candidates, graph)

        # --- S3 剪枝 + S4 截断 ---
        kept, dropped = prune_and_truncate(
            candidates, config.TAU_REL, config.BEAM_WIDTH)

        # 把保留的候选并入证据子图,并登记已访问边(去重的关键)
        for c in kept:
            state.visited_edges.add(c.edge_key)
            state.evidence_nodes.add(c.node)
            state.evidence_edges.append(c)

        # --- S5 决策 ---
        decision = policy.decide(r, kept, state.evidence_nodes)
        state.confidence = decision.confidence

        if tracer:
            tracer.record_round(r, state.frontier, candidates,
                                kept, dropped, decision)

        if decision.stop:
            state.stop_reason = decision.reason
            break

        # 下一轮 frontier = 本轮保留的节点
        state.frontier = [c.node for c in kept]
    else:
        state.stop_reason = "循环正常结束(用尽 D_max)"

    return state
