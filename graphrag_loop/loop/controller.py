"""Agent 检索主循环骨架 —— graphrag_loop 自包含版本。

    for r in 1..D_max:
        S1 expand        扩展 frontier 邻居(跳过已访问边)
        S3 score+prune   打分,砍掉 score<τ_rel
        S4 truncate       取 top-N,并入证据子图
        S5 decide         估置信度,够了/到顶 → 停;否则下一轮

控制流契约与 demo/agent/loop.py 一致(Phase 0 已验证)。Phase 1 起注入的依赖换成
真组件:graph=GraphStore(Neo4j)、ranker=RerankerScorer(bge-reranker)、
policy 后续换成 LLM 版(loop/decision.py)。这套控制流不变。
"""

from loop.beam import expand, score_candidates, prune_and_truncate
from loop.state import LoopState


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
            candidates, config.TAU_REL, config.BEAM_WIDTH,
            mode=getattr(config, "PRUNE_MODE", "ratio"),
            ratio=getattr(config, "PRUNE_RATIO", 0.92),
            floor=getattr(config, "PRUNE_FLOOR", 0.30))

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

        state.frontier = [c.node for c in kept]
    else:
        state.stop_reason = "循环正常结束(用尽 D_max)"

    return state
