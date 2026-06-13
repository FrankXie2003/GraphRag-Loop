"""Agent 检索主循环骨架 —— graphrag_loop 自包含版本。

    for r in 1..D_max:
        S1 expand        扩展 frontier 邻居(跳过已访问边)
        S3 score+prune   打分,砍掉 score<τ_rel
        S4 truncate       取 top-N,并入证据子图(Phase 2:分类到 entity/event)
        S5 decide         估置信度,够了/到顶 → 停;否则下一轮

控制流契约与 demo/agent/loop.py 一致(Phase 0 已验证)。
Phase 2 双层适配点:
  - 每轮新加入证据的节点,通过 graph.get_node_info 一次拿到 type+content,
    分别填到 state.evidence_entities / evidence_events
  - 调用 policy.decide 时多传 state,LLM 决策能读 Event content 判断信息够不够
"""

from loop.beam import expand, score_candidates, prune_and_truncate
from loop.state import LoopState


def _classify_into_state(graph, new_nodes, state):
    """把本轮新加入证据的节点分类填到 state.evidence_entities / evidence_events。"""
    if not new_nodes:
        return
    info = graph.get_node_info(new_nodes) if hasattr(graph, "get_node_info") else {}
    for name in new_nodes:
        meta = info.get(name) or {}
        if meta.get("type") == "event":
            # 去重:同名事件只加一次
            if not any(ev["name"] == name for ev in state.evidence_events):
                state.evidence_events.append({
                    "name": name,
                    "content": meta.get("content") or "",
                })
        else:
            state.evidence_entities.add(name)


def run_loop(graph, ranker, policy, query, seeds, config, tracer=None):
    """执行检索主循环,返回填充好的 LoopState。"""
    state = LoopState(frontier=list(seeds))
    state.evidence_nodes.update(seeds)
    # 把种子也分类(可能种子本身就是 Event)
    _classify_into_state(graph, seeds, state)

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

        # 把保留的候选并入证据子图(平铺 + 分类)
        new_nodes_this_round = []
        for c in kept:
            state.visited_edges.add(c.edge_key)
            if c.node not in state.evidence_nodes:
                new_nodes_this_round.append(c.node)
            state.evidence_nodes.add(c.node)
            state.evidence_edges.append(c)
        _classify_into_state(graph, new_nodes_this_round, state)

        # --- S5 决策(传 query+state,让 LLM 版能读到事件 content)---
        decision = policy.decide(r, kept, state.evidence_nodes,
                                 query=query, state=state)
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
