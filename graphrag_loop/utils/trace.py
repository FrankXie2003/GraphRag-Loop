"""轨迹记录与打印 —— 看清每跳发生了什么(逐跳候选/打分/保留/剪枝/置信度)。

逻辑契约与 demo/utils/trace.py 一致(Phase 0 已验证),graphrag_loop 自包含版本。
"""


class Tracer:
    def __init__(self, tau_rel, beam_width):
        self.tau_rel = tau_rel
        self.beam_width = beam_width

    def header(self, query, seeds, config):
        print("=" * 70)
        print(f"Query : {query}")
        print(f"Seeds : {seeds}")
        print(f"超参  : D_max={config.D_MAX}  beam_width={config.BEAM_WIDTH}  "
              f"τ_rel={config.TAU_REL}  θ_stop={config.THETA_STOP}")
        print("=" * 70)

    def record_round(self, r, frontier, candidates, kept, dropped, decision):
        print(f"\n── 第 {r} 跳 ──────────────────────────────────────────────")
        print(f"  frontier: {frontier}")
        if not candidates:
            print("  扩展候选: (无 —— 邻居都已访问过或无邻居)")
        else:
            print(f"  扩展候选({len(candidates)} 个,已按分排序):")
            kept_set = set(id(c) for c in kept)
            # 被保留的最低分:低于它且未保留 = 被剪枝;不低于它但未保留 = 被 beam 截断
            kept_min = min((c.score for c in kept), default=0.0)
            for c in sorted(candidates, key=lambda x: x.score, reverse=True):
                arrow = f"-[{c.relation}]->" if c.direction == "out" else f"<-[{c.relation}]-"
                if id(c) in kept_set:
                    mark = "[KEEP] 保留"
                elif c.score >= kept_min and kept:
                    mark = "[DROP] 截断(超出 beam_width)"
                else:
                    mark = "[DROP] 剪枝(未达剪枝线)"
                print(f"     [{c.score:4.2f}] {c.parent} {arrow} {c.node}  {mark}")
        print(f"  → 决策: {'停止' if decision.stop else '继续'} "
              f"(conf={decision.confidence:.2f}) — {decision.reason}")

    def footer(self, state):
        print("\n" + "=" * 70)
        print(f"终止原因: {state.stop_reason}")
        print(f"最终置信度: {state.confidence:.2f}")
        print(f"证据节点: {sorted(state.evidence_nodes)}")
        print("证据子图(检索路径):")
        for c in state.evidence_edges:
            arrow = f"-[{c.relation}]->" if c.direction == "out" else f"<-[{c.relation}]-"
            print(f"     {c.parent} {arrow} {c.node}  (score={c.score:.2f})")
        print("=" * 70)


def build_answer(query, state):
    """Phase 1:基于证据子图用模板拼答案(生成留到 generation/generator.py 接 LLM)。"""
    nodes = sorted(state.evidence_nodes)
    return (f"\n[模板答案] 针对「{query}」,检索在 {len(state.evidence_edges)} 步内"
            f"触达 {len(nodes)} 个证据节点:{nodes}。\n"
            f"          (后续由 generation/generator.py 把证据子图交给 LLM 生成自然语言答案。)")
