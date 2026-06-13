"""检索循环纯逻辑单测 —— 不依赖 Neo4j / LLM / reranker。

用 mock graph(get_neighbors / describe / get_node_info)+ mock ranker
+ RuleBasedPolicy 验证:
  - beam 扩展 + 剪枝 + 截断
  - 路径去重(已访问边)
  - 双约束终止(θ_stop / D_max)
  - 双层证据分类(Entity vs Event)
"""

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loop.beam import expand, prune_and_truncate, Candidate
from loop.controller import run_loop
from loop.decision import RuleBasedPolicy


@dataclass
class _RunCfg:
    D_MAX: int = 3
    BEAM_WIDTH: int = 3
    TAU_REL: float = 0.3
    THETA_STOP: float = 0.7
    PRUNE_MODE: str = "ratio"
    PRUNE_RATIO: float = 0.92
    PRUNE_FLOOR: float = 0.30


class _MockGraph:
    """玩具图:支持 get_neighbors / describe / get_node_info(双层)。"""

    def __init__(self, edges, descs, types):
        self._adj = {}
        for h, r, t in edges:
            self._adj.setdefault(h, []).append((r, t, "out"))
            self._adj.setdefault(t, []).append((r, h, "in"))
        self._descs = descs
        self._types = types

    def get_neighbors(self, name, undirected=True):
        return self._adj.get(name, [])

    def describe(self, name):
        return self._descs.get(name, name)

    def get_node_info(self, names):
        return {n: {"type": self._types.get(n, "entity"),
                    "content": self._descs.get(n)} for n in names}


class _KeywordRanker:
    """简单 mock:文本含某关键词得高分,否则低分。"""
    name = "mock"

    def __init__(self, hot="hit"):
        self.hot = hot

    def score_batch(self, query, texts):
        return [0.9 if self.hot in t else 0.1 for t in texts]


# ----- beam.py -----

class TestBeam(unittest.TestCase):
    def test_expand_skips_visited_edges(self):
        g = _MockGraph(
            edges=[("A", "to", "B"), ("A", "to", "C")],
            descs={"A": "a", "B": "b", "C": "c"},
            types={"A": "entity", "B": "entity", "C": "entity"},
        )
        visited = {Candidate("B", "to", "out", "A").edge_key}
        cands = expand(g, ["A"], visited)
        self.assertEqual(sorted(c.node for c in cands), ["C"])

    def test_prune_ratio_mode(self):
        cands = [
            Candidate("X1", "r", "out", "A", 0.90),
            Candidate("X2", "r", "out", "A", 0.85),
            Candidate("X3", "r", "out", "A", 0.50),
        ]
        kept, dropped = prune_and_truncate(cands, 0.0, 10,
                                           mode="ratio", ratio=0.92, floor=0.0)
        self.assertEqual([c.node for c in kept], ["X1", "X2"])
        self.assertEqual([c.node for c in dropped], ["X3"])

    def test_prune_floor_protects_low_signal(self):
        cands = [Candidate(f"X{i}", "r", "out", "A", 0.05) for i in range(3)]
        kept, _ = prune_and_truncate(cands, 0.0, 10,
                                     mode="ratio", ratio=0.92, floor=0.30)
        self.assertEqual(kept, [])

    def test_beam_truncates_above_threshold(self):
        cands = [Candidate(f"X{i}", "r", "out", "A", 0.9 - i * 0.01)
                 for i in range(5)]
        kept, dropped = prune_and_truncate(cands, 0.0, 2,
                                           mode="ratio", ratio=0.92, floor=0.0)
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(dropped), 3)


# ----- controller.py:双层 + 终止条件 -----

class TestController(unittest.TestCase):
    def test_dual_layer_classification(self):
        g = _MockGraph(
            edges=[("人物A", "PARTICIPATES", "事件E"),
                   ("人物A", "妻子", "人物B")],
            descs={"人物A": "a hit", "事件E": "story hit", "人物B": "b cold"},
            types={"人物A": "entity", "事件E": "event", "人物B": "entity"},
        )
        ranker = _KeywordRanker(hot="hit")
        policy = RuleBasedPolicy(theta_stop=0.99, d_max=2)
        cfg = _RunCfg(D_MAX=2, BEAM_WIDTH=5, TAU_REL=0.0,
                      PRUNE_RATIO=0.5, PRUNE_FLOOR=0.0)
        state = run_loop(g, ranker, policy, "q hit", ["人物A"], cfg)
        self.assertIn("人物A", state.evidence_entities)
        event_names = [ev["name"] for ev in state.evidence_events]
        self.assertIn("事件E", event_names)
        self.assertNotIn("人物B", state.evidence_entities)

    def test_stop_at_d_max(self):
        g = _MockGraph(
            edges=[("A", "r", "B"), ("B", "r", "C"), ("C", "r", "D")],
            descs={n: f"{n} hit" for n in "ABCD"},
            types={n: "entity" for n in "ABCD"},
        )
        ranker = _KeywordRanker(hot="hit")
        policy = RuleBasedPolicy(theta_stop=2.0, d_max=2)
        cfg = _RunCfg(D_MAX=2, BEAM_WIDTH=2, TAU_REL=0.0,
                      PRUNE_RATIO=0.5, PRUNE_FLOOR=0.0)
        state = run_loop(g, ranker, policy, "q hit", ["A"], cfg)
        self.assertIn("D_max", state.stop_reason)


if __name__ == "__main__":
    unittest.main()
