"""端到端单测 —— 玩具图谱上跑通 query → 答案,断言证据可溯源。

这里仍不依赖 Neo4j / LLM:用 mock graph + mock ranker + RuleBasedPolicy。
真实端到端验证用 run_phase2.py / verify_*.py(那些需要外部服务)。

本测试的价值:任何破坏控制流契约(loop/state/decision/generator 协议)的改动,
都会被这里的"小型完整链路"捕获。
"""

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loop.controller import run_loop
from loop.decision import RuleBasedPolicy


@dataclass
class _RunCfg:
    D_MAX: int = 3
    BEAM_WIDTH: int = 3
    TAU_REL: float = 0.0
    THETA_STOP: float = 0.99    # 让循环跑满 D_max,便于检验完整轨迹
    PRUNE_MODE: str = "ratio"
    PRUNE_RATIO: float = 0.5    # 不剪太狠
    PRUNE_FLOOR: float = 0.0


class _ToyGraph:
    """玩具双层图谱:
        甄士隐 -[妻子]-> 封氏
        甄士隐 -[PARTICIPATES]-> E1(中秋资助贾雨村)
        贾雨村 -[PARTICIPATES]-> E1
        E1 -[NEXT]-> E2(英莲走失)
        甄士隐 -[PARTICIPATES]-> E2
    """
    EDGES = [
        ("甄士隐", "妻子", "封氏"),
        ("甄士隐", "PARTICIPATES", "E1"),
        ("贾雨村", "PARTICIPATES", "E1"),
        ("E1", "NEXT", "E2"),
        ("甄士隐", "PARTICIPATES", "E2"),
    ]
    DESCS = {
        "甄士隐": "甄士隐 隐居望族",
        "封氏": "封氏 嫡妻",
        "贾雨村": "贾雨村 落魄书生",
        "E1": "中秋夜甄士隐资助贾雨村五十两白银和冬衣",
        "E2": "元宵夜英莲被霍启走失",
    }
    TYPES = {"甄士隐": "entity", "封氏": "entity", "贾雨村": "entity",
             "E1": "event", "E2": "event"}

    def __init__(self):
        self._adj = {}
        for h, r, t in self.EDGES:
            self._adj.setdefault(h, []).append((r, t, "out"))
            self._adj.setdefault(t, []).append((r, h, "in"))

    def get_neighbors(self, name, undirected=True):
        return self._adj.get(name, [])

    def describe(self, name):
        return self.DESCS.get(name, name)

    def get_node_info(self, names):
        return {n: {"type": self.TYPES.get(n, "entity"),
                    "content": self.DESCS.get(n)} for n in names}


class _RankByKeyword:
    name = "kw"
    def __init__(self, kw):
        self.kw = kw
    def score_batch(self, query, texts):
        return [0.9 if self.kw in t else 0.3 for t in texts]


class TestE2E(unittest.TestCase):
    def test_evidence_traceable_to_seeds(self):
        """证据子图里的边都能通过 parent → node 链回种子。"""
        graph = _ToyGraph()
        ranker = _RankByKeyword(kw="资助")
        policy = RuleBasedPolicy(theta_stop=0.99, d_max=2)
        cfg = _RunCfg()

        state = run_loop(graph, ranker, policy,
                         "甄士隐资助贾雨村", ["甄士隐"], cfg)

        # 1. 至少触达 E1(资助事件)
        event_names = {ev["name"] for ev in state.evidence_events}
        self.assertIn("E1", event_names)

        # 2. 证据边的所有 parent 都在已知节点里(可溯源)
        all_nodes = {"甄士隐"} | {c.node for c in state.evidence_edges}
        for c in state.evidence_edges:
            self.assertIn(c.parent, all_nodes,
                          f"边 {c.parent} -> {c.node} 的 parent 无法溯源")

    def test_no_visited_edge_repeated(self):
        graph = _ToyGraph()
        ranker = _RankByKeyword(kw="资助")
        policy = RuleBasedPolicy(theta_stop=0.99, d_max=3)
        cfg = _RunCfg(D_MAX=3, BEAM_WIDTH=10)

        state = run_loop(graph, ranker, policy,
                         "甄士隐资助贾雨村", ["甄士隐"], cfg)

        keys = [c.edge_key for c in state.evidence_edges]
        self.assertEqual(len(keys), len(set(keys)),
                         "证据边出现重复,路径去重失败")

    def test_event_evidence_carries_content(self):
        """Event 证据必须带 content(给 LLM 决策/生成用)。"""
        graph = _ToyGraph()
        ranker = _RankByKeyword(kw="资助")
        policy = RuleBasedPolicy(theta_stop=0.99, d_max=2)
        cfg = _RunCfg()

        state = run_loop(graph, ranker, policy,
                         "资助", ["甄士隐"], cfg)
        for ev in state.evidence_events:
            self.assertTrue(ev.get("content"),
                            f"Event {ev['name']} 没有 content")


if __name__ == "__main__":
    unittest.main()
