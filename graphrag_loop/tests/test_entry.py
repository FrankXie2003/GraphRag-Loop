"""入口召回纯逻辑单测 —— 不依赖 Qdrant / LLM。

测 hybrid_recall 的去重赋权策略(用 mock 的 vec_store + embedder)。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _MockEmbedder:
    dim = 4
    def encode(self, texts, normalize=True):
        if isinstance(texts, str):
            return [0.0] * 4
        return [[0.0] * 4 for _ in texts]


class _MockVecStore:
    """根据 type_filter 返回不同结果的 mock vec_store。"""
    def __init__(self, entity_results, event_results):
        self.entity_results = entity_results
        self.event_results = event_results

    def search(self, vec, top_k=5, type_filter=None):
        pool = (self.entity_results if type_filter == "entity"
                else self.event_results if type_filter == "event"
                else self.entity_results + self.event_results)
        return pool[:top_k]


class _MockLLM:
    """mock LLM:抽 mention 直接返回固定列表。"""
    def __init__(self, mentions):
        self._mentions = mentions

    def chat(self, prompt, **kwargs):
        import json
        return json.dumps(self._mentions, ensure_ascii=False)


class TestHybridRecall(unittest.TestCase):
    def test_dedup_keeps_highest_score(self):
        """同名节点出现在多条路径,取最高分。"""
        from entry.hybrid_recall import hybrid_recall

        # entity 路:整句召回 + 软链接都返回"贾雨村"
        # 整句给 0.5,软链接给 0.9 → 应取 0.9
        ent_pool = [({"name": "贾雨村", "type": "entity"}, 0.5),
                    ({"name": "其他人", "type": "entity"}, 0.3)]
        # 软链接也匹配到贾雨村,分更高
        # 这里偷懒:hybrid_recall 内部对 mention 调 search,再次返回 ent_pool
        # 第一次整句:0.5;mention=贾雨村 → search 返回相同 ent_pool 第一个 = 0.5
        # 但软链接 mention 触发新一次 search,这里 mock 返回更高的
        class _BoostingVecStore:
            calls = 0
            def search(self2, vec, top_k=5, type_filter=None):
                if type_filter == "event":
                    return []
                self2.calls += 1
                # 第一次(整句)给 0.5,后续(mention)给 0.9
                score = 0.9 if self2.calls > 1 else 0.5
                return [({"name": "贾雨村", "type": "entity"}, score)]

        vec = _BoostingVecStore()
        seeds = hybrid_recall("贾雨村做了什么", _MockEmbedder(), vec,
                              llm=_MockLLM(["贾雨村"]), seed_topk=5,
                              per_mention_k=1)
        names = [s[0] for s in seeds]
        self.assertEqual(names.count("贾雨村"), 1)  # 去重
        weights = {s[0]: s[1] for s in seeds}
        self.assertAlmostEqual(weights["贾雨村"], 0.9)  # 取最高分

    def test_dual_type_routing(self):
        """entity 和 event 来自不同搜索路径,均出现在结果中。"""
        from entry.hybrid_recall import hybrid_recall

        ent_results = [({"name": "甄士隐", "type": "entity"}, 0.7)]
        ev_results = [({"name": "中秋资助赴考", "type": "event"}, 0.85)]
        vec = _MockVecStore(ent_results, ev_results)
        seeds = hybrid_recall("question", _MockEmbedder(), vec,
                              llm=_MockLLM([]), seed_topk=5)
        types = {s[0]: s[2] for s in seeds}
        self.assertEqual(types.get("甄士隐"), "entity")
        self.assertEqual(types.get("中秋资助赴考"), "event")

    def test_seeds_sorted_descending(self):
        from entry.hybrid_recall import hybrid_recall

        ent = [({"name": "Z", "type": "entity"}, 0.1),
               ({"name": "A", "type": "entity"}, 0.9)]
        vec = _MockVecStore(ent, [])
        seeds = hybrid_recall("q", _MockEmbedder(), vec,
                              llm=_MockLLM([]), seed_topk=5)
        weights = [s[1] for s in seeds]
        self.assertEqual(weights, sorted(weights, reverse=True))


if __name__ == "__main__":
    unittest.main()
