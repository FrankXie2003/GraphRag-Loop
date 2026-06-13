"""反思层纯逻辑单测 —— 不依赖 NLI 模型 / LLM API。

测试覆盖:
  - generator 的 _split_segments / _is_honest_refusal / _build_evidence_block
  - citation 的 resolve_citations / evidences_for_segment
  - refeed 的 verify_answer 短路逻辑(mock 一个无 NLI、无 LLM 调用的诚实拒答)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestGenerator(unittest.TestCase):
    def test_split_segments_extracts_citations(self):
        from generation.generator import _split_segments
        text = "甄士隐赠银 50 两 [E1]。还赠了冬衣 [E1][E2]。"
        segs = _split_segments(text)
        self.assertEqual(segs[0]["cited_ids"], [1])
        self.assertEqual(segs[1]["cited_ids"], [1, 2])
        self.assertNotIn("[E1]", segs[0]["clean"])

    def test_split_segments_handles_no_citation(self):
        from generation.generator import _split_segments
        segs = _split_segments("甄士隐很慷慨。")
        self.assertEqual(segs[0]["cited_ids"], [])

    def test_is_honest_refusal_detects_phrases(self):
        from generation.generator import _is_honest_refusal
        for txt in ["原文未提供该信息。", "证据不足,无法回答。", "记录中没有相关内容。"]:
            self.assertTrue(_is_honest_refusal(txt), f"应识别: {txt}")
        for txt in ["甄士隐赠银 50 两 [E1]。", "贾雨村吟诗 [E2]。"]:
            self.assertFalse(_is_honest_refusal(txt), f"不应识别: {txt}")

    def test_build_evidence_block_contains_events_and_entities(self):
        from generation.generator import _build_evidence_block
        # mock state-like 对象
        class S:
            evidence_events = [
                {"name": "中秋资助", "content": "甄士隐赠银 50 两"},
            ]
            evidence_entities = {"甄士隐", "贾雨村"}

        block, evmap = _build_evidence_block(S())
        self.assertIn("[E1]", block)
        self.assertIn("中秋资助", block)
        self.assertIn("[E2]", block)  # 实体集合也占一个 ID
        self.assertEqual(evmap[1]["kind"], "event")
        self.assertEqual(evmap[2]["kind"], "entities")


class TestCitation(unittest.TestCase):
    def test_resolve_citations(self):
        from generation.citation import resolve_citations
        evmap = {1: {"kind": "event", "name": "E1", "content": "c1"},
                 2: {"kind": "entities", "name": "all", "content": "甲, 乙"}}
        seg = {"text": "...", "cited_ids": [1, 2]}
        result = resolve_citations(seg, evmap)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "E1")

    def test_evidences_for_segment(self):
        from generation.citation import evidences_for_segment
        evmap = {1: {"kind": "event", "name": "E1", "content": "c1"},
                 2: {"kind": "event", "name": "E2", "content": ""}}  # 空 content 被过滤
        seg = {"text": "...", "cited_ids": [1, 2]}
        result = evidences_for_segment(seg, evmap)
        self.assertEqual(result, ["c1"])

    def test_no_citation_returns_empty(self):
        from generation.citation import evidences_for_segment
        seg = {"text": "...", "cited_ids": []}
        self.assertEqual(evidences_for_segment(seg, {}), [])


class TestVerifyAnswerShortCircuit(unittest.TestCase):
    """诚实拒答应跳过 IsSup,不调 LLM/NLI。"""

    def test_honest_refusal_short_circuits(self):
        from generation.refeed import verify_answer
        from generation.generator import GeneratedAnswer
        from reflection.tokens import IsSup

        ans = GeneratedAnswer(
            text="原文未提供相关信息。",
            segments=[{"text": "原文未提供相关信息。",
                       "clean": "原文未提供相关信息。", "cited_ids": []}],
            evidence_map={},
            is_honest_refusal=True,
        )
        # 不传 llm —— 验证若 LLM 被调用会立刻报错
        verified = verify_answer(ans, llm=None)
        self.assertTrue(verified.fully_supported)
        self.assertEqual(len(verified.judgements), 1)
        self.assertEqual(verified.judgements[0]["token"], IsSup.FULL)
        self.assertIn("诚实拒答", verified.judgements[0]["reason"])

    def test_no_citation_segment_marked_none(self):
        """非诚实拒答但段落无引用 → 直接 NONE,触发回扩。"""
        from generation.refeed import verify_answer
        from generation.generator import GeneratedAnswer
        from reflection.tokens import IsSup

        ans = GeneratedAnswer(
            text="贾雨村做了一件事。",
            segments=[{"text": "贾雨村做了一件事。",
                       "clean": "贾雨村做了一件事。", "cited_ids": []}],
            evidence_map={1: {"kind": "event", "content": "..."}},
            is_honest_refusal=False,
        )
        verified = verify_answer(ans, llm=None)
        self.assertFalse(verified.fully_supported)
        self.assertEqual(verified.judgements[0]["token"], IsSup.NONE)


if __name__ == "__main__":
    unittest.main()
