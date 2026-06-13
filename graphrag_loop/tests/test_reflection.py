"""反思层纯逻辑单测 —— 不依赖 NLI 模型 / LLM API。

测试覆盖:
  - generator 的 _split_segments / _is_honest_refusal / _build_evidence_block / _parse_generation
  - citation 的 resolve_citations / evidences_for_segment
  - refeed 的 verify_answer:
      * 诚实拒答短路
      * 原子断言级逐条验证(question&solution.md #13 升级)
      * 全 FULL → fully_supported,任何一条非 FULL → 不 fully_supported
      * atomic_claims 缺失 → fallback 到 segments(向后兼容)
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
        class S:
            evidence_events = [
                {"name": "中秋资助", "content": "甄士隐赠银 50 两"},
            ]
            evidence_entities = {"甄士隐", "贾雨村"}

        block, evmap = _build_evidence_block(S())
        self.assertIn("[E1]", block)
        self.assertIn("中秋资助", block)
        self.assertIn("[E2]", block)
        self.assertEqual(evmap[1]["kind"], "event")
        self.assertEqual(evmap[2]["kind"], "entities")


class TestParseGeneration(unittest.TestCase):
    """JSON 解析双轨结构 + 容错(question&solution.md #13)。"""

    def test_parses_valid_json(self):
        from generation.generator import _parse_generation
        raw = '''```json
{
  "text": "甄士隐资助 [E1]。",
  "atomic_claims": [
    {"text": "甄士隐资助贾雨村", "cited_ids": [1]},
    {"text": "贾雨村赴京", "cited_ids": [1, 2]}
  ]
}
```'''
        text, claims = _parse_generation(raw)
        self.assertEqual(text, "甄士隐资助 [E1]。")
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["cited_ids"], [1])
        self.assertEqual(claims[1]["cited_ids"], [1, 2])
        self.assertEqual(claims[0]["clean"], "甄士隐资助贾雨村")

    def test_parses_text_inline_citations_into_ids(self):
        """LLM 没在 cited_ids 字段填值,但 text 里有 [E#] —— 应被吸收进 ids。"""
        from generation.generator import _parse_generation
        raw = '{"text":"x","atomic_claims":[{"text":"事实 [E3]"}]}'
        _, claims = _parse_generation(raw)
        self.assertEqual(claims[0]["cited_ids"], [3])

    def test_returns_none_claims_on_invalid_json(self):
        """JSON 解析失败时 atomic_claims=None,调用方 fallback。"""
        from generation.generator import _parse_generation
        text, claims = _parse_generation("纯文本答案 [E1]。")
        self.assertIsNone(claims)

    def test_string_cited_ids_normalized(self):
        """cited_ids: ["1", "2"] 应规范化为 [1, 2]。"""
        from generation.generator import _parse_generation
        raw = '{"text":"x","atomic_claims":[{"text":"a","cited_ids":["1","2"]}]}'
        _, claims = _parse_generation(raw)
        self.assertEqual(claims[0]["cited_ids"], [1, 2])


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
                 2: {"kind": "event", "name": "E2", "content": ""}}
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
            atomic_claims=[],
            segments=[],
            evidence_map={},
            is_honest_refusal=True,
        )
        verified = verify_answer(ans, llm=None)
        self.assertTrue(verified.fully_supported)
        self.assertEqual(len(verified.judgements), 1)
        self.assertEqual(verified.judgements[0]["token"], IsSup.FULL)
        self.assertIn("诚实拒答", verified.judgements[0]["reason"])


class TestAtomicClaimVerification(unittest.TestCase):
    """原子断言级 IsSup(question&solution.md #13)。

    用 monkey-patch 替换 reflection.issup.verify,完全脱离外部 NLI/LLM,
    只测 verify_answer 自身的聚合逻辑。
    """

    def setUp(self):
        """每个测试前 patch verify,测试后恢复。"""
        from reflection import issup
        self._orig_verify = issup.verify
        # 子类自定义 _verify_returns 来控制 mock 行为
        # 默认:claim 包含"FULL" → FULL,否则 NONE
        from reflection.issup import Judgement
        from reflection.tokens import IsSup as _IsSup

        def _mock(claim, evidences, llm=None, tau_sup=None):
            if "FULL" in claim:
                return Judgement(_IsSup.FULL, 1.0, "mock: FULL")
            return Judgement(_IsSup.NONE, 0.0, "mock: NONE")

        # generation/refeed.py 里 import 时绑定了 verify,要 patch 那个
        from generation import refeed as refeed_mod
        self._orig_refeed_verify = refeed_mod.verify
        refeed_mod.verify = _mock

    def tearDown(self):
        from generation import refeed as refeed_mod
        refeed_mod.verify = self._orig_refeed_verify

    def test_all_claims_full_means_fully_supported(self):
        """所有原子断言都 FULL → fully_supported=True。"""
        from generation.refeed import verify_answer
        from generation.generator import GeneratedAnswer

        ans = GeneratedAnswer(
            text="x",
            atomic_claims=[
                # mock 看到 'FULL' 关键字就返回 FULL
                {"text": "claim FULL 1", "clean": "claim FULL 1", "cited_ids": [1]},
                {"text": "claim FULL 2", "clean": "claim FULL 2", "cited_ids": [1]},
            ],
            evidence_map={1: {"kind": "event", "content": "evidence text"}},
            is_honest_refusal=False,
        )
        verified = verify_answer(ans, llm=None)
        self.assertTrue(verified.fully_supported)
        self.assertEqual(len(verified.judgements), 2)

    def test_one_claim_not_full_breaks_aggregation(self):
        """任何一条非 FULL → fully_supported=False(治本:不再"取最大值"假阳性)。"""
        from generation.refeed import verify_answer
        from generation.generator import GeneratedAnswer
        from reflection.tokens import IsSup

        ans = GeneratedAnswer(
            text="x",
            atomic_claims=[
                {"text": "claim FULL 1", "clean": "claim FULL 1", "cited_ids": [1]},
                # claim 2 没引用 → 代码内置规则直接 NONE,不走 mock
                {"text": "claim2", "clean": "claim2", "cited_ids": []},
            ],
            evidence_map={1: {"kind": "event", "content": "..."}},
            is_honest_refusal=False,
        )
        verified = verify_answer(ans, llm=None)
        self.assertFalse(verified.fully_supported)
        tokens = [j["token"] for j in verified.judgements]
        self.assertIn(IsSup.FULL, tokens)
        self.assertIn(IsSup.NONE, tokens)

    def test_fallback_to_segments_when_no_atomic_claims(self):
        """atomic_claims 为空时退回 segments(向后兼容老答案/JSON 解析失败)。"""
        from generation.refeed import verify_answer
        from generation.generator import GeneratedAnswer
        from reflection.tokens import IsSup

        ans = GeneratedAnswer(
            text="x",
            atomic_claims=[],
            segments=[{"text": "段落", "clean": "段落", "cited_ids": []}],
            evidence_map={},
            is_honest_refusal=False,
        )
        verified = verify_answer(ans, llm=None)
        self.assertFalse(verified.fully_supported)
        self.assertEqual(verified.judgements[0]["token"], IsSup.NONE)

    def test_empty_items_not_fully_supported(self):
        """既无 atomic_claims 也无 segments → fully_supported=False。"""
        from generation.refeed import verify_answer
        from generation.generator import GeneratedAnswer

        ans = GeneratedAnswer(text="", atomic_claims=[], segments=[],
                              evidence_map={}, is_honest_refusal=False)
        verified = verify_answer(ans, llm=None)
        self.assertFalse(verified.fully_supported)
        self.assertEqual(len(verified.judgements), 0)


class TestNliFalseNegativeFallback(unittest.TestCase):
    """NLI 假阴性兜底(question&solution.md #13b)—— 极低分 + 实体重叠 → LLM 兜底。"""

    def test_entity_overlap_detection(self):
        """实体重叠检测:claim 任一中文片段(2-gram)在 evidence 里出现就算重叠。"""
        from reflection.issup import _entity_overlap
        # claim 里的"甄士隐家"和"葫芦庙失火"都在 evidence 里
        self.assertTrue(_entity_overlap(
            "甄士隐家中遭遇了葫芦庙失火",
            ["三月十五日,葫芦庙中炸供,火势蔓延,甄家被烧成瓦砾场"],
        ))
        # 完全不相关:claim 谈宋朝皇帝,evidence 谈红楼梦——无任何共同 2-gram
        self.assertFalse(_entity_overlap(
            "宋太祖建立了北宋",
            ["甄士隐资助贾雨村五十两白银和冬衣"],
        ))

    def test_low_nli_with_overlap_falls_back_to_llm(self):
        """NLI 给 0.02 + 实体重叠 → 应转 LLM 兜底。"""
        from reflection import issup
        from reflection.issup import Judgement, verify
        from reflection.tokens import IsSup

        # mock NLI 返回极低分;mock LLM 返回 FULL
        class _MockNLI:
            def predict(self, premise, hypothesis):
                return {"entailment": 0.02, "neutral": 0.5, "contradiction": 0.48}

        class _MockLLM:
            def chat(self, prompt, **kwargs):
                return "FULL"

        # patch get_nli 和 LLMClient
        orig_get_nli = issup.get_nli
        issup.get_nli = lambda: _MockNLI()
        try:
            j = verify("甄士隐家遭遇葫芦庙失火",
                       ["葫芦庙失火,甄家被烧成瓦砾场"],
                       llm=_MockLLM())
            self.assertEqual(j.token, IsSup.FULL)
            self.assertIn("假阴性兜底", j.reason)
            self.assertIn("0.02", j.reason)  # 原 NLI 分数留痕
        finally:
            issup.get_nli = orig_get_nli

    def test_low_nli_no_overlap_keeps_nli(self):
        """NLI 给 0.02 但无实体重叠 → 不兜底,保留 NLI 的 NONE。"""
        from reflection import issup
        from reflection.tokens import IsSup

        class _MockNLI:
            def predict(self, premise, hypothesis):
                return {"entailment": 0.02, "neutral": 0.5, "contradiction": 0.48}

        orig_get_nli = issup.get_nli
        issup.get_nli = lambda: _MockNLI()
        try:
            # claim 谈宋朝,evidence 谈红楼梦,完全无 2-gram 重叠
            j = issup.verify("宋太祖建立了北宋",
                             ["甄士隐资助贾雨村五十两白银"],
                             llm=None)
            # 无重叠 → 不兜底 → 保留 NLI 的 NONE
            self.assertEqual(j.token, IsSup.NONE)
            self.assertIn("NLI:", j.reason)
        finally:
            issup.get_nli = orig_get_nli

    def test_high_nli_does_not_trigger_fallback(self):
        """NLI 高分(0.9)直接返回,不触发兜底——避免无意义的 LLM 调用。"""
        from reflection import issup
        from reflection.tokens import IsSup

        class _MockNLI:
            def predict(self, premise, hypothesis):
                return {"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02}

        # 即使 LLM 不可用也无所谓——根本不该被调到
        class _RaisingLLM:
            def chat(self, *a, **k):
                raise AssertionError("不应该调 LLM")

        orig_get_nli = issup.get_nli
        issup.get_nli = lambda: _MockNLI()
        try:
            j = issup.verify("x", ["y"], llm=_RaisingLLM())
            self.assertEqual(j.token, IsSup.FULL)
        finally:
            issup.get_nli = orig_get_nli


class TestRefeedFingerprint(unittest.TestCase):
    """指纹检测(question&solution.md #12 + 同义改写绕过修复)。

    指纹的设计:用 (foci_set, seeds_set),不含 sub_query 字符串。
    这样 LLM 同义改写 sub_query("殃及"↔"波及")也能识别为同一件事。
    """

    def test_fingerprint_unaffected_by_subquery_synonym_rewrite(self):
        """两轮 sub_query 同义改写,但 foci/seeds 相同 → 应被识别为重复。"""
        from generation.refeed import _refeed_one_round
        from reflection.tokens import IsSup

        # 构造两轮 verified,每轮 LLM 提的 gap 措辞不同但 foci 相同
        rewrites = iter([
            '{"sub_query":"葫芦庙失火殃及甄家","focus_entities":["葫芦庙失火","甄士隐家"]}',
            '{"sub_query":"葫芦庙失火波及甄家","focus_entities":["甄士隐家","葫芦庙失火"]}',
        ])

        class _MockLLM:
            def chat(self, prompt, **kwargs):
                return next(rewrites)

        # mock state(只用到 evidence_entities / events)
        class _State:
            evidence_entities = {"甄士隐"}
            evidence_events = []
            visited_edges = set()
            evidence_nodes = set()
            evidence_edges = []

        # mock verified:1 个 NONE judgement
        class _Verified:
            judgements = [{
                "claim": {"text": "甄家遭遇火灾", "clean": "甄家遭遇火灾", "cited_ids": []},
                "segment": {"text": "x"},
                "token": IsSup.NONE, "score": 0.0, "reason": "test",
            }]

        # run_loop_fn 返回任意非 None 的 mock state(有合并需要的字段)
        class _NewState:
            visited_edges = set()
            evidence_nodes = set()
            evidence_edges = []
            evidence_entities = set()
            evidence_events = []

        run_loop_fn = lambda sq, sd, st: _NewState()

        state = _State()
        verified = _Verified()
        llm = _MockLLM()

        # 跑两轮,记录指纹
        _, fp1 = _refeed_one_round(verified, "q", state, run_loop_fn, llm)
        _, fp2 = _refeed_one_round(verified, "q", state, run_loop_fn, llm)

        # 关键断言:虽然 sub_query 不同,指纹应相同(因为 foci+seeds 集合相同)
        self.assertEqual(fp1, fp2,
                         f"同义改写应被识别为相同指纹\nfp1={fp1}\nfp2={fp2}")


if __name__ == "__main__":
    unittest.main()
