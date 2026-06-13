"""验证三个 Self-RAG 反思组件:Retrieve / IsRel / IsSup。

用法:python verify_reflection.py
前提:已建过双层图谱(ingest_graph_md_v2.py)。

每个组件用对照实验验证:正例应触发"yes/relevant/full",反例应触发相反。
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_this = Path(__file__).resolve().parent
sys.path.insert(0, str(_this))

from entry.route_decision import need_retrieve
from reflection.isrel import get_scorer
from reflection.issup import verify
from reflection.tokens import Retrieve


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main():
    # ===== L1 Retrieve =====
    section("L1 Retrieve:判断 query 是否需要查图谱")
    cases = [
        ("甄士隐送给贾雨村什么东西", Retrieve.YES, "图里有的事实问题"),
        ("英莲是被谁拐走的",         Retrieve.YES, "图里有的事实问题"),
        ("1+1等于几",               Retrieve.NO,  "通用常识,不需检索"),
        ("帮我写一首关于秋天的诗",   Retrieve.NO,  "纯创作,与本图无关"),
        ("你好",                    Retrieve.NO,  "寒暄"),
    ]
    for q, expected, hint in cases:
        got = need_retrieve(q)
        ok = "✓" if got == expected else "✗"
        print(f"  {ok} [{got.value:12s}] '{q}'   ({hint})")

    # ===== S3 IsRel(bge-reranker)=====
    section("S3 IsRel:相关性打分(已在 Phase 1 验证,这里只确认调用通)")
    ranker = get_scorer()
    q = "甄士隐送给贾雨村什么东西"
    cands = [
        "赠银给 贾雨村",
        "女儿 英莲",
        "诗词 月亮",
    ]
    for c, s in zip(cands, ranker.score_batch(q, cands)):
        print(f"  [{s:.2f}] '{c}'")

    # ===== L3 IsSup:段级证据验证 =====
    section("L3 IsSup:段级证据验证(NLI 不可用则 LLM 兜底)")
    # 硬编码一段从图谱抽出的真实事件 content,作为"证据子图"
    # (不依赖 Neo4j,即使图库挂了也能验证 IsSup 本身的逻辑)
    evidence = [
        "中秋夜，贾雨村对月吟诗抒志，甄士隐听后盛赞其才，当场承诺资助。"
        "他命人封五十两白银及两套冬衣赠予贾雨村，并定十九日为黄道吉日，"
        "劝其速赴神京参加次年春闱。贾雨村收下银物，仅略表谢意，仍谈笑如常，直至三更方散。"
    ]
    print(f"证据(模拟从 Event 节点 content 取得,前 80 字):")
    print(f"  {evidence[0][:80]}...")

    cases = [
        ("甄士隐资助贾雨村赴京赶考",         "应 FULL(原文事实)"),
        ("甄士隐赠了贾雨村五十两白银和冬衣", "应 FULL(原文细节)"),
        ("甄士隐拒绝了贾雨村的请求",         "应 NONE(与原文相反)"),
        ("贾雨村写了一本书",                 "应 NONE(原文未提)"),
    ]
    for claim, expected in cases:
        j = verify(claim, evidence)
        print(f"  [{j.token.value:24s}] (score={j.score:.2f}) '{claim}'  — {expected}")
        print(f"     {j.reason}")


if __name__ == "__main__":
    main()
