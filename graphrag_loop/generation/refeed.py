"""定向回扩(refeed)—— Self-RAG 闭环的最后一块。

流程(对应主 README §2 架构图 L3 那个"unsupported → 回到循环"的箭头):

  生成答案 → IsSup 逐条 atomic_claim 验证(原子断言级)
       ├─ 全部 FULL → 输出
       └─ 有 PARTIAL/NONE 断言 → 提缺口 → 定向回扩
                                       ├─ 已达 V_MAX → 接受当前答案
                                       └─ 未达    → 缺口作为子 query 跑 loop,
                                                   继承已有证据节点,补完后重新生成

为什么用原子断言级(question&solution.md #13):NLI 模型对"长 hypothesis 含多个独立事实"
力不从心,"取最大值"会假阳性。让 generator 直接拆原子断言,逐条 NLI,从源头解决。
缺口提取也精准了:每个 non-FULL claim 本身就是单一断言,不需要再拆。

继承已有证据节点是关键:不要"重启 loop"(白白丢掉前面的成果),而是让回扩的 loop
从原有证据集合出发,只补不足的部分。
"""

from dataclasses import dataclass, field
import json
import re

from config.settings import PARAMS
from models.llm import LLMClient
from reflection.tokens import IsSup
from reflection.issup import verify
from generation.citation import evidences_for_segment


_GAP_SYSTEM = "你是缺口提取助手:给定问题、未支撑的原子断言,提炼出'还需要查什么'。"

_GAP_PROMPT = """原始问题:{query}

下面这条原子断言被标记为"未支撑"或"部分支撑",说明现有证据不足以证明它:
{claim}

判定理由:{judgement}

请提炼一个**具体的检索子问题**,描述还需要从知识图谱里查什么才能支撑它。
要求:子问题要具体可查,围绕涉及到的人物或事件。只输出 JSON,不要解释:
{{"sub_query": "...", "focus_entities": ["相关人物或事件名"]}}
"""


@dataclass
class VerifiedAnswer:
    """带 IsSup 验证结果的答案。

    judgements 字段名沿用旧名,但内容是**原子断言级**:每个 element 是
        {claim: {text, clean, cited_ids}, token, score, reason}
    "claim" 替代了旧的 "segment",但为兼容 recorder/旧 tests,**两个字段都填**。
    """
    answer: object             # GeneratedAnswer
    judgements: list = field(default_factory=list)
    refeed_rounds: int = 0
    fully_supported: bool = False  # 所有断言都 FULL


def _judgement_for_claim(claim, evidence_map, llm):
    """对单条原子断言做 IsSup,返回 judgement dict。"""
    evs = evidences_for_segment(claim, evidence_map)  # claim 与 segment 字段同构
    if not evs:
        return {
            "claim": claim, "segment": claim,  # 双写,兼容 recorder
            "token": IsSup.NONE, "score": 0.0,
            "reason": "断言未引用任何证据",
        }
    j = verify(claim["clean"], evs, llm=llm)
    return {
        "claim": claim, "segment": claim,
        "token": j.token, "score": j.score, "reason": j.reason,
    }


def verify_answer(answer, llm=None):
    """对答案逐条 atomic_claim 做 IsSup,返回 VerifiedAnswer。

    诚实拒答(answer.is_honest_refusal=True)直接 FULL 短路,不调 LLM/NLI。
    无 atomic_claims 时(理论上 generator 的 fallback 会填上,这里再兜一次)
    退回到按 segments 验证(老路径,等价 Phase 2 之前的行为)。

    聚合:全部 claims 都 FULL → fully_supported=True;否则 False(任何一条非 FULL 都触发回扩)。
    这种"全或无"的严格策略消除了"取最大值"假阳性,代价是更多 query 会触发回扩——
    但回扩本身是治本的,不该怕它跑。
    """
    if getattr(answer, "is_honest_refusal", False):
        ph = {"text": answer.text.strip(), "clean": answer.text.strip(),
              "cited_ids": []}
        return VerifiedAnswer(
            answer=answer,
            judgements=[{"claim": ph, "segment": ph, "token": IsSup.FULL,
                         "score": 1.0, "reason": "诚实拒答(IsSup 短路)"}],
            fully_supported=True,
        )

    items = answer.atomic_claims or answer.segments or []

    judgements = []
    all_full = True
    for claim in items:
        j = _judgement_for_claim(claim, answer.evidence_map, llm)
        judgements.append(j)
        if j["token"] != IsSup.FULL:
            all_full = False

    return VerifiedAnswer(answer=answer, judgements=judgements,
                          fully_supported=all_full and bool(items))


def _extract_gap(query, judgement, llm):
    """让 LLM 把一条不达支撑的原子断言转成具体子问题 + 关注实体。"""
    claim = judgement.get("claim") or judgement.get("segment") or {}
    prompt = _GAP_PROMPT.format(
        query=query,
        claim=claim.get("text", ""),
        judgement=f"{judgement['token'].value} (score={judgement['score']:.2f}); {judgement['reason']}",
    )
    raw = llm.chat(prompt, stage="L3_generate", system=_GAP_SYSTEM)
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        obj = json.loads(raw[s:e + 1])
    except json.JSONDecodeError:
        return None
    sq = (obj.get("sub_query") or "").strip()
    if not sq:
        return None
    foci = [str(x).strip() for x in (obj.get("focus_entities") or []) if str(x).strip()]
    return {"sub_query": sq, "focus_entities": foci}


def _refeed_one_round(verified, query, state, run_loop_fn, llm):
    """一轮定向回扩:对所有非 FULL 原子断言提缺口,合成一个聚合子 query 跑 loop,
    然后把新 loop 的证据**合并**到原 state(继承,不重启)。

    返回:(merged_state, fingerprint)
      - merged_state 为 None:无可提取的缺口,应中止
      - fingerprint 是 (sub_query, sorted(foci)+sorted(seeds)) 的元组,
        用于上层检测重复回扩
    """
    gaps = []
    foci = []
    for j in verified.judgements:
        if j["token"] == IsSup.FULL:
            continue
        gap = _extract_gap(query, j, llm)
        if gap:
            gaps.append(gap["sub_query"])
            foci.extend(gap["focus_entities"])
    if not gaps:
        return None, None

    sub_query = " ".join(f"({g})" for g in gaps).strip()
    print(f"  [refeed] 子查询: {sub_query}")
    if foci:
        print(f"  [refeed] 关注实体: {foci}")

    # 种子策略:已有证据节点 ∪ LLM 给出的关注实体。
    seeds = list(state.evidence_entities) + foci
    if not seeds:
        seeds = [ev["name"] for ev in state.evidence_events]
    seeds = list(dict.fromkeys(seeds))

    # 指纹:**只用关注实体集合 + 种子集合**,不含 sub_query 本身。
    # 理由:LLM 提缺口时会同义改写 sub_query("殃及"↔"波及","被烧毁"↔"被烧成火海"),
    # 字符串等比对会被绕过(实测 #13 升级后的 case)。但 foci+seeds 表达的是
    # "想去图里查谁",同义改写不会改变这个集合 → 集合相等就是同一件事,继续是徒劳。
    fingerprint = (tuple(sorted(set(foci))),
                   tuple(sorted(set(seeds))))

    new_state = run_loop_fn(sub_query, seeds, state)
    if new_state is None:
        return None, fingerprint

    # ---- 合并:把新 state 的证据并入原 state(继承,而非替换)----
    state.visited_edges |= new_state.visited_edges
    state.evidence_nodes |= new_state.evidence_nodes
    seen_edges = {c.edge_key for c in state.evidence_edges}
    for c in new_state.evidence_edges:
        if c.edge_key not in seen_edges:
            state.evidence_edges.append(c)
            seen_edges.add(c.edge_key)
    state.evidence_entities |= new_state.evidence_entities
    have_event_names = {ev["name"] for ev in state.evidence_events}
    for ev in new_state.evidence_events:
        if ev["name"] not in have_event_names:
            state.evidence_events.append(ev)
            have_event_names.add(ev["name"])

    return state, fingerprint


def refeed_until_supported(query, state, generated, run_loop_fn,
                           llm=None, max_rounds=None,
                           on_each_verification=None):
    """主入口:验证 → 不达支撑则回扩 → 重新生成 → 再验证…

    on_each_verification(round_idx, verified):每次验证完成后回调,供调用方记录中间状态。
    """
    llm = llm or LLMClient()
    max_rounds = max_rounds if max_rounds is not None else PARAMS.V_MAX

    verified = verify_answer(generated, llm=llm)
    if on_each_verification:
        on_each_verification(0, verified)

    rounds = 0
    last_fingerprint = None
    while not verified.fully_supported and rounds < max_rounds:
        non_full = [j for j in verified.judgements if j["token"] != IsSup.FULL]
        print(f"\n[refeed] 第 {rounds+1} 轮:发现 {len(non_full)} 条断言未达支撑,定向回扩")

        merged_state, fingerprint = _refeed_one_round(
            verified, query, state, run_loop_fn, llm)
        if merged_state is None:
            print("[refeed] 无法提取有效缺口,中止回扩")
            break

        if last_fingerprint is not None and fingerprint == last_fingerprint:
            print("[refeed] 检测到重复回扩(子查询/种子与上轮完全相同),中止")
            break
        last_fingerprint = fingerprint

        rounds += 1
        from generation.generator import generate
        generated = generate(query, merged_state, llm=llm)
        verified = verify_answer(generated, llm=llm)
        verified.refeed_rounds = rounds
        if on_each_verification:
            on_each_verification(rounds, verified)

    if not verified.fully_supported and rounds >= max_rounds:
        print(f"[refeed] 已达 V_MAX={max_rounds},接受当前答案(部分断言可能未支撑)")

    verified.refeed_rounds = rounds
    return verified
