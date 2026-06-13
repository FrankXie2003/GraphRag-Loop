"""定向回扩(refeed)—— Self-RAG 闭环的最后一块。

流程(对应主 README §2 架构图 L3 那个"unsupported → 回到循环"的箭头):

  生成答案 → IsSup 逐段验证
       ├─ 全部 FULL → 输出
       └─ 有 PARTIAL/NONE 段 → 提缺口 → 定向回扩
                                    ├─ 已达 V_MAX → 接受当前答案(标注未支撑段)
                                    └─ 未达    → 缺口作为子 query 跑 loop,
                                                继承已有证据节点,补完后重新生成

继承已有证据节点是关键:不要"重启 loop"(白白丢掉前面的成果),而是让回扩的 loop
从原有证据集合出发,只补不足的部分。这跟 Self-RAG 原文的"定向"语义一致。

回扩用 LLM 提炼"信息缺口"——比直接拿原段落当子 query 更聚焦,因为段落本身可能含
已有证据的部分,只有"未支撑的那部分"才需要去查。
"""

from dataclasses import dataclass, field
import json
import re

from config.settings import PARAMS
from models.llm import LLMClient
from reflection.tokens import IsSup
from reflection.issup import verify
from generation.citation import evidences_for_segment


_GAP_SYSTEM = "你是缺口提取助手:给定问题、段落答案、未达支撑的部分,提炼出'还需要查什么'。"

_GAP_PROMPT = """原始问题:{query}

下面这段答案中标记为"未支撑"或"部分支撑",说明现有证据不足以证明它:
{segment}

判定理由:{judgement}

请提炼一个**具体的检索子问题**,描述还需要从知识图谱里查什么才能支撑这段。
要求:子问题要具体可查,围绕涉及到的人物或事件。只输出 JSON,不要解释:
{{"sub_query": "...", "focus_entities": ["相关人物或事件名"]}}
"""


@dataclass
class VerifiedAnswer:
    """带 IsSup 验证结果的答案。"""
    answer: object             # GeneratedAnswer
    judgements: list = field(default_factory=list)   # [{segment, token, score, reason}]
    refeed_rounds: int = 0
    fully_supported: bool = False  # 所有段都 FULL


def verify_answer(answer, llm=None):
    """对答案逐段 IsSup,返回 VerifiedAnswer。无引用的段视作 NONE(触发回扩)。

    诚实拒答(answer.is_honest_refusal=True)直接 FULL 短路,不调 LLM/NLI——
    "原文未提"是对证据状态的诚实陈述,不是事实捏造,无须验证也不应触发回扩。
    判定逻辑集中在 generator._is_honest_refusal,verify_answer 只读字段。
    """
    if getattr(answer, "is_honest_refusal", False):
        seg = {"text": answer.text.strip(), "clean": answer.text.strip(),
               "cited_ids": []}
        return VerifiedAnswer(
            answer=answer,
            judgements=[{"segment": seg, "token": IsSup.FULL,
                         "score": 1.0, "reason": "诚实拒答(IsSup 短路)"}],
            fully_supported=True,
        )

    judgements = []
    all_full = True
    for seg in answer.segments:
        evs = evidences_for_segment(seg, answer.evidence_map)
        if not evs:
            judgements.append({
                "segment": seg, "token": IsSup.NONE,
                "score": 0.0, "reason": "段落未引用任何证据",
            })
            all_full = False
            continue
        j = verify(seg["clean"], evs, llm=llm)
        judgements.append({
            "segment": seg, "token": j.token,
            "score": j.score, "reason": j.reason,
        })
        if j.token != IsSup.FULL:
            all_full = False
    return VerifiedAnswer(answer=answer, judgements=judgements,
                          fully_supported=all_full)


def _extract_gap(query, judgement, llm):
    """让 LLM 把一段不达支撑的内容转成具体子问题 + 关注实体。"""
    prompt = _GAP_PROMPT.format(
        query=query,
        segment=judgement["segment"]["text"],
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
    """一轮定向回扩:对所有非 FULL 段提缺口,合成一个聚合子 query 跑一次 loop,
    然后把新 loop 的证据**合并**到原 state(继承,不重启)。

    返回:(merged_state, fingerprint)
      - merged_state 为 None:无可提取的缺口,应中止
      - fingerprint 是 (sub_query, sorted(foci)+sorted(seeds)) 的元组,
        用于上层检测重复回扩(若两轮指纹相同则 LLM 没有新缺口可提,继续是徒劳)
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
    # 关注实体不存在也无所谓——run_loop 拿到不存在节点 get_neighbors 返回空。
    seeds = list(state.evidence_entities) + foci
    if not seeds:
        seeds = [ev["name"] for ev in state.evidence_events]
    seeds = list(dict.fromkeys(seeds))

    # 指纹:子查询 + 排序去重的关注实体 + 排序去重的种子
    fingerprint = (sub_query,
                   tuple(sorted(set(foci))),
                   tuple(sorted(set(seeds))))

    # 调用方注入的 run_loop_fn 会跑出一个新 state(独立的 LoopState 实例)
    new_state = run_loop_fn(sub_query, seeds, state)
    if new_state is None:
        return None, fingerprint

    # ---- 合并:把新 state 的证据并入原 state(继承,而非替换)----
    state.visited_edges |= new_state.visited_edges
    state.evidence_nodes |= new_state.evidence_nodes
    # edges 保序追加,且去重(按 edge_key)
    seen_edges = {c.edge_key for c in state.evidence_edges}
    for c in new_state.evidence_edges:
        if c.edge_key not in seen_edges:
            state.evidence_edges.append(c)
            seen_edges.add(c.edge_key)
    state.evidence_entities |= new_state.evidence_entities
    # events 按名字去重
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

    run_loop_fn 是调用方传入的回调:run_loop_fn(sub_query, seeds, state) → state'
    (内部应复用同一份 state,不重置 visited_edges / evidence_*)

    on_each_verification(round_idx, verified):每次验证完成后回调,供调用方记录中间状态。
        round_idx=0 表示初次验证,1.. 是第几轮回扩后的验证。
        run_phase2 用这个回调把每轮的 IsSup 结果都记进 RunRecorder。

    返回最终的 VerifiedAnswer(可能仍非 fully_supported,但已尽力 V_MAX 轮)。
    """
    llm = llm or LLMClient()
    max_rounds = max_rounds if max_rounds is not None else PARAMS.V_MAX

    verified = verify_answer(generated, llm=llm)
    if on_each_verification:
        on_each_verification(0, verified)

    rounds = 0
    last_fingerprint = None  # 上一轮指纹,用于检测重复回扩
    while not verified.fully_supported and rounds < max_rounds:
        non_full = [j for j in verified.judgements if j["token"] != IsSup.FULL]
        print(f"\n[refeed] 第 {rounds+1} 轮:发现 {len(non_full)} 段未达支撑,定向回扩")

        merged_state, fingerprint = _refeed_one_round(
            verified, query, state, run_loop_fn, llm)
        if merged_state is None:
            print("[refeed] 无法提取有效缺口,中止回扩")
            break

        # 重复回扩检测:指纹一样说明 LLM 提的子查询 + 种子和上轮完全相同,
        # 继续是徒劳(图里就是没这条信息)。早退,不浪费 LLM 调用。
        if last_fingerprint is not None and fingerprint == last_fingerprint:
            print("[refeed] 检测到重复回扩(子查询/种子与上轮完全相同),中止")
            break
        last_fingerprint = fingerprint

        rounds += 1
        # 重新生成(用合并后的 state)
        from generation.generator import generate
        generated = generate(query, merged_state, llm=llm)
        verified = verify_answer(generated, llm=llm)
        verified.refeed_rounds = rounds
        if on_each_verification:
            on_each_verification(rounds, verified)

    if not verified.fully_supported and rounds >= max_rounds:
        print(f"[refeed] 已达 V_MAX={max_rounds},接受当前答案(部分段可能未支撑)")

    verified.refeed_rounds = rounds
    return verified
