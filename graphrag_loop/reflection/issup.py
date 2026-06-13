"""L3 IsSup —— 段级证据验证(Self-RAG IsSup token)。

判断"生成的某段断言"是否被"证据子图里的内容"支撑。
输出三值:fully_supported / partially_supported / no_support。

两条路径,优先 NLI(便宜可量化),NLI 不可用则 LLM 兜底:

  (1) NLI:把每条证据当 premise,断言当 hypothesis,取最大 entailment 概率。
       p_e ≥ τ_sup        → FULL
       0.3 ≤ p_e < τ_sup  → PARTIAL
       p_e < 0.3          → NONE
  (2) LLM:三选一直接判,prompt 喂证据 + 断言。

使用方式:把生成的答案按句/段切,对每段调 verify(claim, evidences),
有 PARTIAL/NONE 段触发定向回扩(generation/refeed.py)。
"""

from dataclasses import dataclass

from config.settings import PARAMS
from models.nli import get_nli
from models.llm import LLMClient
from reflection.tokens import IsSup


@dataclass
class Judgement:
    token: IsSup        # 三值
    score: float        # NLI:entailment 概率 / LLM:估计 0/0.5/1
    reason: str         # 用哪条路径、命中哪条证据等


def _from_score(p_e, tau_sup):
    if p_e >= tau_sup:
        return IsSup.FULL
    if p_e >= 0.3:
        return IsSup.PARTIAL
    return IsSup.NONE


def verify_with_nli(claim, evidences, tau_sup=None):
    """NLI 路径:对每条证据计算 entailment(claim),取最大值。

    evidences: List[str](证据文本片段,如 Event content / passage)。
    """
    tau_sup = tau_sup if tau_sup is not None else PARAMS.TAU_SUP
    nli = get_nli()
    if nli is None or not evidences:
        return None
    best_p = 0.0
    best_ev = None
    for ev in evidences:
        probs = nli.predict(ev, claim)
        if probs["entailment"] > best_p:
            best_p = probs["entailment"]
            best_ev = ev
    return Judgement(
        token=_from_score(best_p, tau_sup),
        score=best_p,
        reason=f"NLI: best entailment={best_p:.2f} on evidence: {(best_ev or '')[:40]}...",
    )


def verify_with_llm(claim, evidences, llm=None, tau_sup=None):
    """LLM 兜底:把证据拼起来让 LLM 三选一。"""
    tau_sup = tau_sup if tau_sup is not None else PARAMS.TAU_SUP
    if not evidences:
        return Judgement(IsSup.NONE, 0.0, "无证据")
    llm = llm or LLMClient()
    ev_text = "\n".join(f"- {e}" for e in evidences)
    prompt = f"""判断断言是否被证据支撑(三选一)。

证据:
{ev_text}

断言:{claim}

只回答 FULL(证据完全支撑)/ PARTIAL(部分支撑或推断)/ NONE(无支撑或矛盾),不要解释。"""
    out = llm.chat(prompt, stage="L3_issup").strip().upper()
    if out.startswith("FULL"):
        return Judgement(IsSup.FULL, 1.0, "LLM: FULL")
    if out.startswith("PART"):
        return Judgement(IsSup.PARTIAL, 0.5, "LLM: PARTIAL")
    return Judgement(IsSup.NONE, 0.0, "LLM: NONE")


def verify(claim, evidences, llm=None, tau_sup=None):
    """统一入口:NLI 优先,不可用则 LLM 兜底。"""
    j = verify_with_nli(claim, evidences, tau_sup=tau_sup)
    if j is not None:
        return j
    return verify_with_llm(claim, evidences, llm=llm, tau_sup=tau_sup)
