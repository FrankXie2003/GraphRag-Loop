"""L3 IsSup —— 段级证据验证(Self-RAG IsSup token)。

判断"生成的某段断言"是否被"证据子图里的内容"支撑。
输出三值:fully_supported / partially_supported / no_support。

两条路径,优先 NLI(便宜可量化),NLI 不可用或可能假阴性时 LLM 兜底:

  (1) NLI:把每条证据当 premise,断言当 hypothesis,取最大 entailment 概率。
       p_e ≥ τ_sup        → FULL
       0.3 ≤ p_e < τ_sup  → PARTIAL
       p_e < 0.3          → NONE
  (2) LLM 兜底:三选一直接判,prompt 喂证据 + 断言。

NLI 假阴性兜底(question&solution.md #13b):
  现象:NLI 对**间接论元**处理弱(如"葫芦庙失火殃及甄家"=="甄家遭遇葫芦庙失火",
       NLI 可能给 0.02)。证据里关键实体明明都在。
  策略:NLI 极低分(< 0.10)但 claim 的关键名词在 evidence 里出现时,转 LLM 兜底判断。
       这是保守策略——宁可多触发一次 LLM,也别漏假阴性。

使用方式:把生成的答案按句/段切,对每段调 verify(claim, evidences),
有 PARTIAL/NONE 段触发定向回扩(generation/refeed.py)。
"""

import re
from dataclasses import dataclass

from config.settings import PARAMS
from models.nli import get_nli
from models.llm import LLMClient
from reflection.tokens import IsSup


# NLI 假阴性兜底阈值:p_e 低于这个值时,若实体重叠则转 LLM
_NLI_FALSE_NEG_THRESHOLD = 0.10
# 关键名词最小长度:至少 2 个中文字符(单字噪声多)
_MIN_NOUN_LEN = 2


@dataclass
class Judgement:
    token: IsSup        # 三值
    score: float        # NLI:entailment 概率 / LLM:估计 0/0.5/1
    reason: str         # 用哪路径、命中哪条证据等


def _from_score(p_e, tau_sup):
    if p_e >= tau_sup:
        return IsSup.FULL
    if p_e >= 0.3:
        return IsSup.PARTIAL
    return IsSup.NONE


def _key_nouns(text):
    """从中文文本里粗抽连续中文片段(长度 >= _MIN_NOUN_LEN)作为"关键名词候选"。

    粗粒度方案:用正则切非中文字符,留下连续中文段;不做 NER(避免依赖)。
    例:"甄士隐家中遭遇了葫芦庙失火" → ['甄士隐家中遭遇了葫芦庙失火'](整段)
    实践中评估"实体重叠"时,我们检查 claim 的任一连续中文片段
    是否作为子串出现在 evidence 里,所以单串足够。
    """
    return [s for s in re.split(r"[^一-鿿]+", text) if len(s) >= _MIN_NOUN_LEN]


def _entity_overlap(claim, evidences):
    """claim 的任一连续中文片段(长度 >= 2)是否作为子串出现在 evidence 里。

    用更严格的"子串中再切 N-gram"——避免整串"甄士隐家中遭遇了葫芦庙失火"
    在 evidence 里找不到完全相同子串而漏判。我们对 claim 的每段中文做
    N=2 滑窗(2-gram),evidence 里只要命中其一就算重叠。
    """
    ev_blob = "".join(evidences)
    for seg in _key_nouns(claim):
        # 长度 < 4 的整段直接当一个 token
        if len(seg) < 4:
            if seg in ev_blob:
                return True
            continue
        # 长串切 2-gram 滑窗
        for i in range(len(seg) - 1):
            bigram = seg[i:i + 2]
            if bigram in ev_blob:
                return True
    return False


def verify_with_nli(claim, evidences, tau_sup=None):
    """NLI 路径:对每条证据计算 entailment(claim),取最大值。

    evidences: List[str](证据文本片段,如 Event content / passage)。
    返回 None 表示 NLI 不可用 / 调用方应转 LLM。
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
    """统一入口:NLI 优先;NLI 不可用 / NLI 极低分但实体重叠 → LLM 兜底。

    "NLI 极低分但实体重叠"是为了挡住 NLI 对**间接论元**的假阴性
    (如 claim "甄家遭遇葫芦庙失火" + evidence "葫芦庙起火,甄家被烧",
    NLI 可能给 0.02 但 LLM 能识别为 FULL)。
    """
    j = verify_with_nli(claim, evidences, tau_sup=tau_sup)
    if j is None:
        return verify_with_llm(claim, evidences, llm=llm, tau_sup=tau_sup)

    # NLI 假阴性兜底:极低分 + 实体重叠 → 转 LLM 二次判断
    if j.score < _NLI_FALSE_NEG_THRESHOLD and _entity_overlap(claim, evidences):
        llm_j = verify_with_llm(claim, evidences, llm=llm, tau_sup=tau_sup)
        # 拼上 NLI 的原始分数,便于事后追责
        llm_j.reason = f"{llm_j.reason} (NLI 假阴性兜底,原 NLI score={j.score:.2f})"
        return llm_j

    return j
