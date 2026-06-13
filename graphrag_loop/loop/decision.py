"""S5 Agent 决策 —— 规则版(Phase 1)+ LLM 版(Phase 2 Self-RAG)。

两个策略对外接口一致:decide(round, kept, evidence_nodes, *, query=None, state=None) -> Decision

  RuleBasedPolicy(规则版,确定性、便宜):
    历史最高分作为置信度;命中 answer_hint 加成;θ_stop / D_max / 死胡同三约束终止。
    适合开发期、单测、超参调优。

  LLMDecisionPolicy(LLM 版,Phase 2):
    每轮把"已积累的证据(实体集合 + 事件 content)"喂给 LLM,问"信息够不够答 query"。
    LLM 输出 stop/continue + 自评置信度 + 理由。
    适合真实场景:能利用 Event 节点 content 做语义判断,而不是只看分数。
    成本:每跳一次 LLM 调用,可走小模型(stage='S5_decision' → SMALL via routing)。

控制流契约:run_loop 调 decide() 不区分两种策略,通过依赖注入切换。
"""

import json
import re
from dataclasses import dataclass


@dataclass
class Decision:
    stop: bool
    confidence: float
    reason: str


# =====================================================================
# 规则版(Phase 1 沿用)
# =====================================================================

class RuleBasedPolicy:
    def __init__(self, theta_stop, d_max, answer_hint=None):
        self.theta_stop = theta_stop
        self.d_max = d_max
        self.answer_hint = answer_hint
        self._best = 0.0

    def decide(self, round_idx, kept, evidence_nodes, *, query=None, state=None):
        round_best = max((c.score for c in kept), default=0.0)
        self._best = max(self._best, round_best)

        conf = self._best
        if self.answer_hint and self.answer_hint in evidence_nodes:
            conf = min(1.0, conf + 0.4)

        if conf >= self.theta_stop:
            return Decision(True, conf, f"置信度 {conf:.2f} ≥ θ_stop {self.theta_stop}")
        if round_idx >= self.d_max:
            return Decision(True, conf, f"达到 D_max={self.d_max}(兜底终止)")
        if not kept:
            return Decision(True, conf, "本轮无候选存活,无法继续扩展")
        return Decision(False, conf, f"置信度 {conf:.2f} < θ_stop,继续扩展")


# =====================================================================
# LLM 版(Phase 2 Self-RAG)
# =====================================================================

_SYSTEM = ("你是检索循环的决策器:看当前已收集的证据,判断是否足以回答用户问题。"
           "宁可早停一轮,也不要在已经够答时继续扩展浪费成本。")

_PROMPT_TMPL = """用户问题:{query}

【已收集的证据 —— 第 {round} 轮后】

参与的实体({n_ent} 个):{entities}

相关事件({n_ev} 个,带情节):
{events}

请判断证据是否已经足以回答问题,只输出 JSON,不要解释:
{{"stop": true/false, "confidence": 0.0-1.0, "reason": "一句话理由"}}

判据:
- 若证据里已经能直接读到问题的答案 → stop=true,confidence ≥ 0.7
- 若有相关线索但答案还不完整 → stop=false 继续扩展,confidence 取你判断的支撑度
- 若证据完全无关 → stop=false,confidence 较低
"""


def _format_events(events, max_chars=120):
    if not events:
        return "  (暂无)"
    lines = []
    for ev in events[:8]:  # 限制条数,prompt 别太长
        c = (ev.get("content") or "")[:max_chars]
        lines.append(f"  - 【{ev['name']}】{c}")
    return "\n".join(lines)


def _parse_decision_json(raw):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(raw[s:e + 1])
    except json.JSONDecodeError:
        return None


class LLMDecisionPolicy:
    """Self-RAG 风格的 LLM 决策器。

    依赖 controller 在调 decide() 时传入 query 和 state(state 含 evidence_entities
    与 evidence_events)。规则版的 D_max / 死胡同兜底仍保留(防 LLM 不停)。
    """

    def __init__(self, theta_stop, d_max, llm=None):
        self.theta_stop = theta_stop
        self.d_max = d_max
        from models.llm import LLMClient
        self._llm = llm or LLMClient()

    def decide(self, round_idx, kept, evidence_nodes, *, query=None, state=None):
        # 兜底硬约束:不依赖 LLM
        if round_idx >= self.d_max:
            return Decision(True, 0.0, f"达到 D_max={self.d_max}(兜底终止)")
        if not kept and round_idx > 1:
            return Decision(True, 0.0, "本轮无候选存活,无法继续扩展")
        if state is None or query is None:
            # 没传上下文 → 退化为"继续",由 D_max 兜底
            return Decision(False, 0.0, "缺少 query/state 上下文,继续扩展")

        prompt = _PROMPT_TMPL.format(
            query=query,
            round=round_idx,
            n_ent=len(state.evidence_entities),
            n_ev=len(state.evidence_events),
            entities=", ".join(sorted(state.evidence_entities)) or "(暂无)",
            events=_format_events(state.evidence_events),
        )
        try:
            raw = self._llm.chat(prompt, stage="S5_decision", system=_SYSTEM)
        except Exception as e:
            return Decision(False, 0.0, f"LLM 决策失败({e}),继续扩展")
        obj = _parse_decision_json(raw)
        if not obj:
            return Decision(False, 0.0, "LLM 输出无法解析,继续扩展")

        stop = bool(obj.get("stop"))
        conf = float(obj.get("confidence") or 0.0)
        reason = str(obj.get("reason") or "")[:200]

        # θ_stop 提供"模型认为该停 + 置信度足够"的双保险
        if stop and conf >= self.theta_stop:
            return Decision(True, conf, f"LLM:停止(conf={conf:.2f})— {reason}")
        if stop and conf < self.theta_stop:
            # 模型说停但置信度不足:再给一轮机会,除非到 D_max
            return Decision(False, conf,
                            f"LLM:想停但 conf={conf:.2f} < θ_stop,再扩一轮 — {reason}")
        return Decision(False, conf, f"LLM:继续(conf={conf:.2f})— {reason}")
