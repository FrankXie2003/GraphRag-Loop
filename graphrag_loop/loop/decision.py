"""S5 Agent 决策 —— Phase 1 暂用规则模拟,Phase 2 换成 LLM(Self-RAG)。

规则版置信度估计(确定性、可解释、可单测):
  - 历史最高单候选分作为基础置信度(证据越攒越足)
  - 命中 answer_hint 时加成(仅让终止行为更直观,真实系统不依赖)
终止:conf ≥ θ_stop 或 r ≥ D_max 或本轮无候选存活。

逻辑契约与 demo/agent/policy.py 一致(Phase 0 已验证)。
Phase 2 的 LLM 版会换掉 decide() 内部:把"信息够了吗"交给模型判断,接口不变。
"""

from dataclasses import dataclass


@dataclass
class Decision:
    stop: bool
    confidence: float
    reason: str


class RuleBasedPolicy:
    def __init__(self, theta_stop, d_max, answer_hint=None):
        self.theta_stop = theta_stop
        self.d_max = d_max
        self.answer_hint = answer_hint
        self._best = 0.0

    def decide(self, round_idx, kept, evidence_nodes):
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
