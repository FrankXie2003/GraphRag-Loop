"""Agent 决策策略(Phase 0 规则模拟版)。

完整架构里这是一次 LLM 调用(loop/decision.py,Self-RAG 决策):
让 LLM 看当前证据,判断"信息够了吗"。
demo 用一个透明的规则来模拟"累积证据置信度",好处是确定性、可解释、可单测。

置信度估计思路(玩具版):
  - 本轮新证据里的最高分越高,说明越可能命中答案 → 置信度越高
  - 累加历史最高分,模拟"证据越攒越足"
  - 直接命中答案节点(可选提示)时给一个大的加成
真实系统里这步会换成模型对"问题是否已被证据回答"的判断。
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
        # answer_hint:可选,若提供则当证据子图触达该节点时置信度大幅提升。
        # 仅为让 demo 的终止行为更直观,真实系统不依赖它。
        self.answer_hint = answer_hint
        self._best = 0.0  # 历史最高单候选分

    def decide(self, round_idx, kept, evidence_nodes):
        """根据本轮保留的候选 kept 和累积证据节点,决定停or继续。"""
        round_best = max((c.score for c in kept), default=0.0)
        self._best = max(self._best, round_best)

        # 置信度:历史最高分为主,命中答案提示给加成,上限 1.0
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
