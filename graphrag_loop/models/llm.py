"""LLM 统一接口 —— 封装 DashScope(通义千问),按 model_routing 选大/小模型。

所有需要 LLM 的环节(抽取/入口决策/Agent决策/生成)都通过这里调用,
不直接碰 dashscope SDK,便于将来换厂商或加缓存/重试。
"""

from config.connections import DASHSCOPE
from config.model_routing import Tier, ROUTING


class LLMClient:
    def __init__(self):
        if not DASHSCOPE.api_key:
            raise RuntimeError("未配置 DASHSCOPE_API_KEY,请在 .env 中填入")
        import dashscope
        dashscope.api_key = DASHSCOPE.api_key
        self._dashscope = dashscope

    def _model_for(self, tier: Tier) -> str:
        return DASHSCOPE.model if tier == Tier.LARGE else DASHSCOPE.model_small

    def chat(self, prompt, *, stage=None, tier=None, system=None, temperature=0.0):
        """调用对话模型。

        stage:架构步骤名(如 'S5_decision'),用于从 ROUTING 自动选档位;
        tier :显式指定档位,优先级高于 stage。
        """
        if tier is None:
            tier = ROUTING.get(stage, Tier.LARGE)
        model = self._model_for(tier)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._dashscope.Generation.call(
            model=model,
            messages=messages,
            result_format="message",
            temperature=temperature,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope 调用失败 [{resp.status_code}]: {resp.message}")
        return resp.output.choices[0].message.content
