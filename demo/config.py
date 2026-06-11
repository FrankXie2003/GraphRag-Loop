"""Demo 超参集中配置(对应主 README §2.1 的子集)。

命令行参数可覆盖这里的默认值(见 run_demo.py)。
"""

from dataclasses import dataclass


@dataclass
class DemoConfig:
    D_MAX: int = 3          # 最大跳数 / 循环轮次兜底
    BEAM_WIDTH: int = 3     # 每跳保留的候选节点数(demo 用 3 便于观察截断)
    TAU_REL: float = 0.25   # 相关性剪枝阈值,score < τ_rel 的候选被砍
    THETA_STOP: float = 0.6 # 终止置信度阈值
    USE_EMBEDDING: bool = False  # 是否用句向量打分(需 sentence-transformers)


DEFAULT = DemoConfig()
