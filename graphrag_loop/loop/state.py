"""循环状态对象 —— frontier / 已访问边 / 证据子图 / 置信度。

逻辑契约与 demo/agent/loop.py 的 LoopState 一致(Phase 0 已验证),
这里是 graphrag_loop 自包含的版本,不依赖 demo 包。
"""

from dataclasses import dataclass, field


@dataclass
class LoopState:
    frontier: list                                      # 当前层待扩展的节点
    visited_edges: set = field(default_factory=set)     # 路径记忆:已走过的边
    evidence_nodes: set = field(default_factory=set)    # 累积证据节点
    evidence_edges: list = field(default_factory=list)  # 证据子图的边(含来源)
    confidence: float = 0.0
    stop_reason: str = ""
