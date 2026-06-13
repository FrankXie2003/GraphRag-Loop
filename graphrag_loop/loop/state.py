"""循环状态对象 —— frontier / 已访问边 / 证据子图 / 置信度。

Phase 2 双层 schema 下,证据子图分类组织:
  evidence_entities:已触达的 Entity 名集合(图谱骨架)
  evidence_events:已触达的 Event 列表 [{name, content, order}](情节,带时序)
  evidence_edges:走过的边(保留完整路径,供生成时引用对齐)

这种分层组织有两个直接好处:
  1. LLM 决策(decide)时能拿到事件 content,而不是只看节点名/分数
  2. 生成阶段可以分开用:Entity 集合做引用骨架,Event 按时序拼成"情节链"
"""

from dataclasses import dataclass, field


@dataclass
class LoopState:
    frontier: list                                       # 当前层待扩展的节点
    visited_edges: set = field(default_factory=set)      # 路径记忆:已走过的边
    evidence_nodes: set = field(default_factory=set)     # 累积证据节点(平铺,兼容老代码)
    evidence_edges: list = field(default_factory=list)   # 证据子图的边
    # ---- Phase 2 双层证据 ----
    evidence_entities: set = field(default_factory=set)  # 实体名
    evidence_events: list = field(default_factory=list)  # [{name, content, order}],按加入顺序
    confidence: float = 0.0
    stop_reason: str = ""
