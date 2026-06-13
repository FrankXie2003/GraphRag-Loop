"""Self-RAG 反思 token 定义 —— 统一各反思组件的取值与数据结构。

Self-RAG 原文有四个反思 token。本系统用前三个驱动检索流程,IsUse 暂作预留:

  Retrieve (L1)  要不要检索        → 驱动:走图谱 / 直接生成   [entry/route_decision.py]
  IsRel    (S3)  候选相关吗        → 驱动:每跳剪枝             [reflection/isrel.py, bge-reranker]
  IsSup    (L3)  生成有证据支撑吗  → 驱动:定向回扩 / 输出       [reflection/issup.py, NLI/LLM]
  IsUse          答案有用吗(1-5)  → 不驱动流程,仅评质量;预留给续写场景的质量评分

为什么 IsUse 暂不用:前三个是流程控制信号(决定走不走图、剪不剪枝、回不回扩),
IsUse 只对最终答案打有用性分,不改变检索流程,且评的是通用生成质量而非图检索价值。
留到续写生成阶段作为"续写质量评分器",与 CoVe 自我核验一起用更合适。
"""

from enum import Enum


class Retrieve(str, Enum):
    """L1:是否需要检索图谱。"""
    YES = "retrieve"        # 需要走图谱
    NO = "no_retrieve"      # LLM 内部知识即可,直接生成


class IsRel(str, Enum):
    """S3:候选与 query 是否相关。实际打分用 bge-reranker 的连续分 + τ_rel 阈值;
    这里给离散语义供需要时使用。"""
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"


class IsSup(str, Enum):
    """L3:生成内容是否被证据支撑(Self-RAG 三值)。"""
    FULL = "fully_supported"
    PARTIAL = "partially_supported"
    NONE = "no_support"


class IsUse(int, Enum):
    """预留:答案有用性 1-5 分(续写质量评分用)。"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5
