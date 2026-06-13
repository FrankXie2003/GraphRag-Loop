"""模型路由表 —— 每个环节映射到模型档位,实现成本/延迟分层(对应主 README §2.2)。

理由:高频、判断简单的环节用小模型(qwen-turbo);需要推理/质量的环节用大模型(qwen-plus);
打分类环节用本地重排/NLI,便宜可量化。改这里即可全局切换某环节用什么模型。
"""

from enum import Enum


class Tier(str, Enum):
    SMALL = "small"      # qwen-turbo:入口决策、关系剪枝
    LARGE = "large"      # qwen-plus :Agent 决策、最终生成、实体关系抽取
    RERANK = "rerank"    # bge-reranker cross-encoder:IsRel 打分
    NLI = "nli"          # 蕴含模型:IsSup 段级验证


# 环节 → 档位。键对应架构图中的步骤标签。
ROUTING = {
    "L1_route_decision":   Tier.SMALL,   # 要不要走图谱
    "L1b_soft_link":       Tier.SMALL,   # NER / 软链接辅助
    "S2_relation_prune":   Tier.SMALL,   # 关系类型剪枝
    "S3_isrel":            Tier.RERANK,  # 节点相关性打分(剪枝)
    "S5_decision":         Tier.LARGE,   # 信息够了吗
    "L3_generate":         Tier.LARGE,   # 最终答案生成
    "L3_issup":            Tier.NLI,     # 段级证据验证
    "ingest_extract":      Tier.LARGE,   # 建图:抽实体关系
    "ingest_align":        Tier.LARGE,   # 建图:实体/关系对齐、矛盾消解
}
