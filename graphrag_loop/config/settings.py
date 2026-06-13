"""全局超参集中(对应主 README §2.1)—— 单一事实来源,各层 import 这里。

Phase 0 的 demo 用 demo/config.py;本文件是完整架构用的、含 Self-RAG 反思与 PPR 的全集。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HyperParams:
    # --- 检索主循环 ---
    D_MAX: int = 3            # 最大跳数 / 循环轮次兜底
    BEAM_WIDTH: int = 5       # 每跳保留的候选节点数
    # τ_rel 依赖打分器的分数分布:bge-reranker 经 sigmoid 后,无关项普遍聚在 0.50
    # 附近,需略高于 0.5 才能剪掉噪声(实测 0.52 可干净分离玩具图的信号/噪声)。
    # 换打分器或换数据集需重新校准此值。仅在 PRUNE_MODE='absolute' 时使用。
    #
    # Phase 2 观察:引入 Event 节点(content 长)后,reranker 对 Event 候选的打分
    # 普遍偏高(0.50-0.55 范围),Entity 静态关系候选也聚在类似区间,导致信号/噪声
    # 区分度变小。建议:1) 仍用 ratio 模式(默认),让阈值跟绝对分数脱钩;
    # 2) 必要时把 PRUNE_RATIO 调高到 0.94-0.95 收紧;3) 跑 probe_scores.py 看实际分布。
    TAU_REL: float = 0.52     # IsRel / 重排剪枝阈值(绝对模式)
    THETA_STOP: float = 0.7   # 终止置信度阈值

    # 剪枝模式:缓解 τ_rel 对绝对分数敏感的问题。
    #   'absolute' —— 老办法:score >= TAU_REL(对打分器分布敏感)
    #   'ratio'    —— 相对剪枝:score >= 本轮最高分 × PRUNE_RATIO(跟绝对值脱钩,鲁棒)
    #   'gap'      —— 断崖剪枝:在排序后相邻分数差最大处切割(信号/噪声间通常有跳变)
    PRUNE_MODE: str = "ratio"
    PRUNE_RATIO: float = 0.92   # ratio 模式:保留达到最高分这一比例的候选
    PRUNE_FLOOR: float = 0.30   # 相对模式的绝对下限,防止"矮子里拔将军"(全员低分时仍全留)

    # --- Self-RAG 验证 ---
    TAU_SUP: float = 0.6      # IsSup 支撑阈值(NLI 蕴含概率)
    V_MAX: int = 2            # 定向回扩最大次数(防无限打转)

    # --- 入口召回 ---
    SEED_TOPK: int = 5        # 入口候选种子数(宁多勿漏)
    PPR_ALPHA: float = 0.5    # Personalized PageRank restart 概率
    PPR_TOPK: int = 50        # PPR 扩散后保留的节点数(top-p 截断)


PARAMS = HyperParams()
