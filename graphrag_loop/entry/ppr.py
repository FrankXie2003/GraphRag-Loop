"""L1b PPR 入口扩散 —— Personalized PageRank 给入口节点加先验分。

# TODO(待 #10 暴露时实施)
当前状态:占位,未实现。
触发条件:question&solution.md #10 "入口召回噪声种子"成为实际问题时(整句向量召回引入
        过多无关弱节点,影响 BFS 起点质量),再启用 PPR 在多种子上扩散平滑噪声。

实现要点:
  - 用 Neo4j GDS 插件的 gds.pageRank.stream(已在 docker-compose.yml 启用)
  - 以 hybrid_recall 给的多种子作 restart 分布,alpha=PARAMS.PPR_ALPHA(0.5)
  - 取扩散后 top-PPR_TOPK 节点作为最终种子(替代当前 hybrid_recall 的直接输出)
  - 详见架构图主 README §2.1 中的 PPR_alpha / PPR_topk 超参

设计参考:HippoRAG(arXiv:2502.14802)用 PPR 实现"软多种子扩散"代替"硬精确实体链接"。
"""
