"""高频子图模式缓存 —— 同 query / 同种子时复用证据子图,降低 LLM 调用成本。

# TODO(Phase 3+,生产部署时)
当前状态:占位,未实现。
触发条件:对外提供服务、有真实 QPS 时再做。开发期每次 query 都跑全链路,反而便于看清行为。

三种粒度,可分阶段做:
  L1 查询级:hash(query) → answer。最简单,完全相同 query 直接返回。
  L2 子图级:hash(sorted(seeds)) → evidence_subgraph。query 措辞不同但种子相同时复用。
  L3 节点扩展级:(node, query) → ranker 分数。reranker 是热点,缓存性价比高。

为什么续写任务对此意义不大:每次续写 query 都不同,L1 命中率低;但 L2/L3 仍有价值
        (续写过程中可能反复检索同一批人物的事件子图)。

实现要点:
  - 后端:内存 dict(开发期)→ Redis(生产)
  - 失效策略:图谱重建时清空(增加 graph_store.clear() 的 hook)
  - 命中率监控:打印 cache hit ratio 进 RunRecorder
"""
