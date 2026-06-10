# Phase 0 — 最小可跑闭环 Demo

不依赖 Neo4j、不依赖大模型(或只用小句向量模型 / mock 打分)。
目标:**验证 Agent 检索循环的控制流是否正确** —— Beam 扩展、剪枝、终止条件、路径去重。

## 它验证什么
- [ ] 主循环能正确扩展邻居、按 beam_width 截断
- [ ] 剪枝阈值 τ_rel 生效,无关分支被砍
- [ ] 双约束终止(θ_stop / D_max)先到为准
- [ ] 路径记忆:不重复遍历同一条边
- [ ] 终止后能基于证据子图给出答案

## 文件结构
```
demo/
├── README.md
├── run_demo.py            # 入口:加载玩具图 → 跑一次完整 loop → 打印轨迹
├── config.py             # 超参:D_max / beam_width / τ_rel / θ_stop
├── toy_data.py           # 手写的玩具知识图谱(dict / 三元组)
├── graph/
│   ├── __init__.py
│   └── memory_graph.py   # networkx 内存图:邻居查询、边去重
├── retrieval/
│   ├── __init__.py
│   ├── ranker.py         # 节点打分:句向量 cosine 或 mock
│   └── beam.py           # Beam 扩展 + 截断
├── agent/
│   ├── __init__.py
│   ├── loop.py           # 检索主循环(本 demo 的核心)
│   └── policy.py         # Agent 决策:规则模拟置信度 / 是否停止
└── utils/
    ├── __init__.py
    └── trace.py          # 轨迹记录与可视化打印(看清每跳发生了什么)
```

## 怎么跑
```bash
cd demo
python run_demo.py            # 用默认 query + 玩具图
python run_demo.py --query "..." --beam 3 --dmax 4
```

## 与完整架构的关系
demo 里的 `agent/loop.py` 决策流程,会在 `graphrag_loop/loop/` 里被替换成接真 Neo4j、真重排模型、真 LLM 的版本。loop 的**控制流契约保持一致**,所以 demo 验证过的逻辑可直接迁移。
