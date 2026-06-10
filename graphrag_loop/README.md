# graphrag_loop — 完整架构

Phase 1-3 的生产级实现,分层模块化。依赖自下而上单向流动:
`storage → ingestion / entry → loop ← reflection → generation → orchestration → api`

## 分层职责

| 层 / 目录 | 职责 | 对应架构图 | Phase |
|----------|------|-----------|-------|
| `config/` | 全局超参、模型路由、连接配置 | §2.1 / §2.2 | 1 |
| `storage/` | 各数据库客户端封装(Neo4j / 向量库 / 对象存储 / ES) | §4 | 1,3 |
| `ingestion/` | 离线建图:抽实体关系、多模态特征提取、入库 | §4 流程 | 1,3 |
| `entry/` | L1 入口决策 + L1b 混合入口召回(软链接 + PPR) | L1 / L1b | 1,2 |
| `loop/` | Agent 检索主循环:Beam 扩展、关系剪枝、Beam 截断、决策 | 主循环 S1-S5 | 1 |
| `reflection/` | Self-RAG 反思:Retrieve / IsRel / IsSup / IsUse | 三层反思 | 2 |
| `generation/` | 基于证据子图生成 + 引用 + 定向回扩触发 | L3 | 1,2 |
| `orchestration/` | 把以上串成一个端到端 pipeline(LangGraph 状态机) | 全图 | 1 |
| `models/` | LLM / 重排 / NLI / embedding 的统一调用接口与分层路由 | §2.2 | 1 |
| `api/` | 对外服务(FastAPI) | — | 1 |
| `tests/` | 单测 + 端到端测试 | — | all |

## 目录结构
```
graphrag_loop/
├── README.md
├── config/
│   ├── __init__.py
│   ├── settings.py            # 超参 D_max/N/τ_rel/θ_stop/τ_sup/V_max...
│   ├── model_routing.py       # 哪个环节用哪个模型(成本/延迟分层)
│   └── connections.py         # Neo4j/向量库/MinIO/ES 连接配置
├── storage/
│   ├── __init__.py
│   ├── graph_store.py         # Neo4j 封装:邻居查询、Cypher、PPR 调用(GDS)
│   ├── vector_store.py        # Qdrant/Chroma:ANN 检索、跨模态
│   ├── object_store.py        # MinIO/S3:原始大文件存取
│   └── fulltext_store.py      # Elasticsearch:关键词/布尔过滤
├── ingestion/
│   ├── __init__.py
│   ├── pipeline.py            # 建图总流程编排
│   ├── chunker.py             # 文本分块
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── entity_relation.py # LLM 抽实体关系 → 三元组
│   │   └── alignment.py       # 实体对齐 / 增量更新
│   └── multimodal/
│       ├── __init__.py
│       ├── image_encoder.py   # CLIP/ViT
│       ├── audio_encoder.py   # Whisper
│       └── video_encoder.py   # 抽帧 + CLIP
├── entry/
│   ├── __init__.py
│   ├── route_decision.py      # L1 Retrieve:要不要走图谱
│   ├── soft_linker.py         # L1b NER + 向量软链接 → 多候选节点
│   ├── hybrid_recall.py       # 向量召回 ∪ 软链接,带权种子
│   └── ppr.py                 # Personalized PageRank 入口预热/扩散
├── loop/
│   ├── __init__.py
│   ├── controller.py          # 主循环骨架(契约同 demo/agent/loop.py)
│   ├── expand.py              # S1 邻居扩展
│   ├── relation_prune.py      # S2 关系剪枝(ToG 式先选边)
│   ├── beam.py                # S4 Beam 截断 + 路径记忆去重
│   ├── decision.py            # S5 Agent 决策(LLM,双约束终止)
│   └── state.py               # 循环状态:frontier/已访问边/证据/置信度
├── reflection/
│   ├── __init__.py
│   ├── tokens.py              # 反思 token 枚举与三值定义
│   ├── isrel.py               # S3 IsRel:Cross-Encoder/NLI 打分剪枝
│   ├── issup.py               # L3 IsSup:段级 NLI 蕴含验证
│   └── isuse.py               # IsUse:整体有用性(可选)
├── generation/
│   ├── __init__.py
│   ├── generator.py           # 证据子图 → 带引用答案
│   ├── citation.py            # inline 引用对齐
│   └── refeed.py              # 定向回扩:unsupported 段 → 回主循环(≤V_max)
├── orchestration/
│   ├── __init__.py
│   ├── graph_rag_agent.py     # 端到端编排(LangGraph 状态机)
│   └── cache.py               # 高频子图模式缓存
├── models/
│   ├── __init__.py
│   ├── llm.py                 # 大/小 LLM 统一接口
│   ├── reranker.py            # bge-reranker Cross-Encoder
│   ├── nli.py                 # 蕴含模型(IsSup/IsRel 可复用)
│   └── embedding.py           # bge-m3 / CLIP 文本侧 / 多模态
├── api/
│   ├── __init__.py
│   └── server.py              # FastAPI:/query 端点
└── tests/
    ├── __init__.py
    ├── test_loop.py           # 循环控制流单测
    ├── test_entry.py          # 入口召回/软链接
    ├── test_reflection.py     # IsRel/IsSup 阈值
    └── test_e2e.py            # 端到端
```

## 依赖方向(避免循环依赖)
```
config ──▶ 所有层
storage ──▶ ingestion, entry, loop, generation
models  ──▶ entry, loop, reflection, generation
loop ◀──▶ reflection (loop 调 isrel;generation 触发 refeed 回 loop)
orchestration ──▶ entry, loop, reflection, generation
api ──▶ orchestration
```

## 渐进式接入建议
1. 先 `storage/graph_store.py` + `loop/` 接通(把 demo 逻辑接真 Neo4j)
2. 再 `entry/` 混合召回 + `reflection/isrel.py` 重排
3. 再 `generation/` + `reflection/issup.py` + `refeed.py`
4. 最后 `ingestion/multimodal/` + 多模态存储
