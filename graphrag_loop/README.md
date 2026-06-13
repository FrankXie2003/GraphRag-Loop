# graphrag_loop —— 完整架构(Phase 1-2 已实现,Phase 3 占位)

分层模块化,依赖自下而上单向流动:
`config / storage / models → ingestion / entry → loop ↔ reflection → generation → orchestration → api`

## 状态图例

| 标记 | 含义 |
|------|------|
| ✅ | 已实现并经测试验证 |
| 🟡 | 历史保留(被新版替代,但保留作演进对照) |
| ⏳ | 占位待实现(文件头有 `# TODO` 说明触发条件) |
| ⛔ | 已删除(早期设计冗余) |

## 分层职责

| 层 / 目录 | 职责 | 状态 |
|----------|------|------|
| `config/` | 全局超参、模型路由、连接配置 | ✅ |
| `storage/` | Neo4j / Qdrant 客户端 + 多模态 / 全文检索占位 | ✅ 主用 / ⏳ object/fulltext |
| `ingestion/` | 离线建图:抽实体关系、双层 schema、多模态占位 | ✅ |
| `entry/` | L1 入口决策 + L1b 混合入口召回 + PPR 占位 | ✅ 主用 / ⏳ ppr |
| `loop/` | Agent 检索主循环:Beam 扩展、剪枝、决策 | ✅ |
| `reflection/` | Self-RAG 反思:Retrieve / IsRel / IsSup(IsUse 占位) | ✅ Retrieve+IsRel+IsSup / ⏳ IsUse |
| `generation/` | 答案生成 + 引用 + 原子断言级 IsSup + 定向回扩 | ✅ |
| `orchestration/` | LangGraph 编排 + 高频缓存(均占位) | ⏳ |
| `models/` | LLM(DashScope)/ NLI / embedding 调用 | ✅ |
| `api/` | FastAPI 对外服务 | ⏳ |
| `tests/` | 单测套件(33/33 通过) | ✅ |
| `utils/` | 轨迹打印 + 运行落盘 | ✅ |

## 目录结构(标注真实状态)

```
graphrag_loop/
├── README.md                  ← 本文件
├── PHASE2.md                  ← Phase 2 过程文档(双层 schema + Self-RAG 反思 + 回扩)
│
├── run_phase2.py              ✅ 主入口:L1→入口召回→loop→生成→IsSup→refeed 完整闭环
├── run_phase1.py              🟡 Phase 1 历史入口(只跑 loop,不含生成)
├── run_tests.py               ✅ 一键跑所有单测(33/33,~30ms)
│
├── check_connections.py       ✅ 4 条线连通性自检(Neo4j/Qdrant/LLM/Embedding)
├── verify_entry.py            ✅ 单测入口召回:query → 种子
├── verify_reflection.py       ✅ 单测三个反思组件
├── verify_nli.py              ✅ 单测 NLI 加载与 IsSup 7 用例
├── probe_scores.py            ✅ Reranker 分数分布探针(τ_rel 校准用)
│
├── config/
│   ├── settings.py            ✅ 全局超参 D_MAX/BEAM/τ_rel/θ_stop/V_MAX/PRUNE_MODE...
│   ├── model_routing.py       ✅ 环节→模型档位(成本/延迟分层)
│   └── connections.py         ✅ Neo4j/Qdrant/DashScope/NLI 连接(从 .env 读)
│
├── storage/
│   ├── graph_store.py         ✅ Neo4j 封装:get_neighbors / 双层节点 / NEXT 时序 / GDS-PPR
│   ├── vector_store.py        ✅ Qdrant 封装:type_filter 双型搜索
│   ├── object_store.py        ⏳ TODO Phase 3:MinIO/S3
│   └── fulltext_store.py      ⏳ TODO 可选:Elasticsearch BM25
│
├── ingestion/
│   ├── chunker.py             ✅ 按段落聚合分块,带 overlap
│   ├── ingest_graph_md_v2.py  ✅ Phase 2 双层建图主脚本(Entity+Event+三类边)
│   ├── ingest_graph_md.py     🟡 Phase 1 单层版(历史保留,见文件头)
│   ├── ingest_toy_data.py     🟡 Phase 0→1 玩具数据迁移演示(历史保留)
│   ├── pipeline.py            ⏳ TODO 可选重构:统一编排
│   ├── extractors/
│   │   ├── event_relation.py  ✅ Phase 2 双层抽取:事件+静态关系
│   │   ├── entity_relation.py 🟡 Phase 1 单层抽取(历史保留)
│   │   └── alignment.py       ✅ 实体/关系归一 + 矛盾边消解
│   └── multimodal/
│       ├── image_encoder.py   ⏳ TODO Phase 3:CLIP
│       ├── audio_encoder.py   ⏳ TODO Phase 3:Whisper
│       └── video_encoder.py   ⏳ TODO Phase 3:抽帧+CLIP
│
├── entry/
│   ├── route_decision.py      ✅ L1 Retrieve:要不要走图谱
│   ├── soft_linker.py         ✅ NER + 向量软链接 → entity 节点
│   ├── hybrid_recall.py       ✅ 双型分路召回(entity + event)
│   └── ppr.py                 ⏳ TODO 待 #10 暴露:PPR 扩散
│
├── loop/
│   ├── controller.py          ✅ 主循环 S1→S3→S4→S5(契约同 demo)
│   ├── beam.py                ✅ expand + 三种剪枝模式(默认 ratio)
│   ├── decision.py            ✅ Rule-based + LLMDecisionPolicy(Self-RAG)
│   └── state.py               ✅ LoopState 双层证据(entities + events)
│
├── reflection/
│   ├── tokens.py              ✅ Retrieve / IsRel / IsSup / IsUse 枚举
│   ├── isrel.py               ✅ bge-reranker cross-encoder
│   ├── issup.py               ✅ NLI 优先 + LLM 兜底 + 假阴性兜底
│   └── isuse.py               ⏳ TODO Phase 3:续写质量评分
│
├── generation/
│   ├── generator.py           ✅ 双轨输出:text + atomic_claims(原子断言级)
│   ├── citation.py            ✅ 引用对齐工具
│   └── refeed.py              ✅ 逐条 IsSup + 缺口提取 + 指纹去重 + V_MAX 兜底
│
├── orchestration/
│   ├── graph_rag_agent.py     ⏳ TODO Phase 3:LangGraph 多 tool 编排
│   └── cache.py               ⏳ TODO Phase 3+:高频子图缓存
│
├── models/
│   ├── llm.py                 ✅ DashScope 封装,按 routing 选大/小
│   ├── nli.py                 ✅ mDeBERTa-v3-xnli(可选,优雅降级)
│   └── embedding.py           ✅ bge-m3(本地)/ DashScope(可切)
│
├── api/
│   └── server.py              ⏳ TODO Phase 3+:FastAPI /query
│
├── utils/
│   ├── trace.py               ✅ 逐跳轨迹打印(stdout)
│   └── recorder.py            ✅ 运行落盘到 logs/*.md
│
└── tests/                     ✅ 33/33 通过, ~30ms
    ├── test_loop.py           ← beam / 剪枝 / 终止 / 双层证据
    ├── test_entry.py          ← hybrid_recall 去重赋权
    ├── test_reflection.py     ← generator / citation / atomic_claims / NLI 兜底 / 指纹
    └── test_e2e.py            ← 玩具图谱端到端
```

## 依赖方向(避免循环依赖)

```
config / models ──▶ 所有层
storage         ──▶ ingestion, entry, loop, generation
loop ◀──▶ reflection           (loop 调 isrel;refeed 触发 loop)
orchestration   ──▶ entry, loop, reflection, generation
api             ──▶ orchestration
```

## 端到端入口

```bash
# 主入口(Phase 2 完整闭环)
python run_phase2.py --query "甄士隐家中遭遇了哪些灾难"

# 配置
python run_phase2.py --policy {rule|llm}     # S5 决策策略
python run_phase2.py --no-refeed              # 关闭定向回扩(对比)
python run_phase2.py --vmax 3                 # 调最大回扩轮数

# 单测(完全离线,33/33 应通过)
python run_tests.py

# 工具
python check_connections.py                   # 基础设施自检
python verify_entry.py --query "..."          # 看入口召回种子
python probe_scores.py --query "..."          # 看 reranker 分数分布
```

## 渐进式接入建议(给未来的人)

1. 先跑 `check_connections.py`,确认 4 条线全绿
2. 跑 `python ingestion/ingest_graph_md_v2.py --probe --probe-chunk 8` 看抽取质量
3. 全量建图 `python ingestion/ingest_graph_md_v2.py`
4. 跑 `run_phase2.py --query "..."` 看完整闭环
5. 看 `logs/*.md` 复盘
6. 改 prompt 或超参 → 跑 `run_tests.py` 防回归

## 相关文档

- 主架构与设计哲学:仓库根 [README.md](../README.md)
- 13 个真实问题与解法:仓库根 [question&solution.md](../question&solution.md)(最值钱的资产)
- Phase 2 演进记录:[PHASE2.md](PHASE2.md)
- 思考过程:仓库根 [think-about-node.md](../think-about-node.md)
