# Phase 1 过程文档 —— 单模态文本 GraphRAG 接真组件

> Phase 0(demo/)在内存里验证了检索循环的控制流。
> Phase 1 把同一套控制流接到**真组件**:Neo4j 图谱、bge-reranker 重排、DashScope LLM、本地 bge-m3 向量,
> 并打通"真实文档建图 → 自然语言入口召回 → 多跳检索"的端到端链路。

本文档记录 Phase 1 做了什么、遇到什么真实问题、怎么解决的,以及验证结果。

---

## 1. 目标与边界

**目标**:证明 Phase 0 验证过的循环逻辑能无缝迁移到真后端,并跑通端到端检索。

**本阶段做了**:
- 存储层接真:Neo4j(图)+ Qdrant(向量)
- 模型层接真:DashScope LLM + 本地 bge-m3 embedding + bge-reranker 重排
- 真实建图:红楼梦第一回 → LLM 抽三元组 → 图谱
- 入口召回:自然语言 query → 自动找 BFS 起始种子(替换手写种子)
- 剪枝鲁棒性:用相对剪枝缓解 τ_rel 阈值敏感

**尚未做(留后续 Phase)**:实体/关系对齐、Self-RAG 三层反思(IsSup 验证)、真 LLM 生成、PPR 入口扩散、多模态。

---

## 2. 端到端数据流

```
                          ┌─────────── 建图(离线)───────────┐
  graph.md(红楼梦第一回)→ chunk → LLM抽三元组 → Neo4j(图)
                                          └→ 实体向量化 → Qdrant(向量)
                          └──────────────────────────────────┘

  用户 query(自然语言)
        │
        ▼
  ┌─ 入口召回(entry/)──────────────────────────────┐
  │  ① 整句向量召回:query 向量 → Qdrant top-k 节点    │
  │  ② 软链接:LLM 抽 mention → 每个 mention 匹配节点   │
  │  → 去重赋权 → 起始种子 seeds                       │
  └───────────────────────────────────────────────────┘
        │
        ▼
  ┌─ 检索循环(loop/,控制流同 Phase 0)──────────────┐
  │  for r in 1..D_max:                                │
  │    S1 expand     GraphStore.get_neighbors(Neo4j)   │
  │    S3 score      RerankerScorer(bge-reranker)      │
  │    S3/S4 prune   相对剪枝 + beam 截断               │
  │    S5 decide     规则策略(Phase 2 换 LLM)         │
  └───────────────────────────────────────────────────┘
        │
        ▼
  证据子图 →(Phase 2:交 LLM 生成带引用的答案)
```

**核心设计**:loop/controller.py 的控制流与 demo/agent/loop.py **逐行一致**。
迁移时只替换注入的组件(`MemoryGraph→GraphStore`、`WordOverlapRanker→RerankerScorer`),控制流一行未改。

---

## 3. 组件清单

| 文件 | 职责 | 接的真组件 |
|------|------|-----------|
| `config/connections.py` | 从 .env 读连接配置,绝不硬编码密钥 | — |
| `config/settings.py` | 全局超参(单一事实来源) | — |
| `config/model_routing.py` | 环节→模型档位路由(成本分层) | — |
| `models/llm.py` | LLM 统一接口 | DashScope 通义千问 |
| `models/embedding.py` | 文本向量化(可切后端) | 本地 bge-m3 |
| `reflection/isrel.py` | S3 相关性打分 | bge-reranker cross-encoder |
| `storage/graph_store.py` | 图访问层,`get_neighbors` 契约同 demo | Neo4j(Bolt) |
| `storage/vector_store.py` | 向量库:建集合/upsert/ANN 检索 | Qdrant |
| `ingestion/chunker.py` | 按段落分块,留重叠 | — |
| `ingestion/extractors/entity_relation.py` | LLM 抽三元组,JSON 容错+约束 prompt | DashScope |
| `ingestion/ingest_graph_md.py` | 建图脚本:probe 验证 + 全量 | 全栈 |
| `entry/soft_linker.py` | 抽 mention + 软链接到图节点 | DashScope + Qdrant |
| `entry/hybrid_recall.py` | 整句向量召回 ∪ 软链接,去重赋权 | Qdrant |
| `loop/{controller,beam,state,decision}.py` | 检索循环(逻辑同 Phase 0,自包含) | — |
| `check_connections.py` | 四条线连通性自检 | 全栈 |
| `run_phase1.py` | 端到端入口 | 全栈 |

---

## 4. 关键过程与真实发现

按实际推进顺序记录,重点是**真实暴露的问题**——这些是玩具阶段看不到、接真组件才会浮现的。

### 4.1 先写连通性自检,再写业务
接真组件前先写 `check_connections.py` 逐条点亮 Neo4j/Qdrant/LLM/Embedding。
**原因**:把基础设施问题和业务逻辑问题隔离开,出错时不用在一堆代码里猜是 key 错、Docker 没起、还是逻辑 bug。

### 4.2 loop 迁移:控制流契约的价值
demo 的控制流接到真图时一行没改。代价是踩了两个工程坑:
- **包名冲突**:demo/config.py(模块)和 graphrag_loop/config/(包)撞名。
  解法:graphrag_loop 自包含(把 loop 逻辑移植到 loop/),只导入 `demo.toy_data` 这一个纯数据模块。
- **Qdrant point id**:不接受中文字符串,只接受整数/UUID。解法:整数 id + payload 存 name。

### 4.3 τ_rel 阈值敏感(重要)
bge-reranker 的分数经 sigmoid 后,**无关候选普遍聚在 0.50 附近**,区分度低。
demo 的词重叠打分器分布散,τ_rel=0.25~0.5 都能用;换成真 reranker 后 τ_rel=0.5 几乎不剪枝。
这是接真模型才暴露的问题。

### 4.4 相对剪枝:解决阈值敏感
在 `loop/beam.py` 实现三种剪枝模式(可 `--prune-mode` 切换):

| 模式 | 规则 | 评价 |
|------|------|------|
| `absolute` | `score ≥ τ_rel` | 对打分器分布敏感,换模型/数据要重调 |
| **`ratio`(默认)** | `score ≥ 本轮最高分 × 0.92` 且 `≥ floor` | **跟绝对值脱钩,跨 query 稳定** |
| `gap` | 排序后相邻分数差最大处切割 | 分数相等时切割点随机,**可能误杀,不做默认** |

**验证**:真实图谱上两个 query 用**同一组参数**:
- Q1「甄士隐送贾雨村什么」:24 个候选里精准只留 `赠银给`(0.73)、`赠衣给`(0.72),其余全剪。
- Q2「英莲的父亲」:候选全挤在 0.50-0.52,ratio 切不开留 5 条,但正确答案排最高分、未误杀。

结论:ratio 把"敏感的魔数"变成"相对规则",真实复杂分布下比固定阈值鲁棒得多。

### 4.5 真实建图:probe-first 策略
不直接全量跑(22 块 = 22 次 LLM 调用),先 `--probe` 抽单块看质量。
**在花钱前**发现三个抽取问题并修正 prompt:
1. 把"姓甄名费字士隐"拆成 `姓/名/字` 三条边(属性当关系)→ prompt 明确忽略
2. 同时抽 A→B 和 B→A 反向冗余边 → prompt 要求只抽一个方向
3. overlap 导致跨块重复 → pipeline 去重

修正后全量:**47 节点 / 74 边**,核心人物关系(赠银、赠衣、女儿、妻子、岳丈)都正确。
也保留了真实噪声(`邀/邀请` 同义未合并、个别方向抽反),这是实体对齐要解决的,符合预期。

### 4.6 入口召回:从手写种子到自然语言
HippoRAG 思路——不做精确实体链接(单点故障),而是多种子+概率扩散:
- 整句向量召回 + 每个 mention 软链接,取并集赋权。
- 精确 mention(甄士隐、贾雨村)以 weight=1.0 命中,排最前。

### 4.7 版本一致性坑
qdrant-client 1.18 用 `query_points`,但 server 1.9 不支持该端点(404)。
**解法**:升级 docker 的 Qdrant 镜像到 1.12.4,client/server 对齐。治本而非将就。

---

## 5. 如何运行 / 验证

前置:Docker 起 Neo4j+Qdrant,venv 激活,.env 配好 DashScope key。

```bash
# ① 基础设施自检(应 4/4)
python check_connections.py

# ② 真实建图(先 probe 看质量,再全量)
python ingestion/ingest_graph_md.py --probe --probe-chunk 7
python ingestion/ingest_graph_md.py

# ③ 单看入口召回(query → 种子)
python verify_entry.py --query "英莲的父亲是谁"

# ④ 端到端(query → 入口召回 → 检索 → 证据子图)
python run_phase1.py --query "甄士隐送给贾雨村什么东西"
```

---

## 6. 已知局限与下一步

**已知局限**(都在架构规划内,留后续):
1. **入口召回有噪声种子**:整句向量召回会引入弱相关节点(如"雨露""警幻仙子")。
   不致命——低权重噪声进 BFS 起点后会被剪枝收敛。可优化:给软链接命中的种子更高优先级。
2. **实体/关系未对齐**:`邀/邀请`、`寄居于葫芦庙/寄居于庙中` 同义未合并;个别关系方向抽反。
3. **决策与生成仍是占位**:S5 用规则策略,答案是模板拼接。

**下一步候选**:
- 实体/关系对齐(`extractors/alignment.py`)
- Phase 2:Self-RAG 三层反思(IsRel/IsSup/决策都换 LLM/NLI)+ 真 LLM 生成
- 朝**终极测试**(喂红楼梦前 50 回,让模型续写第 51-55 回)扩大图谱规模

---

## 附:关键超参(config/settings.py)

| 超参 | 值 | 含义 |
|------|----|----|
| D_MAX | 3 | 最大跳数 |
| BEAM_WIDTH | 5 | 每跳保留候选数 |
| TAU_REL | 0.52 | 绝对剪枝阈值(仅 absolute 模式) |
| PRUNE_MODE | ratio | 默认相对剪枝 |
| PRUNE_RATIO | 0.92 | 保留达最高分这一比例的候选 |
| THETA_STOP | 0.7 | 终止置信度 |
| SEED_TOPK | 5 | 入口种子数 |
