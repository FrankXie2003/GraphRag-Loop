# Phase 0 — 最小可跑闭环 Demo

不依赖 Neo4j、不依赖大模型,**零第三方依赖**(只用 Python 标准库)即可运行。
目标:**验证 Agent 检索循环的控制流是否正确** —— Beam 扩展、阈值剪枝、top-N 截断、双约束终止、路径去重。

> 这是整个项目的"骨架验证":在玩具规模上把检索主循环跑对,再去 `graphrag_loop/` 接真 Neo4j / 真模型。
> demo 的控制流契约与 `graphrag_loop/loop/controller.py` 一致,验证过的逻辑可直接迁移。

---

## 快速开始

```bash
cd demo
python run_demo.py                                  # 默认 query + 玩具图
python run_demo.py --query "马云毕业于哪所学校" --answer-hint 杭州师范大学
python run_demo.py --beam 5 --dmax 4 --tau 0.3      # 覆盖超参
python run_demo.py --embedding                      # 用句向量打分(需 sentence-transformers,否则自动回退)
```

零依赖即可跑。`--embedding` 是唯一需要装包的可选项(见 `requirements.txt`)。

环境提示:本机用 winget 装的 Python 3.12,装好后**需重开终端**才能让 `python` 命令进 PATH。
若 `python` 仍报 "not found / 跳 Microsoft Store",说明安装时未勾选 *Add to PATH*,需手动补或重装勾选。

---

## 它验证什么

- [x] 主循环能正确扩展邻居
- [x] 阈值剪枝 τ_rel 生效,低分(无关)分支被砍
- [x] top-N 截断 beam_width 生效,高分但超量的分支被丢
- [x] 双约束终止(θ_stop / D_max)先到为准
- [x] 路径记忆:不重复遍历同一条边(图里埋了环来检验)
- [x] 终止后能基于证据子图给出答案
- [x] **附带发现**:打分必须把"关系类型"纳入,否则正确路径会被噪声盖过(见下文)

---

## 打通的是哪段逻辑(边界)

只打通完整架构里的 **Agent 检索主循环**,加一个极简入口和结尾。
**不含**:文档建图(三元组手写)、真 LLM、Neo4j、真重排模型、Self-RAG 反思、多模态。

```
  query + 手写种子节点                  ← 简化入口(不做 L1 决策 / 软链接 / PPR)
        │
        ▼
  ┌──────────────────── Agent Loop:for r = 1 .. D_max ────────────────────┐
  │                                                                        │
  │  S1 扩展          expand():frontier 各节点取邻居                       │
  │     │             已访问边 / 本轮重复边 → 跳过 ← 路径记忆去重           │
  │     ▼                                                                  │
  │  S3 打分          score_candidates():用 [关系名 + 邻居描述] 打分       │
  │     │             score ∈ [0,1]                                        │
  │     ▼                                                                  │
  │  ┌── prune_and_truncate() ─────────────────────────────────┐          │
  │  │  ① 阈值剪枝     砍掉 score < τ_rel 的候选                 │          │
  │  │  ② top-N 截断   剩下的按分排序,取前 beam_width 个        │          │
  │  └──────────────────────────────────────────────────────────┘         │
  │     │             保留的候选 → 并入证据子图 + 登记已访问边              │
  │     ▼                                                                  │
  │  S5 决策          policy.decide():估累积置信度 conf                    │
  │     │                                                                  │
  │     ├─ conf ≥ θ_stop      → break,去生成                              │
  │     ├─ r ≥ D_max          → break(兜底)                              │
  │     ├─ 本轮无候选存活      → break(死胡同)                            │
  │     └─ 否则 → frontier = 本轮保留节点,进入 r+1 ──────────────────────┤
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  基于证据子图拼出答案(模板,非真 LLM)+ 打印完整轨迹
```

**精确执行顺序**(你问的那条链,逐步对应 [agent/loop.py](agent/loop.py)):

```
扩展邻居 → 打分 → ①阈值剪枝(τ_rel) → ②top-N截断(beam_width)
        → 并入证据 → 决策:达θ_stop?到D_max?死胡同? → break / 下一轮
```

注意 ① 和 ② 的先后:**先按阈值过滤,再按数量截断**,两步都在 `prune_and_truncate()` 内。

**对应关系**:demo 的 `S1/S3/S4/S5` 对应主架构 [README §2](../README.md) 的同名步骤。
省掉的 L1、L1b、S2、L3 各自是独立可验证的环节,放到 Phase 1+ 单独接,避免一次耦合太多未验证的部分。

---

## 实际运行输出(默认 query)

```
Query : 阿里巴巴的总部在哪个省
Seeds : ['阿里巴巴']
超参  : D_max=3  beam_width=3  τ_rel=0.25  θ_stop=0.6

── 第 1 跳 ──
  frontier: ['阿里巴巴']
     [0.40] 阿里巴巴 -[总部位于]-> 杭州      [KEEP] 保留
     [0.40] 阿里巴巴 -[创始人]-> 马云        [KEEP] 保留
     [0.00] 阿里巴巴 -[主营]-> 电子商务      [DROP] 剪枝(score<0.25)
  → 决策: 继续 (conf=0.40) — 置信度 < θ_stop

── 第 2 跳 ──
  frontier: ['杭州', '马云']
     [0.30] 杭州 -[位于]-> 浙江省            [KEEP] 保留
     [0.30] 杭州 -[省会]-> 浙江省            [KEEP] 保留
     [0.20] 杭州 -[相邻]-> 绍兴              [DROP] 剪枝(score<0.25)
     [0.10] 杭州 -[著名景点]-> 西湖          [DROP] 剪枝(score<0.25)
     ... (亚运会/临安/英语教师等噪声全部被剪) ...
  → 决策: 停止 (conf=0.80) — 置信度 0.80 ≥ θ_stop 0.6

终止原因: 置信度 0.80 ≥ θ_stop 0.6
证据子图: 阿里巴巴 -[总部位于]-> 杭州 -[位于]-> 浙江省  ✓ 多跳路径走通
```

第 2 跳成功触达答案 `浙江省`,噪声分支被剪枝,θ_stop 先于 D_max 触发终止。

---

## 一个真实发现:关系必须参与打分

最初实现里 `ranker` 只给**邻居节点描述**打分,忽略了**关系类型**,结果第 1 跳:
正确边 `阿里巴巴-[总部位于]->杭州` 得 0.20 被剪,噪声边 `-[创始人]->马云` 反而得 0.40 保留 → 走进死胡同。

根因:问题里的"总部""省"应让"总部位于"这条**边**高分,但只看 tail 节点描述时,
"马云"的描述恰好与 query 有词重叠,被误判。

修复:打分文本改为 `f"{关系名} {邻居描述}"`(见 [retrieval/beam.py](retrieval/beam.py) `score_candidates`)。
这等价于把 ToG 的 **S2 关系剪枝**合进了 S3 统一打分 —— 也印证了完整架构里 S2 不能省。

> 这正是 demo 的价值:在玩具规模上提前暴露会影响完整架构的设计点。

---

## 超参与调优权衡

集中在 [config.py](config.py),命令行可覆盖:

| 超参 | 含义 | 默认 | 调高 | 调低 |
|------|------|------|------|------|
| `D_MAX` | 最大跳数 / 轮次兜底 | 3 | 支持更深推理,延迟↑ | 可能走不到答案 |
| `BEAM_WIDTH` | 每跳保留候选数 | 3 | 召回↑,图爆炸风险↑ | 快,易漏正确分支 |
| `TAU_REL` | 阈值剪枝线 | 0.25 | 噪声少,**易误杀正确路径** | 召回全,噪声多 |
| `THETA_STOP` | 终止置信度 | 0.6 | 更保守,多探几跳 | 早停,可能证据不足 |

> 实测:`--tau 0.35` 时正确边 `杭州-[位于]->浙江省`(0.30)被误杀,直观演示了
> τ_rel "太低噪声多 / 太高漏召回" 的权衡。这是后续在真数据上调参的重要直觉。

---

## 玩具图设计(toy_data.py)

手写三元组,**不基于文档抽取**(那是 Phase 1 的事)。图虽小,但精心埋了四类"考点":

| 考点 | 埋点 | 验证什么 |
|------|------|---------|
| 多跳路径 | 阿里巴巴→杭州→浙江省 | loop 能走到底 |
| 噪声分支 | 马云/西湖/亚运会/英语教师 | 剪枝 + 截断把它们砍掉 |
| 环 | 杭州↔浙江省、杭州师范大学→杭州 | 路径去重不无限打转 |
| 高度数节点 | 杭州挂 7+ 邻居 | beam_width 截断生效 |

---

## 文件结构

```
demo/
├── README.md
├── requirements.txt       # 零依赖;仅 --embedding 时需解开注释
├── run_demo.py            # 入口:加载图 → 跑 loop → 打印轨迹(强制 UTF-8 输出)
├── config.py              # 超参 DemoConfig
├── toy_data.py            # 玩具图三元组 + 节点描述 + 默认 query/seeds
├── graph/
│   └── memory_graph.py    # 纯 dict 内存图:双向邻居查询(契约同 Neo4j 版)
├── retrieval/
│   ├── ranker.py          # 词重叠打分(默认,确定性)+ 句向量(可选)
│   └── beam.py            # expand / score / prune_and_truncate
├── agent/
│   ├── loop.py            # 检索主循环 + LoopState(本 demo 核心)
│   └── policy.py          # 规则模拟置信度,双约束终止决策
└── utils/
    └── trace.py           # 逐跳轨迹打印 + 模板答案
```

各文件职责一行版:

- `toy_data.py` — 数据。改这里换图/换问题。
- `memory_graph.py` — 图访问层。换成 Neo4j 时只动这个文件的实现,接口不变。
- `ranker.py` — 打分。对应完整架构的 Cross-Encoder / NLI 重排。
- `beam.py` — 宽度控制(防图爆炸的核心:阈值 + top-N)。
- `policy.py` — 决策。对应完整架构里换成真 LLM 的 Self-RAG 决策。
- `loop.py` — 把以上串成一轮轮循环;`LoopState` 持有 frontier / 已访问边 / 证据 / 置信度。
- `trace.py` — 可观测性,看清每跳发生了什么。

---

## 与完整架构的关系

| demo 文件 | 迁移到 graphrag_loop 后变成 | 接什么真组件 |
|-----------|---------------------------|------------|
| `graph/memory_graph.py` | `storage/graph_store.py` | Neo4j(Bolt + Cypher) |
| `retrieval/ranker.py` | `reflection/isrel.py` | bge-reranker / NLI |
| `retrieval/beam.py` | `loop/expand.py` + `loop/beam.py` | 真邻居查询 + 关系剪枝 |
| `agent/policy.py` | `loop/decision.py` | 真 LLM(Self-RAG 决策) |
| `agent/loop.py` | `loop/controller.py` | 控制流不变,只换各步实现 |

**控制流契约保持一致**是这套设计的关键 —— Phase 0 验证的循环逻辑,到 Phase 1 一行控制流都不用重写,只替换被调用的实现。
