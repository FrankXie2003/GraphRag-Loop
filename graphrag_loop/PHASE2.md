# Phase 2 过程文档 —— 双层 schema 建图 + Self-RAG 反思

> Phase 1 用单层「实体-关系」三元组建图,跑通了检索循环。
> Phase 2 升级为「实体 + 事件」双层 schema(见主 README §3.5),让图谱能承载叙事情节,
> 直接服务**终极测试**(喂红楼梦前 50 回,让模型续写第 51-55 回)。

本文档随 Phase 2 推进持续更新,记录设计、真实问题、验证结果。

---

## 1. 为什么要双层 schema

Phase 1 的单层 schema 暴露了叙事文本的根本矛盾:**边既要表达静态关系,又要承载动态情节**。
- 静态:`甄士隐-[女儿]->英莲`(薄,客观拓扑)
- 动态:`甄士隐-[赠银给]->贾雨村`(其实背后是"中秋夜资助贾雨村赴考"这个有细节的事件)

把情节塞进边 → 图爆炸 + 信息载体错位(详见主 README §3.5)。
解法:**事件节点化**——Entity(薄)+ Event(厚)双层节点 + 三类边。

**对续写的意义**:续写需要的是带细节的事件(情节)+ 事件先后顺序,而非静态拓扑。
Event 节点的 content + `[:NEXT]` 时序边正是为此设计。

---

## 2. 目标 schema

```
节点:
  (:Entity {name, desc})              人物/地点/物 —— 薄
  (:Event  {name, content, chunk_id, order})  事件 —— 厚

边(全部薄边):
  (:Entity)-[:RELATES {type}]->(:Entity)     静态本体关系
  (:Entity)-[:PARTICIPATES]->(:Event)        参与
  (:Event)-[:NEXT]->(:Event)                 时序(order 相邻)

Qdrant(双类型 point):
  type=entity : 实体名向量,payload {name, type}
  type=event  : 事件 content 向量,payload {name, type, chunk_id, participants}
```

---

## 3. 任务分解与进度

| # | 任务 | 状态 |
|---|------|------|
| 1 | 主 README 加双层 schema 设计(§3.5) | ✅ |
| 2 | 本文档 | ✅(随进度更新) |
| 3 | 抽取器:每 chunk 抽 {主事件 + 参与人物 + 本体关系} | ✅ |
| 4 | graph_store:Event 节点 / 三类边 / 时序边写入 | ✅ |
| 5 | 建图脚本:Event content 入 Qdrant(带 type) | ✅ |
| 6 | 重建双层图谱(qwen-max + 多事件 prompt) | ✅ 30 事件 / 29 时序边 |
| 7 | Self-RAG 三层反思:Retrieve / IsRel / IsSup | ✅ 12/12 验证通过 |
| 8 | 检索循环适配双层(Event 参与 BFS、双型召回、LLM 决策) | ✅ |
| 9 | 生成 + 定向回扩(refeed) | ✅ |
| 10 | P0-1 诚实拒答跳过 IsSup | ✅ 实测验证 |
| 11 | P0-2 重复回扩检测(指纹相同则中止) | ✅ |
| 12 | P0-3 τ_rel 重新校准 → probe_scores.py 实测后**保持 0.92** | ✅ 6/26 精准切噪声 |
| 13 | P1-4 NLI 真接入(mDeBERTa-v3-xnli) | ✅ verify_nli 7/7 通过 |
| 14 | P1-7 单测套件(stdlib unittest) | ✅ 21/21 通过, 33ms |
| 15 | P1-8 轨迹落地(logs/*.md) | ✅ |

---

## 4. 设计决策记录

### 4.1 事件边界 = chunk 边界
不让 LLM 凭空划事件边界(易碎且不一致),而是**一个 chunk 抽一个主事件**。
chunk 已按语义段落切,一个段落通常对应一个场景,粒度天然合适。
事件的 `order` 直接用 chunk_id(分块本就是顺序的),用来建 `[:NEXT]` 时序边。

### 4.2 抽取输出结构变化
```
Phase 1:  chunk → [(h, r, t), ...]
Phase 2:  chunk → {
            event: {name, content, participants: [...]},   # 0 或 1 个主事件
            relations: [(h, r, t), ...]                     # 静态本体关系
          }
```
prompt 要给 LLM「静态关系 vs 动态事件」的判据:亲属/从属/称谓=静态;有动作/情节/时间性=事件。

### 4.3 对齐如何适配
实体对齐(alignment.py)继续作用于 Entity 与 participants。
Event 一般不需别名合并(每个 chunk 一个,天然唯一),但 participants 里的人物名要走实体归一,
保证 PARTICIPATES 边连到的是规范实体。

---

## 5. 运行 / 验证(随实现补充)

```bash
# 双层建图(probe 先看单块抽取结构,再全量)
python ingestion/ingest_graph_md_v2.py --probe --probe-chunk 8
python ingestion/ingest_graph_md_v2.py

# 三个反思组件验证(Retrieve / IsRel / IsSup)
python verify_reflection.py
```

## 6. 真实发现(已记入项目)

### 6.1 事件抽取召回率受 prompt 门槛与模型能力主导
最初 prompt 把"事件"判据设保守(单事件、要求"核心剧情"),qwen-plus 下 22 块只抽 3 事件。
两项改动后(降低门槛允许多事件 + 粒度控制 + qwen-max),事件数从 3 升至 30,
有事件块占比 14% → 82%,且粒度稳定在 1-3。说明:**抽取质量主战场是 prompt + 模型,
对齐只是兜底**(对齐合并别名,但解决不了"漏抽"和"方向抽反")。

### 6.2 事件粒度需要 prompt 显式约束
qwen-plus 旧 prompt 块 12 抽 7 个碎事件("抱女欲入"+"念偈"+"消失"+"悔未问"...);
prompt 加"以场景为粒度,合并连续琐碎动作,1-3 个/块"约束后,合并为 2 个完整场景事件
("癞僧念谶语后与道人飘然离去"涵盖了之前的 4 个碎动作)。

### 6.3 IsUse 暂不实现的理由
Self-RAG 四 token 中只用了前三个。Retrieve/IsRel/IsSup 都驱动检索流程(走不走图、
剪不剪枝、回不回扩),IsUse 只对答案打有用性分,不驱动流程。预留给续写场景作为
"续写质量评分器",与 CoVe 自我核验一起用更合适。tokens.py 已定义但未启用。

### 6.4 NLI 的优雅降级设计
IsSup 优先 NLI(便宜可量化),NLI 不可用(.env 未配 NLI_MODEL 或加载失败)
则透明退到 LLM 兜底。这避免了"必须下载 NLI 模型才能跑 Phase 2"的强依赖。
当前 LLM(qwen-max)兜底效果已经很好(4/4),后续要量化分布时再接 NLI 模型。
