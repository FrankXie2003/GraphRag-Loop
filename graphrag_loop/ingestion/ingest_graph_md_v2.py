"""双层 schema 建图(Phase 2):graph.md → chunk → 抽 {事件+参与+本体关系}
   → Neo4j(Entity+Event 双层节点,三类边)+ Qdrant(双类型向量)。

用法:
    python ingestion/ingest_graph_md_v2.py --probe --probe-chunk 8   # 看单块双层结构
    python ingestion/ingest_graph_md_v2.py                            # 全量建图

与 Phase 1 的 ingest_graph_md.py 区别:产出 Event 节点 + PARTICIPATES + NEXT 时序边,
Event content 入 Qdrant 作为续写检索主力。
"""

import sys
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_pkg = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_pkg))

from ingestion.chunker import chunk_text
from ingestion.extractors.event_relation import extract_dual
from ingestion.extractors.alignment import canonicalize
from storage.graph_store import GraphStore
from storage.vector_store import VectorStore
from models.llm import LLMClient
from models.embedding import get_embedder

GRAPH_MD = _pkg / "graph.md"


def load_text():
    return GRAPH_MD.read_text(encoding="utf-8")


def probe(max_chars, idx):
    chunks = chunk_text(load_text(), source="graph.md", max_chars=max_chars)
    print(f"共 {len(chunks)} 块。抽取第 {idx} 块的双层结构:")
    print("  原文:" + chunks[idx].text[:100].replace("\n", " ") + "...")
    print("-" * 60)
    events, relations = extract_dual(chunks[idx].text, llm=LLMClient())
    if events:
        print(f"抽出 {len(events)} 个事件:")
        for ev in events:
            print(f"  ● {ev['name']}  参与: {ev['participants']}")
            print(f"    content: {ev['content']}")
    else:
        print("事件: (无 —— 纯抒情/议论)")
    print(f"静态本体关系 {len(relations)} 条:")
    for h, r, t in relations:
        print(f"  ({h}) -[{r}]-> ({t})")


def build(max_chars):
    chunks = chunk_text(load_text(), source="graph.md", max_chars=max_chars)
    print(f"共 {len(chunks)} 块,开始双层抽取...")

    llm = LLMClient()
    events = []        # [{name, content, chunk_id, order, participants}]
    relations = []     # [(h,r,t)]
    order = 0          # 全局事件序号(跨块累加,保证 NEXT 时序唯一且单调)
    for c in chunks:
        evs, rels = extract_dual(c.text, llm=llm)
        relations.extend(rels)
        for ev in evs:
            ev["chunk_id"] = c.chunk_id
            ev["order"] = order
            order += 1
            events.append(ev)
        print(f"  块 {c.chunk_id}: 事件 {len(evs)},关系 {len(rels)}")

    # 实体对齐:统一 relations 与 participants 里的人物名
    all_entities = ([h for h, _, _ in relations] + [t for _, _, t in relations]
                    + [p for ev in events for p in ev["participants"]])
    ent_map = canonicalize(all_entities, "实体", llm)
    merges = {k: v for k, v in ent_map.items() if k != v}
    if merges:
        print(f"  实体合并 {len(merges)} 项: {merges}")

    relations = [(ent_map.get(h, h), r, ent_map.get(t, t)) for h, r, t in relations]
    relations = [tr for tr in dict.fromkeys(relations) if tr[0] != tr[2]]
    for ev in events:
        ev["participants"] = list({ent_map.get(p, p) for p in ev["participants"]})

    # ---- 写 Neo4j ----
    graph = GraphStore()
    print("清空 Neo4j...")
    graph.clear()

    # Entity + 静态关系
    for h, r, t in relations:
        graph.upsert_entity(h)
        graph.upsert_entity(t)
        graph.upsert_relation(h, r, t)

    # Event + PARTICIPATES
    for ev in events:
        graph.upsert_event(ev["name"], ev["content"],
                           chunk_id=ev["chunk_id"], order=ev["order"])
        for p in ev["participants"]:
            graph.upsert_entity(p)
            graph.upsert_participation(p, ev["name"])

    # NEXT 时序边(按 order 相邻连接)
    events_sorted = sorted(events, key=lambda e: e["order"])
    for a, b in zip(events_sorted, events_sorted[1:]):
        graph.upsert_event_sequence(a["name"], b["name"])

    # ---- 写 Qdrant(双类型)----
    embedder = get_embedder()
    vec = VectorStore()
    vec.ensure_collection(dim=embedder.dim)

    entities = sorted(set([h for h, _, _ in relations] + [t for _, _, t in relations]
                          + [p for ev in events for p in ev["participants"]]))
    # entity 向量:用实体名;event 向量:用 content(检索主力)
    ent_payloads = [{"name": e, "type": "entity", "text": e} for e in entities]
    ev_payloads = [{"name": ev["name"], "type": "event",
                    "text": ev["content"], "chunk_id": ev["chunk_id"],
                    "participants": ev["participants"]} for ev in events]

    texts = [p["text"] for p in ent_payloads] + [p["text"] for p in ev_payloads]
    payloads = ent_payloads + ev_payloads
    vecs = embedder.encode(texts)
    vec.upsert(list(range(len(payloads))), vecs, payloads)

    graph.close()
    print(f"\n双层建图完成:")
    print(f"  Entity {len(entities)} | Event {len(events)} | 静态边 {len(relations)} | NEXT 边 {max(0,len(events)-1)}")
    print(f"  Qdrant: {len(entities)} entity向量 + {len(events)} event向量")
    print(f"  查看 http://localhost:7474")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    p.add_argument("--probe-chunk", type=int, default=8)
    p.add_argument("--max-chars", type=int, default=500)
    args = p.parse_args()
    if args.probe:
        probe(args.max_chars, args.probe_chunk)
    else:
        build(args.max_chars)


if __name__ == "__main__":
    main()
