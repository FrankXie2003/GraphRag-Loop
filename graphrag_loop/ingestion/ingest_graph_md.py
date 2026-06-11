"""真实建图:graph.md(红楼梦第一回)→ chunk → LLM 抽三元组 → Neo4j + Qdrant。

用法:
    python ingestion/ingest_graph_md.py --probe     # 只抽第1块,看抽取质量(省钱,先验证)
    python ingestion/ingest_graph_md.py             # 全量建图
    python ingestion/ingest_graph_md.py --keep       # 不清空,增量追加

设计:先 --probe 验证 LLM 抽取质量,满意了再全量。每块抽取后实体描述用块文本,
节点向量化入 Qdrant(供入口召回),三元组写 Neo4j。
"""

import sys
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_pkg = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_pkg))

from ingestion.chunker import chunk_text
from ingestion.extractors.entity_relation import extract_triples
from storage.graph_store import GraphStore
from storage.vector_store import VectorStore
from models.llm import LLMClient
from models.embedding import get_embedder

GRAPH_MD = _pkg / "graph.md"


def load_text():
    return GRAPH_MD.read_text(encoding="utf-8")


def probe(max_chars, idx=0):
    """只抽指定块,打印三元组,验证抽取质量。不写库。"""
    chunks = chunk_text(load_text(), source="graph.md", max_chars=max_chars)
    print(f"共切成 {len(chunks)} 块。抽取第 {idx} 块(预览前 120 字):")
    print("  " + chunks[idx].text[:120].replace("\n", " ") + "...")
    print("-" * 60)
    llm = LLMClient()
    triples = extract_triples(chunks[idx].text, llm=llm)
    print(f"抽出 {len(triples)} 条三元组:")
    for h, r, t in triples:
        print(f"  ({h}) -[{r}]-> ({t})")


def build(max_chars, keep):
    text = load_text()
    chunks = chunk_text(text, source="graph.md", max_chars=max_chars)
    print(f"共切成 {len(chunks)} 块,开始抽取...")

    llm = LLMClient()
    all_triples = []
    for c in chunks:
        triples = extract_triples(c.text, llm=llm)
        all_triples.extend(triples)
        print(f"  块 {c.chunk_id}: {len(triples)} 条")

    # 去重
    uniq = list(dict.fromkeys(all_triples))
    print(f"\n合计 {len(all_triples)} 条,去重后 {len(uniq)} 条三元组。")

    # 写 Neo4j
    graph = GraphStore()
    if not keep:
        print("清空 Neo4j...")
        graph.clear()
    for h, r, t in uniq:
        graph.upsert_entity(h)
        graph.upsert_entity(t)
        graph.upsert_relation(h, r, t)

    # 收集所有实体,向量化入 Qdrant(描述暂用实体名,后续可换为聚合上下文)
    entities = sorted(set([h for h, _, _ in uniq] + [t for _, _, t in uniq]))
    print(f"写入 {len(entities)} 个实体节点。向量化中...")
    embedder = get_embedder()
    vec = VectorStore()
    vec.ensure_collection(dim=embedder.dim)
    vecs = embedder.encode(entities)
    vec.upsert(list(range(len(entities))), vecs,
               [{"name": e, "text": e} for e in entities])

    graph.close()
    print(f"\n建图完成。{len(entities)} 节点 / {len(uniq)} 边。查看 http://localhost:7474")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true", help="只抽第1块验证质量,不写库")
    p.add_argument("--probe-chunk", type=int, default=0, help="--probe 时抽第几块")
    p.add_argument("--keep", action="store_true", help="不清空,增量追加")
    p.add_argument("--max-chars", type=int, default=500)
    args = p.parse_args()

    if args.probe:
        probe(args.max_chars, args.probe_chunk)
    else:
        build(args.max_chars, args.keep)


if __name__ == "__main__":
    main()
