"""Phase 1 最小建图(**历史保留**):把 demo 玩具三元组灌进真 Neo4j + Qdrant。

──────────────────────────────────────────────────────────────────────
⚠️  这是 Phase 0→Phase 1 迁移期的对照演示脚本。
    日常使用请用 ingest_graph_md_v2.py(Phase 2 双层 schema,真实数据)。

    保留原因:
      - 是"控制流契约一致是迁移安全网"这条经验的具体证据(question&solution.md 关键认知 #1)
      - 用 demo 玩具数据验证"loop 接真图能跑"和"LLM 抽取准不准"的解耦
      - run_phase1.py 仍引用此脚本,作为 Phase 1 端到端的对照可跑版本
──────────────────────────────────────────────────────────────────────

用法:
    python ingestion/ingest_toy_data.py

作用:让真图里的数据和 demo 内存图一致,这样"loop 能不能在真图上跑"和"LLM 抽取准不准"
就解耦了——先验证前者(loop 接真图),LLM 抽取留到下一阶段。

会清空 Neo4j 和 Qdrant,慎用。生产数据用 ingest_graph_md_v2.py。
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 让脚本无论从哪个目录运行都能 import:
#   _pkg  = graphrag_loop/  → import storage / models / config
#   _root = 仓库根/         → import demo.toy_data
_pkg = Path(__file__).resolve().parents[1]
_root = _pkg.parent
sys.path.insert(0, str(_pkg))
sys.path.insert(0, str(_root))

from demo.toy_data import TRIPLES, NODE_DESC, DEFAULT_QUERY, DEFAULT_SEEDS
from storage.graph_store import GraphStore
from storage.vector_store import VectorStore
from models.embedding import get_embedder


def main():
    print("=" * 60)
    print("Phase 1 最小建图:灌 demo 玩具数据到真 Neo4j + Qdrant")
    print("=" * 60)

    # --- 1. 清空旧数据(可选,但方便重跑) ---
    print("\n[1/3] 清空 Neo4j...")
    graph = GraphStore()
    graph.clear()

    # --- 2. 写三元组 → Neo4j ---
    print(f"[2/3] 写 {len(TRIPLES)} 条三元组 → Neo4j...")
    for head, rel, tail in TRIPLES:
        graph.upsert_entity(head, desc=NODE_DESC.get(head))
        graph.upsert_entity(tail, desc=NODE_DESC.get(tail))
        graph.upsert_relation(head, rel, tail)
    print(f"      → 写入 {len(set(h for h,_,_ in TRIPLES) | set(t for _,_,t in TRIPLES))} 个节点")

    # --- 3. 节点描述向量化 → Qdrant ---
    print("[3/3] 节点描述向量化 → Qdrant...")
    embedder = get_embedder()
    vec_store = VectorStore()
    vec_store.ensure_collection(dim=embedder.dim)

    nodes_with_desc = [(name, desc) for name, desc in NODE_DESC.items() if desc]
    if nodes_with_desc:
        # Qdrant point id 只接受无符号整数 / UUID,中文 name 不能直接做 id;
        # 用枚举整数做 id,真正的 name 放进 payload。
        ids = list(range(len(nodes_with_desc)))
        texts = [desc for _, desc in nodes_with_desc]
        vecs = embedder.encode(texts)
        payloads = [{"name": name, "text": desc} for name, desc in nodes_with_desc]
        vec_store.upsert(ids, vecs, payloads)
        print(f"      → 写入 {len(ids)} 个节点向量(dim={embedder.dim})")

    graph.close()
    print("\n" + "=" * 60)
    print("建图完成。可在 http://localhost:7474 查看 Neo4j 图谱。")
    print("默认 query:", DEFAULT_QUERY)
    print("默认 seeds:", DEFAULT_SEEDS)
    print("=" * 60)


if __name__ == "__main__":
    main()
