"""Phase 1 连通性自检 —— 装完依赖、起好 Docker 后第一个该跑的脚本。

逐条点亮三个外部依赖,任何一条失败都单独报出来,便于定位:
    python check_connections.py

通过后再进行建图 / 检索,避免在业务代码里 debug 基础设施问题。
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def check(name, fn):
    try:
        info = fn()
        print(f"[OK]   {name}: {info}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False


def _neo4j():
    from storage.graph_store import GraphStore
    g = GraphStore()
    try:
        return g.verify()
    finally:
        g.close()


def _qdrant():
    from storage.vector_store import VectorStore
    return VectorStore().verify()


def _dashscope():
    from models.llm import LLMClient
    out = LLMClient().chat("只回复两个字:正常", tier=None, stage="L1_route_decision")
    return f"LLM 回复 = {out!r}"


def _embedding():
    from models.embedding import get_embedder
    emb = get_embedder()
    v = emb.encode("连通性测试")
    return f"backend ok, dim={len(v)}"


def main():
    print("=" * 60)
    print("Phase 1 连通性自检")
    print("=" * 60)
    results = [
        check("Neo4j", _neo4j),
        check("Qdrant", _qdrant),
        check("DashScope LLM", _dashscope),
        check("Embedding", _embedding),
    ]
    print("=" * 60)
    ok = sum(results)
    print(f"通过 {ok}/{len(results)}")
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
