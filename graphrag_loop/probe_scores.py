"""分数分布探针 —— 在真实 query 上看 reranker 的分数分布,辅助校准 τ_rel / PRUNE_*。

为什么需要:Phase 2 引入 Event 节点(content 长,reranker 给分偏高),分数分布
和 Phase 1 不同。日志里能看到候选分数挤在 0.50-0.55 区间,beam 几乎没空间分层。

用法(venv 激活、Neo4j 起来):
    python probe_scores.py
    python probe_scores.py --query "甄士隐家中遭遇了哪些灾难"

输出:对几个种子做 1 跳邻居展开 + reranker 打分,打印分数直方图,帮你判断:
  - 信号(相关项)和噪声(无关项)之间是否有可见 gap
  - 当前 τ_rel=0.52 是否合适(看噪声项分数中位数)
  - PRUNE_RATIO=0.92 在新分布下是否还能切干净

不修改图,只读不写。
"""

import sys
import argparse
from pathlib import Path
from statistics import median, mean

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_this = Path(__file__).resolve().parent
sys.path.insert(0, str(_this))

from storage.graph_store import GraphStore
from storage.vector_store import VectorStore
from reflection.isrel import get_scorer
from models.embedding import get_embedder
from entry.hybrid_recall import hybrid_recall


def histogram(scores, buckets=10):
    """简单文本直方图。"""
    if not scores:
        return ""
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-6:
        return f"  全部分数 ≈ {lo:.3f}"
    width = (hi - lo) / buckets
    counts = [0] * buckets
    for s in scores:
        idx = min(int((s - lo) / width), buckets - 1)
        counts[idx] += 1
    max_c = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        lower = lo + i * width
        bar = "█" * int(40 * c / max_c)
        lines.append(f"  [{lower:.3f}-{lower+width:.3f}] {bar} {c}")
    return "\n".join(lines)


def probe(query, top_seeds=3):
    print("=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    graph = GraphStore()
    ranker = get_scorer()
    embedder = get_embedder()
    vec_store = VectorStore()

    seeds = hybrid_recall(query, embedder, vec_store)
    print(f"\n种子(前 {top_seeds}):")
    for name, w, ntype in seeds[:top_seeds]:
        print(f"  [{ntype:6s}] {name}  (weight={w:.3f})")

    for name, _, ntype in seeds[:top_seeds]:
        print(f"\n--- 从 {name}({ntype}) 出发的 1 跳邻居打分 ---")
        nbrs = graph.get_neighbors(name)
        if not nbrs:
            print("  (无邻居)")
            continue
        # 拼成 reranker 的输入文本
        texts = [f"{rel} {graph.describe(nbr)}" for rel, nbr, _ in nbrs]
        scores = ranker.score_batch(query, texts)

        ranked = sorted(zip(nbrs, scores, texts), key=lambda x: x[1], reverse=True)
        print(f"  共 {len(nbrs)} 个邻居,分数统计:")
        print(f"    min={min(scores):.3f}  median={median(scores):.3f}  "
              f"max={max(scores):.3f}  mean={mean(scores):.3f}")
        print(histogram(scores))
        print(f"  Top 5:")
        for (rel, nbr, d), s, _ in ranked[:5]:
            arrow = "->" if d == "out" else "<-"
            print(f"    [{s:.3f}] {arrow}[{rel}] {nbr}")
        if len(ranked) > 5:
            print(f"  Bottom 3(噪声参考):")
            for (rel, nbr, d), s, _ in ranked[-3:]:
                arrow = "->" if d == "out" else "<-"
                print(f"    [{s:.3f}] {arrow}[{rel}] {nbr}")

        # ratio 模式生效情况:看 top × 0.92 这条线砍掉多少
        top_score = ranked[0][1]
        ratio_thresh = top_score * 0.92
        survivors = sum(1 for _, s, _ in ranked if s >= ratio_thresh)
        print(f"  PRUNE_RATIO=0.92 剪枝后保留:{survivors}/{len(ranked)} 项 "
              f"(阈值线={ratio_thresh:.3f})")

    graph.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="甄士隐送给贾雨村什么东西")
    p.add_argument("--top-seeds", type=int, default=3)
    args = p.parse_args()
    probe(args.query, top_seeds=args.top_seeds)


if __name__ == "__main__":
    main()
