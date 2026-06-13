"""单独验证入口召回 —— 看自然语言 query 如何变成图中的起始种子。

用法:
    python verify_entry.py
    python verify_entry.py --query "英莲的父亲是谁"

只跑入口召回这一步(不跑后续检索循环),便于单独理解 query→种子 的过程:
  ① LLM 从 query 抽实体提及(mention)
  ② 整句向量召回 + 每个 mention 软链接到图节点
  ③ 去重赋权,输出种子
"""

import sys
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_this = Path(__file__).resolve().parent
sys.path.insert(0, str(_this))

from models.embedding import get_embedder
from models.llm import LLMClient
from storage.vector_store import VectorStore
from entry.soft_linker import extract_mentions
from entry.hybrid_recall import hybrid_recall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="甄士隐送给贾雨村什么东西")
    args = p.parse_args()

    print("=" * 60)
    print(f"Query: {args.query}")
    print("=" * 60)

    llm = LLMClient()
    embedder = get_embedder()
    vec = VectorStore()

    # ① LLM 抽实体提及
    mentions = extract_mentions(args.query, llm=llm)
    print(f"\n① LLM 抽出的实体提及(mention): {mentions}")

    # ②③ 混合召回(双型分路,Phase 2)
    print("\n②③ 双型分路混合召回 → 起始种子(按权重降序):")
    seeds = hybrid_recall(args.query, embedder, vec, llm=llm)
    for name, w, ntype in seeds:
        print(f"     [{ntype:6s}] {name}  (weight={w:.3f})")

    print("\n这些种子就是检索循环 BFS 的起点。")


if __name__ == "__main__":
    main()
