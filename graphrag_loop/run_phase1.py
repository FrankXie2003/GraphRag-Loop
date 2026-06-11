"""Phase 1 端到端检索 —— 检索循环接到真 Neo4j + bge-reranker。

用法(在 graphrag_loop/ 目录下,venv 已激活):
    python run_phase1.py
    python run_phase1.py --query "马云毕业于哪所学校" --answer-hint 杭州师范大学

前提:已跑过 python ingestion/ingest_toy_data.py,Neo4j 里有玩具图。

设计:loop/controller.py 的控制流与 Phase 0 demo 验证过的完全一致,
这里只注入真组件——graph=GraphStore(Neo4j)、ranker=RerankerScorer(bge-reranker)。
控制流一行未改,证明 demo 逻辑可无缝迁移到真后端。

唯一对 demo 的依赖是 demo.toy_data(纯数据:默认 query/seeds),不导入 demo 的任何逻辑,
因此不会与 graphrag_loop 的 config 包撞名。
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_this = Path(__file__).resolve().parent
_root = _this.parent
sys.path.insert(0, str(_this))   # graphrag_loop:loop / storage / reflection / config
sys.path.insert(0, str(_root))   # 仓库根:仅为 import demo.toy_data(纯数据)

import argparse
from dataclasses import dataclass

from demo.toy_data import DEFAULT_QUERY, DEFAULT_SEEDS  # 纯数据,无逻辑
from loop.controller import run_loop
from loop.decision import RuleBasedPolicy
from storage.graph_store import GraphStore
from storage.vector_store import VectorStore
from reflection.isrel import get_scorer
from models.embedding import get_embedder
from entry.hybrid_recall import hybrid_recall
from utils.trace import Tracer, build_answer
from config.settings import PARAMS


@dataclass
class RunConfig:
    D_MAX: int
    BEAM_WIDTH: int
    TAU_REL: float
    THETA_STOP: float
    PRUNE_MODE: str = "ratio"
    PRUNE_RATIO: float = 0.92
    PRUNE_FLOOR: float = 0.30


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1:检索循环 + 真 Neo4j")
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--seeds", nargs="+", default=None,
                   help="手动指定种子;不给则由入口召回从 query 自动找种子")
    p.add_argument("--answer-hint", default=None)
    p.add_argument("--tau", type=float, default=PARAMS.TAU_REL)
    p.add_argument("--beam", type=int, default=PARAMS.BEAM_WIDTH)
    p.add_argument("--dmax", type=int, default=PARAMS.D_MAX)
    p.add_argument("--theta", type=float, default=PARAMS.THETA_STOP)
    p.add_argument("--prune-mode", default=PARAMS.PRUNE_MODE,
                   choices=["ratio", "absolute", "gap"])
    p.add_argument("--ratio", type=float, default=PARAMS.PRUNE_RATIO)
    p.add_argument("--floor", type=float, default=PARAMS.PRUNE_FLOOR)
    return p.parse_args()


def resolve_seeds(args, embedder, vec_store):
    """有 --seeds 用手写的;否则用入口召回从 query 自动找种子。"""
    if args.seeds:
        print(f"种子来源: 手动指定 {args.seeds}")
        return args.seeds
    seeds_scored = hybrid_recall(args.query, embedder, vec_store)
    print("种子来源: 入口召回(hybrid_recall)")
    for name, w in seeds_scored:
        print(f"    {name}  (weight={w:.3f})")
    return [name for name, _ in seeds_scored]


def main():
    args = parse_args()
    config = RunConfig(D_MAX=args.dmax, BEAM_WIDTH=args.beam,
                       TAU_REL=args.tau, THETA_STOP=args.theta,
                       PRUNE_MODE=args.prune_mode, PRUNE_RATIO=args.ratio,
                       PRUNE_FLOOR=args.floor)

    graph = GraphStore()    # get_neighbors / describe 契约同 demo/MemoryGraph
    ranker = get_scorer()   # score_batch 契约同 demo/WordOverlapRanker

    # 入口召回:把自然语言 query 转成起始种子(替换手写 seeds)
    embedder = get_embedder()
    vec_store = VectorStore()
    seeds = resolve_seeds(args, embedder, vec_store)

    policy = RuleBasedPolicy(theta_stop=config.THETA_STOP, d_max=config.D_MAX,
                             answer_hint=args.answer_hint)
    tracer = Tracer(tau_rel=config.TAU_REL, beam_width=config.BEAM_WIDTH)

    tracer.header(args.query, seeds, config)
    print(f"打分器: {ranker.name}")
    print(f"剪枝模式: {config.PRUNE_MODE}"
          + (f"(ratio={config.PRUNE_RATIO}, floor={config.PRUNE_FLOOR})"
             if config.PRUNE_MODE != "absolute" else f"(τ_rel={config.TAU_REL})"))
    print(f"图后端: Neo4j({graph.verify()})")

    state = run_loop(graph, ranker, policy, args.query, seeds, config, tracer)

    tracer.footer(state)
    print(build_answer(args.query, state))
    graph.close()


if __name__ == "__main__":
    main()
