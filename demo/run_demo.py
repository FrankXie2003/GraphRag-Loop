"""Phase 0 demo 入口。

加载玩具图 → 初始化种子 → 跑一次完整 Agent 检索循环 → 打印轨迹与答案。
零依赖即可运行;命令行可覆盖超参。

用法:
    python run_demo.py
    python run_demo.py --query "马云毕业于哪里" --seeds 阿里巴巴 --beam 3 --dmax 4
    python run_demo.py --tau 0.5            # 调高剪枝阈值,看更多分支被砍
    python run_demo.py --embedding          # 若装了 sentence-transformers,用句向量打分
"""

import argparse
import sys

# Windows 终端默认 GBK,强制 stdout 用 UTF-8,避免中文乱码 / Unicode 报错。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import DEFAULT
from toy_data import TRIPLES, NODE_DESC, DEFAULT_QUERY, DEFAULT_SEEDS
from graph.memory_graph import MemoryGraph
from retrieval.ranker import get_ranker
from agent.policy import RuleBasedPolicy
from agent.loop import run_loop
from utils.trace import Tracer, build_answer


def parse_args():
    p = argparse.ArgumentParser(description="GraphRAG-Loop Phase 0 Demo")
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--seeds", nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--dmax", type=int, default=DEFAULT.D_MAX)
    p.add_argument("--beam", type=int, default=DEFAULT.BEAM_WIDTH)
    p.add_argument("--tau", type=float, default=DEFAULT.TAU_REL)
    p.add_argument("--theta", type=float, default=DEFAULT.THETA_STOP)
    p.add_argument("--embedding", action="store_true",
                   help="用句向量打分(需 sentence-transformers,否则自动回退)")
    p.add_argument("--answer-hint", default="浙江省",
                   help="可选:证据触达该节点时提升置信度,仅让终止更直观")
    return p.parse_args()


def main():
    args = parse_args()

    # 用命令行覆盖默认超参
    config = DEFAULT
    config.D_MAX = args.dmax
    config.BEAM_WIDTH = args.beam
    config.TAU_REL = args.tau
    config.THETA_STOP = args.theta

    graph = MemoryGraph(TRIPLES, NODE_DESC)
    ranker = get_ranker(use_embedding=args.embedding)
    policy = RuleBasedPolicy(theta_stop=config.THETA_STOP,
                             d_max=config.D_MAX,
                             answer_hint=args.answer_hint)
    tracer = Tracer(tau_rel=config.TAU_REL, beam_width=config.BEAM_WIDTH)

    tracer.header(args.query, args.seeds, config)
    print(f"打分器: {ranker.name}")

    state = run_loop(graph, ranker, policy, args.query, args.seeds, config, tracer)

    tracer.footer(state)
    print(build_answer(args.query, state))


if __name__ == "__main__":
    main()
