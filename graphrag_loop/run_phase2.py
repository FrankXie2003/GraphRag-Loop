"""Phase 2 端到端入口 —— 完整闭环:入口召回 → 循环检索 → 生成 → IsSup → 定向回扩。

用法(在 graphrag_loop/ 目录下,venv 激活):
    python run_phase2.py
    python run_phase2.py --query "甄士隐为何家道中落"
    python run_phase2.py --policy rule       # 切回规则版决策做对比
    python run_phase2.py --no-refeed          # 关闭回扩,只做一次性生成

前提:已用 ingestion/ingest_graph_md_v2.py 建过双层图谱。

与 run_phase1.py 的区别:
  Phase 1 输出"模板答案"(只是把节点拼成字符串,不是真生成)。
  Phase 2 加了真 LLM 生成 + IsSup 段级验证 + 定向回扩(V_MAX 次兜底)。
"""

import sys
import argparse
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_this = Path(__file__).resolve().parent
_root = _this.parent
sys.path.insert(0, str(_this))
sys.path.insert(0, str(_root))

from loop.controller import run_loop
from loop.decision import RuleBasedPolicy, LLMDecisionPolicy
from storage.graph_store import GraphStore
from storage.vector_store import VectorStore
from reflection.isrel import get_scorer
from reflection.tokens import IsSup
from models.embedding import get_embedder
from models.llm import LLMClient
from entry.hybrid_recall import hybrid_recall
from entry.route_decision import need_retrieve
from reflection.tokens import Retrieve
from generation.generator import generate
from generation.refeed import refeed_until_supported, verify_answer
from utils.trace import Tracer
from utils.recorder import RunRecorder
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
    p = argparse.ArgumentParser(description="Phase 2:完整闭环(loop+生成+IsSup+回扩)")
    p.add_argument("--query", default="甄士隐送给贾雨村什么东西")
    p.add_argument("--seeds", nargs="+", default=None)
    p.add_argument("--policy", choices=["rule", "llm"], default="llm")
    p.add_argument("--no-refeed", action="store_true",
                   help="关闭定向回扩,只做一次性生成(对比用)")
    p.add_argument("--vmax", type=int, default=PARAMS.V_MAX,
                   help="定向回扩最大轮数")
    p.add_argument("--tau", type=float, default=PARAMS.TAU_REL)
    p.add_argument("--beam", type=int, default=PARAMS.BEAM_WIDTH)
    p.add_argument("--dmax", type=int, default=PARAMS.D_MAX)
    p.add_argument("--theta", type=float, default=PARAMS.THETA_STOP)
    return p.parse_args()


def resolve_seeds(args, embedder, vec_store):
    if args.seeds:
        print(f"种子来源: 手动指定 {args.seeds}")
        return args.seeds
    seeds_scored = hybrid_recall(args.query, embedder, vec_store)
    print("种子来源: 入口召回(双型分路)")
    for name, w, ntype in seeds_scored:
        print(f"    [{ntype:6s}] {name}  (weight={w:.3f})")
    return [name for name, _, _ in seeds_scored]


def print_verification(verified):
    """打印 IsSup 验证结果。"""
    print("\n" + "─" * 60)
    print(f"IsSup 段级验证(回扩 {verified.refeed_rounds} 轮):")
    icon = {IsSup.FULL: "✓ FULL ", IsSup.PARTIAL: "△ PART ", IsSup.NONE: "✗ NONE "}
    for j in verified.judgements:
        seg_text = j["segment"]["text"]
        if len(seg_text) > 70:
            seg_text = seg_text[:70] + "..."
        print(f"  {icon[j['token']]} (score={j['score']:.2f}) {seg_text}")
    if verified.fully_supported:
        print("→ 全部段落 FULL,答案完全支撑")
    else:
        print(f"→ 仍有非 FULL 段(已用尽回扩或无法提取缺口)")
    print("─" * 60)


def main():
    args = parse_args()
    config = RunConfig(D_MAX=args.dmax, BEAM_WIDTH=args.beam,
                       TAU_REL=args.tau, THETA_STOP=args.theta)

    rec = RunRecorder(args.query, log_dir=Path(_this) / "logs")
    rec.section("配置").kv(
        policy=args.policy, refeed=("off" if args.no_refeed else f"V_max={args.vmax}"),
        D_MAX=config.D_MAX, BEAM_WIDTH=config.BEAM_WIDTH,
        TAU_REL=config.TAU_REL, THETA_STOP=config.THETA_STOP,
        PRUNE=f"{config.PRUNE_MODE}(ratio={config.PRUNE_RATIO},floor={config.PRUNE_FLOOR})",
    )

    graph = GraphStore()
    ranker = get_scorer()
    embedder = get_embedder()
    vec_store = VectorStore()
    llm = LLMClient()

    # ===== L1 Retrieve:判断是否需要走图谱 =====
    print(f"\n[L1 Retrieve] 判断 query 是否需要查图谱...")
    retrieve_decision = need_retrieve(args.query, llm=llm)
    print(f"  → {retrieve_decision.value}")
    rec.section("L1 Retrieve").log(f"决策: `{retrieve_decision.value}`")
    if retrieve_decision == Retrieve.NO:
        print("\n→ 无需检索,直接 LLM 生成:")
        ans = llm.chat(args.query, stage="L3_generate")
        print(ans)
        rec.section("直接 LLM 生成(未走图)").code_block(ans)
        path = rec.save()
        print(f"\n轨迹已记录: {path}")
        graph.close()
        return

    seeds = resolve_seeds(args, embedder, vec_store)
    if args.policy == "llm":
        policy = LLMDecisionPolicy(theta_stop=config.THETA_STOP, d_max=config.D_MAX)
    else:
        policy = RuleBasedPolicy(theta_stop=config.THETA_STOP, d_max=config.D_MAX)

    tracer = Tracer(tau_rel=config.TAU_REL, beam_width=config.BEAM_WIDTH)

    # ===== 第一次检索循环 =====
    tracer.header(args.query, seeds, config)
    print(f"打分器: {ranker.name}  |  决策: {args.policy}  |  回扩: {'off' if args.no_refeed else f'V_max={args.vmax}'}")
    print(f"图后端: Neo4j({graph.verify()})")
    rec.section("入口召回种子").log("\n".join(f"- {s}" for s in seeds))

    state = run_loop(graph, ranker, policy, args.query, seeds, config, tracer)
    tracer.footer(state)
    rec.evidence_subgraph(state).log(
        f"\n**终止原因**: {state.stop_reason}  |  **置信度**: {state.confidence:.2f}")

    # ===== 生成 =====
    print("\n" + "=" * 70)
    print("生成答案中...")
    print("=" * 70)
    answer = generate(args.query, state, llm=llm)
    print(f"\n[初次生成]\n{answer.text}")
    rec.answer("初次生成", answer)

    # ===== IsSup 验证 + 定向回扩 =====
    if args.no_refeed:
        verified = verify_answer(answer, llm=llm)
        rec.verification(verified, label="单次(--no-refeed)")
        print_verification(verified)
    else:
        # 闭包:让 refeed 能继续跑 loop(共享 state、policy、tracer 配置)
        def run_loop_callback(sub_query, sub_seeds, sub_state):
            print(f"  [refeed] 跑 loop:种子={sub_seeds[:5]}{'...' if len(sub_seeds)>5 else ''}")
            return run_loop(graph, ranker, policy, sub_query, sub_seeds,
                            config, tracer=None)  # 子 loop 不打印轨迹,避免输出过多

        # 每次验证都进 recorder,这样能看到"初次→回扩→再验证"的完整轨迹
        def on_verify(round_idx, verified):
            label = "初次" if round_idx == 0 else f"回扩 {round_idx} 轮后"
            rec.verification(verified, label=label)

        verified = refeed_until_supported(
            args.query, state, answer,
            run_loop_fn=run_loop_callback, llm=llm, max_rounds=args.vmax,
            on_each_verification=on_verify,
        )
        # 回扩可能多次重新生成,verified.answer 是最终答案
        if verified.refeed_rounds > 0:
            print(f"\n[回扩后最终答案]\n{verified.answer.text}")
            rec.answer(f"回扩后最终答案(回扩 {verified.refeed_rounds} 轮)",
                       verified.answer)
            # 回扩后证据子图也要记
            rec.section("回扩后证据子图")
            rec.evidence_subgraph(state)
        print_verification(verified)

    path = rec.save()
    print(f"\n轨迹已记录: {path}")

    graph.close()


if __name__ == "__main__":
    main()
