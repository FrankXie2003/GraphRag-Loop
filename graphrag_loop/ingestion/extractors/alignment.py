"""实体/关系对齐 + 关系矛盾消解 —— 在三元组写入图谱前清洗。

处理三类问题(对应 PHASE1.md 暴露的真实噪声):
  1. 别名实体:宝玉/贾宝玉、士隐/甄士隐 → 合并为规范名(LLM 分组)
  2. 同义关系:邀/邀请、寄居于葫芦庙/寄居于庙中 → 归一到规范关系词(LLM 分组)
  3. 矛盾边:A-[女儿]->B 和 B-[女儿]->A 同时存在(互为女儿)→ 消解

重要边界:对齐**解决不了关系方向抽错**本身——那是抽取(LLM 能力 + prompt)的问题。
对齐能做的是:合并同义、**检测**矛盾边、在有原文出处(provenance)时让 LLM 按原文裁决方向。
没有出处时,矛盾边采取保守策略(都丢弃并报告),宁可少一条也不留错的。
"""

import json
import re

from models.llm import LLMClient


# 非对称关系:同义反向即矛盾(互为女儿/父亲不可能)。对称关系(如"相邻")反向不算矛盾。
_ASYMMETRIC_HINT = {"女儿", "儿子", "父亲", "母亲", "妻子", "丈夫", "岳丈",
                    "祖父", "祖母", "主人", "仆人", "师傅", "徒弟"}


def _parse_json(raw, default):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    s = raw.find("{") if default == {} else raw.find("[")
    e = raw.rfind("}") if default == {} else raw.rfind("]")
    if s == -1 or e == -1:
        return default
    try:
        return json.loads(raw[s:e + 1])
    except json.JSONDecodeError:
        return default


def canonicalize(items, kind, llm):
    """让 LLM 把一批名称(实体或关系)分组,返回 {原名: 规范名}。

    kind: '实体' 或 '关系',用于 prompt 措辞。未被合并的项映射到自身。
    """
    if not items:
        return {}
    listing = "\n".join(f"- {x}" for x in sorted(set(items)))
    prompt = f"""下面是从同一文本抽取出的{kind}名称列表,其中有些是**同一对象/同一含义的不同写法**。
请把它们分组,每组选一个最规范、最完整的名称作为代表。

{kind}列表:
{listing}

只输出 JSON 对象,键是原名称,值是该组的规范代表名(没有同义项的就映射到自己)。
不要把**不同**的对象错误合并(如"封氏"和"封肃"是不同的人,不要合并)。
只输出 JSON,不要解释。示例格式:{{"宝玉":"贾宝玉","贾宝玉":"贾宝玉","邀":"邀请","邀请":"邀请"}}"""
    out = llm.chat(prompt, stage="ingest_align",
                   system=f"你是知识图谱{kind}对齐专家。")
    mapping = _parse_json(out, {})
    # 兜底:列表里没被 LLM 覆盖的项映射到自身
    for x in items:
        mapping.setdefault(x, x)
    return mapping


def resolve_contradictions(triples, llm, provenance=None):
    """消解互为非对称关系的矛盾边。

    triples: [(h, r, t)]
    provenance: {(h,r,t): 原文块文本},有则让 LLM 按原文裁决方向;无则保守丢弃。
    返回 (cleaned_triples, report)。
    """
    triple_set = set(triples)
    dropped, kept_overrides = [], []
    seen_pairs = set()
    result = []

    for (h, r, t) in triples:
        if (h, r, t) in seen_pairs:
            continue
        rev = (t, r, h)
        is_asym = r in _ASYMMETRIC_HINT
        if is_asym and rev in triple_set and rev != (h, r, t):
            # 矛盾:A-[r]->B 与 B-[r]->A 同时存在
            seen_pairs.add((h, r, t))
            seen_pairs.add(rev)
            winner = None
            if provenance:
                ctx = provenance.get((h, r, t)) or provenance.get(rev)
                if ctx:
                    winner = _judge_direction(h, r, t, ctx, llm)
            if winner:
                result.append(winner)
                kept_overrides.append((winner, "LLM按原文裁决"))
            else:
                dropped.append(((h, r, t), rev))  # 无据可依,两条都丢
        else:
            result.append((h, r, t))

    report = {"contradictions_dropped": dropped, "resolved": kept_overrides}
    return result, report


def _judge_direction(a, r, b, context, llm):
    """给定原文,让 LLM 裁决 a-[r]->b 还是 b-[r]->a 正确,返回正确的三元组或 None。"""
    prompt = f"""根据原文判断下面哪个关系方向正确(读作"head 的 {r} 是 tail"):
A. {a} 的 {r} 是 {b}
B. {b} 的 {r} 是 {a}

原文片段:
{context[:300]}

只回复 A 或 B;都不对回复 N。"""
    out = llm.chat(prompt, stage="ingest_align").strip().upper()
    if out.startswith("A"):
        return (a, r, b)
    if out.startswith("B"):
        return (b, r, a)
    return None


def align(triples, llm=None, provenance=None):
    """对齐总流程:实体归一 → 关系归一 → 矛盾消解 → 去重。

    返回 (aligned_triples, report)。
    """
    llm = llm or LLMClient()

    entities = [h for h, _, _ in triples] + [t for _, _, t in triples]
    relations = [r for _, r, _ in triples]

    ent_map = canonicalize(entities, "实体", llm)
    rel_map = canonicalize(relations, "关系", llm)

    # 应用映射(同时把 provenance 的键也映射过去)
    mapped = []
    mapped_prov = {}
    for (h, r, t) in triples:
        nh, nr, nt = ent_map.get(h, h), rel_map.get(r, r), ent_map.get(t, t)
        if nh == nt:  # 归一后变自环,丢弃
            continue
        mapped.append((nh, nr, nt))
        if provenance and (h, r, t) in provenance:
            mapped_prov[(nh, nr, nt)] = provenance[(h, r, t)]

    # 矛盾消解
    resolved, report = resolve_contradictions(mapped, llm,
                                              provenance=mapped_prov or None)

    # 去重(保序)
    aligned = list(dict.fromkeys(resolved))

    report.update({
        "entity_merges": {k: v for k, v in ent_map.items() if k != v},
        "relation_merges": {k: v for k, v in rel_map.items() if k != v},
        "before": len(triples),
        "after": len(aligned),
    })
    return aligned, report
