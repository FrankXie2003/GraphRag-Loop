"""L3 生成 —— 基于证据子图生成带 inline 引用的答案。

Phase 2 双层 schema 下,证据子图天然分两层:
  Entity 集合:骨架(回答里出现的人物/地点)
  Event 列表:情节(回答的具体内容来源,带 content)
  → 引用基准用 Event(content 是情节,最适合作证据);Entity 作为骨架辅助。

输出结构化结果(为后续 IsSup 分段验证 + refeed 缺口提取铺路):
  GeneratedAnswer.text         带 [E1] 这种 inline 引用编号的全文
  GeneratedAnswer.segments     按句切分的段落 [{text, cited_ids}]
  GeneratedAnswer.evidence_map {id: {kind, name, content}}  引用 ID → 证据来源

Prompt 约束:
  1. 只用证据里的内容,不编造
  2. 每个事实陈述都要在末尾标注引用编号,如 "甄士隐赠银 50 两 [E1]"
  3. 不知道就直接说"原文未提",不要圆话
"""

from dataclasses import dataclass, field
import re

from models.llm import LLMClient


_SYSTEM = ("你是知识图谱问答助手。严格基于给定的证据回答问题,"
           "不允许添加证据以外的事实。每个事实陈述必须在句末标注引用编号 [E1]/[E2]。"
           "证据里没有的就直接说'原文未提',不要推测、不要编造。")

_PROMPT_TMPL = """问题:{query}

【证据】
{evidence_block}

请回答问题,要求:
1. 只用证据里的内容,**绝不**添加证据外的事实(包括常识)。
2. 每个事实陈述末尾标注引用编号(如 [E1]、[E2]),引用编号严格对应上面证据的 E# 标号。
3. 一个陈述如果由多个证据共同支撑,可以同时引用 [E1][E2]。
4. 答案简洁,直接回答问题,不要重复问题。
5. 证据不足以回答时,直接说"原文未提供足够信息回答此问题"。
"""


@dataclass
class GeneratedAnswer:
    text: str                                  # 完整答案(含 [E#] 引用)
    segments: list = field(default_factory=list)   # [{text, cited_ids:[int]}]
    evidence_map: dict = field(default_factory=dict)  # {id: {'kind','name','content'}}
    is_honest_refusal: bool = False            # 诚实拒答(原文未提),IsSup 应跳过


def _build_evidence_block(state):
    """把 state 里的证据组织成可引用的 E1/E2/... 文本块,返回 (block, evidence_map)。

    Event 优先(因为有 content,信息密度高),按 state 中已有顺序(即 NEXT 时序)排;
    Entity 集合作为辅助附在末尾(只列名)。
    """
    evidence_map = {}
    lines = []
    eid = 0
    for ev in state.evidence_events:
        eid += 1
        evidence_map[eid] = {"kind": "event", "name": ev["name"],
                             "content": ev.get("content") or ""}
        c = (ev.get("content") or "").replace("\n", " ")
        lines.append(f"[E{eid}] 事件【{ev['name']}】:{c}")

    if state.evidence_entities:
        # Entity 单独一组,继续 E# 编号(让 LLM 也能引用人物存在性)
        ents = sorted(state.evidence_entities)
        eid += 1
        evidence_map[eid] = {"kind": "entities", "name": "实体集合",
                             "content": ", ".join(ents)}
        lines.append(f"[E{eid}] 涉及实体:{', '.join(ents)}")

    return "\n".join(lines) if lines else "(无证据)", evidence_map


_CITE_RE = re.compile(r"\[E(\d+)\]")

# 诚实拒答的特征短语:命中其一即视为诚实拒答(IsSup 跳过,不该回扩)。
# 这些短语必须和 generator prompt 第 5 条要求保持一致。
_REFUSAL_PATTERNS = [
    "原文未提",
    "原文未提供",
    "证据不足",
    "无法回答",
    "未提到",
    "未提及",
    "没有相关信息",
    "不知道",
    "找不到",
    "记录中没有",
]


def _is_honest_refusal(text):
    """判断是否为诚实拒答。简单规则:答案极短或命中拒答短语。"""
    clean = _CITE_RE.sub("", text).strip()
    if any(p in clean for p in _REFUSAL_PATTERNS):
        return True
    return False


def _split_segments(text):
    """把答案按中文句号/问号/感叹号 + 换行切成段,每段抽出引用编号集合。"""
    raw_segs = re.split(r"(?<=[。!?\n])", text)
    segments = []
    for s in raw_segs:
        s = s.strip()
        if not s:
            continue
        ids = sorted({int(m) for m in _CITE_RE.findall(s)})
        # 去掉引用标记后的纯文本(供 IsSup 验证用)
        clean = _CITE_RE.sub("", s).strip()
        segments.append({"text": s, "clean": clean, "cited_ids": ids})
    return segments


def generate(query, state, llm=None):
    """生成答案,返回 GeneratedAnswer。

    若 state 没有任何证据(连一个事件都没找到),仍会调 LLM 让它说"原文未提",
    保持下游处理一致。诚实拒答(answer 含"原文未提"等)会被打上
    is_honest_refusal=True,IsSup 阶段直接跳过、不触发回扩。
    """
    llm = llm or LLMClient()
    block, evidence_map = _build_evidence_block(state)
    prompt = _PROMPT_TMPL.format(query=query, evidence_block=block)
    text = llm.chat(prompt, stage="L3_generate", system=_SYSTEM).strip()
    segments = _split_segments(text)
    return GeneratedAnswer(
        text=text, segments=segments, evidence_map=evidence_map,
        is_honest_refusal=_is_honest_refusal(text),
    )
