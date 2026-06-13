"""L3 生成 —— 基于证据子图生成带 inline 引用的答案 + 原子断言数组(供逐条 IsSup)。

Phase 2 双层 schema 下,证据子图天然分两层:
  Entity 集合:骨架(回答里出现的人物/地点)
  Event 列表:情节(回答的具体内容来源,带 content)
  → 引用基准用 Event(content 是情节,最适合作证据);Entity 作为骨架辅助。

输出双轨结构(避免 #13 "长复合答案 vs 单条证据" 假阳性):
  GeneratedAnswer.text          自然语言答案,含 [E#] inline 引用 —— 给用户看
  GeneratedAnswer.atomic_claims [{text, cited_ids}, ...] 原子断言数组 —— IsSup 逐条验证
  GeneratedAnswer.segments      旧的按句切段(兜底:atomic_claims 解析失败时回退)
  GeneratedAnswer.evidence_map  {id: {kind,name,content}} 引用 ID → 证据来源

为什么要"原子断言":NLI 模型擅长"单 premise 蕴含单 hypothesis",不擅长"含多个独立事实的长 hypothesis"。
让 generator 直接拆好原子断言,逐条 NLI,从源头消除"取最大值"假阳性(see question&solution.md #13)。

Prompt 约束:
  1. 只用证据里的内容,不编造
  2. 同时输出 text(自然语言)和 atomic_claims(每条只含一个不可再分事实)
  3. 不知道就直接说"原文未提",不要圆话(诚实拒答短路 IsSup)
"""

from dataclasses import dataclass, field
import json
import re

from models.llm import LLMClient


_SYSTEM = ("你是知识图谱问答助手。严格基于给定的证据回答问题,不允许添加证据以外的事实。"
           "每个事实陈述必须在句末标注引用编号 [E1]/[E2]。"
           "证据里没有的就直接说'原文未提',不要推测、不要编造。")

_PROMPT_TMPL = """问题:{query}

【证据】
{evidence_block}

请回答问题,**同时输出两份**:
1. text:自然语言答案,带 inline 引用编号 [E1]/[E2](展示给用户看)。
2. atomic_claims:**原子断言数组**,把答案拆成不可再分的单一事实,每条只含一个断言 + 引用编号。
   - 例如答案"甄士隐资助贾雨村五十两[E1],并赠两套冬衣[E1]" → 拆成 2 条:
     [{{"text":"甄士隐资助贾雨村五十两白银","cited_ids":[1]}},
      {{"text":"甄士隐赠贾雨村两套冬衣","cited_ids":[1]}}]
   - 每条断言必须**自包含**(不要用"他/她"代词,要写出具体人名/事件名,以便单独验证)。
   - 一个断言可以对应多个引用([E1][E2])。

要求:
1. 只用证据里的内容,**绝不**添加证据外的事实(包括常识)。
2. 引用编号严格对应上面证据的 E# 标号。
3. 答案简洁,直接回答问题,不要重复问题。
4. 证据不足以回答时:text 写"原文未提供足够信息回答此问题",atomic_claims 设为空数组 []。

只输出 JSON,不要解释:
{{
  "text": "...",
  "atomic_claims": [{{"text":"...","cited_ids":[1,2]}}]
}}
"""


@dataclass
class GeneratedAnswer:
    text: str                                       # 完整答案(含 [E#] 引用,展示用)
    atomic_claims: list = field(default_factory=list)   # [{text, cited_ids}] 原子断言(IsSup 用)
    segments: list = field(default_factory=list)        # 兜底:按句切段
    evidence_map: dict = field(default_factory=dict)    # {id: {'kind','name','content'}}
    is_honest_refusal: bool = False                 # 诚实拒答(原文未提),IsSup 应跳过


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
    """判断是否为诚实拒答。简单规则:命中拒答短语。"""
    clean = _CITE_RE.sub("", text).strip()
    return any(p in clean for p in _REFUSAL_PATTERNS)


def _split_segments(text):
    """把答案按中文句号/问号/感叹号 + 换行切成段,每段抽出引用编号集合。

    兜底用:LLM JSON 解析失败时,把它当 atomic_claims 用。
    """
    raw_segs = re.split(r"(?<=[。!?\n])", text)
    segments = []
    for s in raw_segs:
        s = s.strip()
        if not s:
            continue
        ids = sorted({int(m) for m in _CITE_RE.findall(s)})
        clean = _CITE_RE.sub("", s).strip()
        segments.append({"text": s, "clean": clean, "cited_ids": ids})
    return segments


def _parse_generation(raw):
    """从 LLM 输出里抠出 {text, atomic_claims}。

    返回 (text, atomic_claims) —— 解析失败时 atomic_claims 为 None,调用方 fallback。
    """
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1:
        return raw, None  # 不是 JSON,raw 当 text
    try:
        obj = json.loads(raw[s:e + 1])
    except json.JSONDecodeError:
        return raw, None

    text = str(obj.get("text") or "").strip()
    if not text:
        return raw, None

    claims_raw = obj.get("atomic_claims") or []
    claims = []
    for c in claims_raw:
        if not isinstance(c, dict):
            continue
        ctext = str(c.get("text") or "").strip()
        if not ctext:
            continue
        ids = c.get("cited_ids") or []
        # 容错:cited_ids 可能是 ["1","2"] 或 [1,2]
        norm_ids = []
        for x in ids:
            try:
                norm_ids.append(int(x))
            except (TypeError, ValueError):
                continue
        # 也允许从 text 里再抠 [E#],与字段合并
        norm_ids += [int(m) for m in _CITE_RE.findall(ctext)]
        claims.append({
            "text": ctext,
            "clean": _CITE_RE.sub("", ctext).strip(),
            "cited_ids": sorted(set(norm_ids)),
        })

    return text, claims


def generate(query, state, llm=None):
    """生成答案,返回 GeneratedAnswer。

    流程:
      1. LLM 输出 {text, atomic_claims} 双轨结构
      2. 解析失败时 fallback:把 text 按句切作 atomic_claims(老逻辑)
      3. 诚实拒答打 is_honest_refusal=True,IsSup 直接 FULL 短路

    诚实拒答时 atomic_claims 应为空数组,verify_answer 看到 is_honest_refusal 短路。
    """
    llm = llm or LLMClient()
    block, evidence_map = _build_evidence_block(state)
    prompt = _PROMPT_TMPL.format(query=query, evidence_block=block)
    raw = llm.chat(prompt, stage="L3_generate", system=_SYSTEM).strip()

    text, claims = _parse_generation(raw)
    segments = _split_segments(text)
    if claims is None:
        # 兜底:用按句切段当 atomic_claims(诊断信息保留在日志里)
        print("[generator] atomic_claims 解析失败,fallback 到按句切段")
        claims = list(segments)

    return GeneratedAnswer(
        text=text,
        atomic_claims=claims,
        segments=segments,
        evidence_map=evidence_map,
        is_honest_refusal=_is_honest_refusal(text),
    )
