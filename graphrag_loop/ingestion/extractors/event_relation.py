"""双层 schema 抽取器(Phase 2)—— 每个 chunk 抽 {主事件 + 参与人物 + 静态本体关系}。

区别于 Phase 1 的 entity_relation.py(只抽三元组),这里产出双层结构:
  - event:该 chunk 的 1 个主事件(name + content + participants),信息密度低时可为 None
  - relations:静态本体关系三元组(亲属/从属/称谓等)

判据(写进 prompt):
  静态关系 → 亲属、从属、称谓、地理归属等客观、非时间依赖的拓扑(如"女儿""丫鬟""位于")
  动态事件 → 有动作、情节、时间性的剧情段落(如"中秋夜资助贾雨村赴考")

事件边界对齐 chunk(一个语义段≈一个场景),不让 LLM 凭空划界。
"""

import json
import re

from models.llm import LLMClient


_SYSTEM = "你是知识图谱构建专家,从中文叙事文本中区分并抽取「静态关系」与「动态事件」。"

_PROMPT_TMPL = """从下面的文本中抽取知识图谱信息,区分两类:

【静态本体关系】客观、非时间依赖的拓扑关系:亲属(女儿/妻子)、从属(丫鬟/仆人)、称谓、地理归属(位于)等。
【动态事件】**只要文本里有"人物做了某事 / 发生了某情节"就算一个事件**,门槛要低:
  对话、相遇、赠予、做梦、吟诗、外出、患病、搬迁、失火、走失……都是事件。
  反例(不要错过):不要因为"这段也有静态关系"就只抽关系而漏掉其中的剧情。
  只有**纯抒情/纯议论/纯诗词、完全无情节**的段落才返回空数组。

  **粒度控制(重要)**:以"场景/情节单元"为粒度,把**连续的琐碎动作合并成一个有意义的事件**,
  不要把一个连贯场景拆成一堆碎动作。一段文本通常抽 **1-3 个事件**(信息极密集时最多 4 个)。
  反例(过碎,禁止):"甄士隐抱女欲入"+"僧大笑念偈"+"僧道消失"+"甄士隐悔未问"
  正确(合并为一个场景事件):"癞僧念谶语预言英莲命运后与道人飘然离去"(把上述连贯动作合并)。

要求:
1. 实体用规范全名(宝玉/贾宝玉 → 贾宝玉;士隐 → 甄士隐)。
2. 静态关系方向:长辈/拥有者为 head(如 head=甄士隐,relation=女儿,tail=英莲,读作"甄士隐的女儿是英莲")。
3. 每个事件:简短 name(≤15字)、content(50-180字,概括这段剧情)、participants(涉及人物名数组)。
4. 不抽代词,不把"姓X名Y字Z"拆成边。
5. 只输出 JSON,格式如下,不要解释:
{{
  "events": [
    {{"name": "事件名", "content": "情节概括", "participants": ["人物1","人物2"]}}
  ],
  "relations": [{{"head":"A","relation":"关系","tail":"B"}}]
}}
若本段确无任何情节,events 用空数组 []。

文本:
{text}
"""


def _parse_obj(raw):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1:
        return {}
    try:
        return json.loads(raw[s:e + 1])
    except json.JSONDecodeError:
        return {}


def extract_dual(text, llm=None):
    """抽取双层结构,返回 (events_list, relations_list)。

    events:    [{"name","content","participants":[...]}, ...](可为空)
    relations: [(head, relation, tail), ...]
    """
    llm = llm or LLMClient()
    out = llm.chat(_PROMPT_TMPL.format(text=text),
                   stage="ingest_extract", system=_SYSTEM)
    obj = _parse_obj(out)

    # 事件(多个,按文中顺序)
    events = []
    for ev in obj.get("events") or []:
        if not isinstance(ev, dict) or not ev.get("name") or not ev.get("content"):
            continue
        parts = ev.get("participants") or []
        events.append({
            "name": str(ev["name"]).strip(),
            "content": str(ev["content"]).strip(),
            "participants": [str(p).strip() for p in parts if str(p).strip()],
        })

    # 静态关系
    relations = []
    for it in obj.get("relations") or []:
        if not isinstance(it, dict):
            continue
        h, r, t = it.get("head"), it.get("relation"), it.get("tail")
        if h and r and t and h != t:
            relations.append((str(h).strip(), str(r).strip(), str(t).strip()))

    return events, relations
