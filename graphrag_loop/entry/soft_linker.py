"""L1b 软链接 —— 把 query 里的实体提及(mention)挂到图中节点。

为什么"软":不要求 mention 与节点名精确相等。先用 LLM 抽出 query 里的实体词,
再用向量相似度为每个 mention 找图中最接近的若干候选节点(而非唯一匹配)。
这样即使 query 写"宝玉"而图里是"贾宝玉"、或用户拼写不规范,也能挂上,
避免精确实体链接的单点故障(HippoRAG 思路)。

依赖:embedder(把 mention 向量化)、vec_store(图节点向量已在建图时入库)。
"""

import json
import re

from models.llm import LLMClient


_NER_SYSTEM = "你是实体识别助手,从问题中抽取出其中提到的实体名词。"
_NER_PROMPT = """从下面的问题中抽取出所有**实体提及**(人名、地名、物品名等具体名词),
不要抽取疑问词、动词、形容词。只输出 JSON 字符串数组,不要解释。

示例:
问题:甄士隐送给贾雨村什么东西?
输出:["甄士隐","贾雨村"]

问题:{query}
"""


def extract_mentions(query, llm=None):
    """用 LLM 从 query 抽实体提及,返回 [str]。失败则退回空(交给整句向量召回兜底)。"""
    llm = llm or LLMClient()
    out = llm.chat(_NER_PROMPT.format(query=query),
                   stage="L1b_soft_link", system=_NER_SYSTEM)
    out = re.sub(r"^```(?:json)?|```$", "", out.strip(), flags=re.MULTILINE).strip()
    s, e = out.find("["), out.rfind("]")
    if s == -1 or e == -1:
        return []
    try:
        items = json.loads(out[s:e + 1])
        return [str(x).strip() for x in items if str(x).strip()]
    except json.JSONDecodeError:
        return []


def link_mention(mention, embedder, vec_store, top_k=3):
    """把单个 mention 向量匹配到图节点,返回 [(name, score), ...]。"""
    vec = embedder.encode(mention)
    hits = vec_store.search(vec, top_k=top_k)
    return [(p.get("name"), score) for p, score in hits if p.get("name")]


def soft_link(query, embedder, vec_store, llm=None, top_k=3):
    """对 query 里每个 mention 做软链接,汇总候选节点 [(name, score), ...]。"""
    mentions = extract_mentions(query, llm=llm)
    candidates = []
    for m in mentions:
        candidates.extend(link_mention(m, embedder, vec_store, top_k=top_k))
    return candidates
