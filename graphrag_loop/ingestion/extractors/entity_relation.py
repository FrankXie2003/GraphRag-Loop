"""LLM 抽取实体关系 —— 从文本块抽出 (head, relation, tail) 三元组。

设计要点:
  1. 强制 JSON 输出 + 容错解析(LLM 偶尔会包 markdown 代码块或多余文字)。
  2. prompt 约束:只抽明确的人物/地点/事件关系;实体用规范全名(缓解"宝玉/贾宝玉"
     这类指代不一致——这是轻量对齐,完整实体对齐留给 extractors/alignment.py)。
  3. few-shot 示例引导抽取风格。

不在这里做去重/合并;pipeline 汇总所有块的三元组后统一入库,MERGE 天然去重。
"""

import json
import re

from models.llm import LLMClient


_SYSTEM = "你是知识图谱构建专家,从中文文本中抽取实体关系三元组。"

_PROMPT_TMPL = """从下面的文本中抽取知识图谱三元组,用于人物关系/事件图谱。

要求:
1. 只抽取文本中**明确表达**的关系,不要推测或补全常识。
2. 实体用**规范全名**:同一对象在文中有多个称呼时统一(如"宝玉/贾宝玉"统一为"贾宝玉";"士隐/甄士隐"统一为"甄士隐")。
3. 关系用简短动词或名词短语(如"妻子""女儿""寄居于""赠银给")。
4. **不要**把"姓X名Y字Z"这类称谓拆成"姓""名""字"三条边——这些是属性不是关系,直接忽略。
5. **关系方向规则(重要)**:head 是关系的主体/拥有者,tail 是客体。
   亲属关系以**长辈/拥有者为 head**:如"英莲是甄士隐的女儿"→ 应抽 head=甄士隐, relation=女儿, tail=英莲
   (读作"甄士隐 的女儿 是 英莲")。**绝不能**同时抽出 A→B 和 B→A 这种互为亲属的矛盾边。
6. 不要抽代词(他/她/它)作为实体。
7. 抽完后**自检一遍**:每条三元组按"head 的 relation 是 tail"读一遍是否通顺、方向是否正确;
   去掉读不通或方向反的。
8. 只输出 JSON 数组,不要任何解释文字。格式:
[{{"head": "实体A", "relation": "关系", "tail": "实体B"}}]

示例:
文本:这甄士隐,姓甄名费字士隐,嫡妻封氏,情性贤淑。只有一女,乳名英莲。
输出:[{{"head":"甄士隐","relation":"妻子","tail":"封氏"}},{{"head":"甄士隐","relation":"女儿","tail":"英莲"}}]

现在抽取下面的文本:
{text}
"""


def _parse_json_array(raw):
    """从 LLM 输出里稳健地抠出 JSON 数组。"""
    # 去掉可能的 markdown 代码围栏
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    # 截取第一个 [ 到最后一个 ] 之间
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []


def extract_triples(text, llm=None):
    """对一段文本抽三元组,返回 [(head, relation, tail), ...]。"""
    llm = llm or LLMClient()
    out = llm.chat(_PROMPT_TMPL.format(text=text),
                   stage="ingest_extract", system=_SYSTEM)
    items = _parse_json_array(out)

    triples = []
    for it in items:
        if not isinstance(it, dict):
            continue
        h, r, t = it.get("head"), it.get("relation"), it.get("tail")
        if h and r and t and h != t:  # 丢弃自环和缺字段的
            triples.append((str(h).strip(), str(r).strip(), str(t).strip()))
    return triples
