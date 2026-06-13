"""L1 入口决策(Self-RAG Retrieve token)—— 判断 query 是否需要走图谱检索。

不是所有问题都该查图:
  - 需要检索:涉及具体人物/事件/关系的事实性问题(图里有答案)。
  - 不需检索:寒暄、常识、纯创作指令、与本图谱无关的问题 → 直接交 LLM 生成。

返回 Retrieve 枚举。用小模型(高频、判断简单),走 ROUTING['L1_route_decision']=SMALL。
保守策略:拿不准就判 YES(宁可多查一次,也别漏掉图里的证据)。
"""

from models.llm import LLMClient
from reflection.tokens import Retrieve


_SYSTEM = "你是检索路由助手,判断一个问题是否需要查询知识图谱才能回答。"

_PROMPT = """知识图谱内容:中文叙事文本(如《红楼梦》)里的人物、事件、人物关系。

判断下面的问题是否需要**查询该知识图谱**才能准确回答:
- 若问题涉及具体人物、事件、人物关系、情节(图谱里可能有答案)→ 回答 YES
- 若是寒暄、通用常识、与该文本无关的问题、或纯创作指令(无需查证)→ 回答 NO
- 拿不准时回答 YES。

问题:{query}

只回答 YES 或 NO,不要解释。"""


def need_retrieve(query, llm=None):
    """返回 Retrieve.YES / Retrieve.NO。"""
    llm = llm or LLMClient()
    out = llm.chat(_PROMPT.format(query=query),
                   stage="L1_route_decision", system=_SYSTEM).strip().upper()
    # 保守:只有明确 NO 才不检索
    if out.startswith("NO"):
        return Retrieve.NO
    return Retrieve.YES
