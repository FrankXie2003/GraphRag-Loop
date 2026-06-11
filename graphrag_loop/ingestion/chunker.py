"""文本分块 —— 把长文档切成适合 LLM 抽取的小块。

中文按字数控制窗口(LLM 单次抽取的文本太长会漏实体、太短会割裂关系)。
按段落边界切,尽量不从句子中间断开;块之间留少量重叠,避免跨块关系丢失。
每块带 chunk_id 和来源信息,供后续引用对齐。
"""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: int
    text: str
    source: str = ""


def chunk_text(text, source="", max_chars=500, overlap=80):
    """按段落聚合到 max_chars 上限切块,相邻块重叠 overlap 字。

    max_chars=500:古典/叙事中文一块约几百字,信息量够抽出若干关系又不至于太长。
    overlap:防止"宝玉……他的母亲是王夫人"这种跨段关系被切断。
    """
    # 先按空行 / 段落切成自然段
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    buf = ""
    cid = 0
    for para in paragraphs:
        if buf and len(buf) + len(para) > max_chars:
            chunks.append(Chunk(cid, buf, source))
            cid += 1
            # 新块带上一块尾部 overlap 字做重叠
            buf = (buf[-overlap:] + para) if overlap else para
        else:
            buf = (buf + "\n" + para) if buf else para

    if buf:
        chunks.append(Chunk(cid, buf, source))
    return chunks
