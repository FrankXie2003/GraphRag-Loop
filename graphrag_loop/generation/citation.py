"""引用对齐 —— 把答案段落映射回证据节点/passage。

generator.py 已经做了引用编号解析(每段含 cited_ids),本模块提供:
  resolve_citations(segment, evidence_map)  → 段落引用的真实证据列表
  evidences_for_segment(segment, evidence_map)  → 段落引用对应的证据文本数组(供 IsSup 验证)

这个分离的目的:让 IsSup / refeed 不直接读 evidence_map 内部结构,以后引用结构升级
(如加 chunk_id 回溯、按字符位置精确对齐)只动这里一个文件。
"""


def resolve_citations(segment, evidence_map):
    """段落 → 引用的证据条目列表 [{'kind','name','content'}, ...]。"""
    return [evidence_map[i] for i in segment.get("cited_ids", [])
            if i in evidence_map]


def evidences_for_segment(segment, evidence_map):
    """段落 → 引用对应的证据文本数组(供 IsSup verify 直接用)。

    若段落未带任何引用,返回空数组(IsSup 会判 NONE → 触发 refeed)。
    """
    return [c["content"] for c in resolve_citations(segment, evidence_map)
            if c.get("content")]
