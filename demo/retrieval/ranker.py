# 节点/路径打分。
# 句向量 cosine 相似度,或纯 mock 打分(便于单测确定性)。
# 输出 [0,1] 分数,供 beam 截断和 tau_rel 剪枝使用。
# 对应完整架构里 reflection/isrel.py 的 Cross-Encoder/NLI 版本。
