"""内存知识图谱。

用纯 Python dict 实现(不强依赖 networkx,保证零依赖可跑)。
职责与 graphrag_loop/storage/graph_store.py(Neo4j 版)保持相同接口契约:
    - get_neighbors(node): 返回从 node 出发的 (relation, tail) 边
完整架构里只需把这个类换成 Neo4j 实现,loop 代码不动。

路径记忆(已访问边去重)不放在图里,而是放在循环状态 LoopState 中,
因为"访问过没有"是单次检索的状态,不是图的固有属性。
"""

from collections import defaultdict


class MemoryGraph:
    def __init__(self, triples, node_desc=None):
        # 邻接表:head -> [(relation, tail), ...]
        self._adj = defaultdict(list)
        # 反向邻接表:tail -> [(relation, head), ...],支持无向扩展(关系可双向走)
        self._radj = defaultdict(list)
        for head, relation, tail in triples:
            self._adj[head].append((relation, tail))
            self._radj[tail].append((relation, head))
        self.node_desc = node_desc or {}

    def get_neighbors(self, node, undirected=True):
        """返回 node 的邻居边列表:[(relation, neighbor, direction), ...]

        undirected=True 时同时沿出边和入边扩展(知识图谱里关系常需双向推理,
        比如 '杭州 位于 浙江省' 也意味着从浙江省能反查到杭州)。
        direction 用 'out'/'in' 标记,仅用于轨迹展示和边去重的方向区分。
        """
        edges = [(rel, tail, "out") for rel, tail in self._adj.get(node, [])]
        if undirected:
            edges += [(rel, head, "in") for rel, head in self._radj.get(node, [])]
        return edges

    def describe(self, node):
        """返回节点文本描述,供 ranker 打分;无描述则退回节点名本身。"""
        return self.node_desc.get(node, node)

    def has_node(self, node):
        return node in self._adj or node in self._radj or node in self.node_desc
