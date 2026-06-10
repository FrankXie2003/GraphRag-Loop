# networkx 包装的内存图。
# 职责:get_neighbors(node) 返回邻居及边;记录已访问边集合做去重(路径记忆)。
# 对应完整架构里 storage/graph_store.py 的 Neo4j 版本接口。
