"""Neo4j 封装 —— 检索循环的图访问层。

设计要点:get_neighbors() 的返回契约与 demo/graph/memory_graph.py 完全一致
    [(relation, neighbor, direction), ...]
所以 demo 验证过的 loop 控制流接到这里时,一行都不用改。

额外提供:Cypher 执行、建图写入(upsert 节点/边)、GDS Personalized PageRank。
连接配置从 config.connections.NEO4J 读(来自 .env)。
"""

from contextlib import contextmanager

from config.connections import NEO4J


class GraphStore:
    def __init__(self):
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(
            NEO4J.uri, auth=(NEO4J.user, NEO4J.password)
        )

    def close(self):
        self._driver.close()

    @contextmanager
    def _session(self):
        session = self._driver.session()
        try:
            yield session
        finally:
            session.close()

    def verify(self):
        """连通性自检:返回服务端版本信息,连不上会抛异常。"""
        with self._session() as s:
            rec = s.run("CALL dbms.components() YIELD name, versions "
                        "RETURN name, versions[0] AS version").single()
            return f"{rec['name']} {rec['version']}"

    # ---------- 检索:供 loop 调用(契约同 demo) ----------

    def get_neighbors(self, node, undirected=True):
        """返回 node 的邻居:[(relation, neighbor, direction), ...]

        节点用 name 属性标识(与建图时一致)。direction 'out'/'in'。
        """
        cypher = """
        MATCH (n:Entity {name: $name})-[r]->(m:Entity)
        RETURN type(r) AS rel, m.name AS nbr, 'out' AS dir
        """
        if undirected:
            cypher += """
            UNION
            MATCH (n:Entity {name: $name})<-[r]-(m:Entity)
            RETURN type(r) AS rel, m.name AS nbr, 'in' AS dir
            """
        with self._session() as s:
            return [(rec["rel"], rec["nbr"], rec["dir"])
                    for rec in s.run(cypher, name=node)]

    def describe(self, node):
        """返回节点的文本描述(供 ranker 打分);无则退回名字。"""
        cypher = "MATCH (n:Entity {name:$name}) RETURN n.desc AS desc"
        with self._session() as s:
            rec = s.run(cypher, name=node).single()
            return (rec and rec["desc"]) or node

    # ---------- 建图:供 ingestion 调用 ----------

    def upsert_entity(self, name, desc=None):
        cypher = """
        MERGE (n:Entity {name:$name})
        SET n.desc = coalesce($desc, n.desc)
        """
        with self._session() as s:
            s.run(cypher, name=name, desc=desc)

    def upsert_relation(self, head, relation, tail):
        """建一条有类型的边。关系类型作为 Neo4j 关系 type(需是合法标识符)。"""
        # 用 APOC-free 写法:关系类型不能参数化,故用 f-string 但先做白名单清洗
        safe_rel = "".join(c for c in relation if c.isalnum() or c == "_") or "REL"
        cypher = f"""
        MERGE (h:Entity {{name:$head}})
        MERGE (t:Entity {{name:$tail}})
        MERGE (h)-[:`{safe_rel}`]->(t)
        """
        with self._session() as s:
            s.run(cypher, head=head, tail=tail)

    def clear(self):
        """清空图(开发期重建用)。"""
        with self._session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    # ---------- PPR:供 entry/ppr.py 调用 ----------

    def personalized_pagerank(self, seed_nodes, alpha=0.5, top_k=50):
        """用 GDS 跑 Personalized PageRank,以 seed_nodes 为 restart 源。

        返回 [(name, score), ...] 按分降序。需要 Neo4j 装了 GDS 插件
        (docker-compose 已配)。这里用 stream 模式,跑完即弃投影图。
        """
        cypher = """
        MATCH (src:Entity) WHERE src.name IN $seeds
        WITH collect(src) AS sources
        CALL gds.pageRank.stream({
            nodeProjection: 'Entity',
            relationshipProjection: '*',
            sourceNodes: sources,
            dampingFactor: $alpha
        })
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).name AS name, score
        ORDER BY score DESC LIMIT $top_k
        """
        with self._session() as s:
            return [(rec["name"], rec["score"])
                    for rec in s.run(cypher, seeds=list(seed_nodes),
                                     alpha=alpha, top_k=top_k)]
