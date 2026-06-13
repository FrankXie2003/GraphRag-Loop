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

        契约同 Phase 1(loop 不用改),但双层 schema 下放宽节点标签:
        既遍历 Entity-Entity 静态关系,也遍历 Entity-Event(PARTICIPATES)
        和 Event-Event(NEXT),让事件节点自然参与 BFS。
        节点用 name 属性标识。direction 'out'/'in'。
        """
        # 不限定 m 的标签(可为 Entity 或 Event);n 也按 name 匹配两类
        cypher = """
        MATCH (n {name: $name})-[r]->(m)
        WHERE (n:Entity OR n:Event) AND (m:Entity OR m:Event)
        RETURN type(r) AS rel, m.name AS nbr, 'out' AS dir
        """
        if undirected:
            cypher += """
            UNION
            MATCH (n {name: $name})<-[r]-(m)
            WHERE (n:Entity OR n:Event) AND (m:Entity OR m:Event)
            RETURN type(r) AS rel, m.name AS nbr, 'in' AS dir
            """
        with self._session() as s:
            return [(rec["rel"], rec["nbr"], rec["dir"])
                    for rec in s.run(cypher, name=node)]

    def describe(self, node):
        """返回节点的文本描述(供 ranker 打分);无则退回名字。

        Event 节点返回其 content(情节描述)—— 这正是事件节点化的价值:
        给 cross-encoder 打分的是有信息量的情节文本,而非干瘪的关系名。
        Entity 节点返回 desc。
        """
        cypher = """
        MATCH (n {name:$name}) WHERE n:Entity OR n:Event
        RETURN coalesce(n.content, n.desc) AS text
        """
        with self._session() as s:
            rec = s.run(cypher, name=node).single()
            return (rec and rec["text"]) or node

    def get_node_info(self, names):
        """批量查节点元数据,返回 {name: {'type':'entity'|'event', 'content':str|None}}。

        Phase 2 双层适配用:loop 每轮结束需要把"本轮新加进证据子图的节点"分成
        Entity / Event 两类,Event 还要拿到 content 给 LLM 决策读。逐个 describe()
        会触发 N 次往返,这里一次查清。
        """
        if not names:
            return {}
        cypher = """
        MATCH (n) WHERE n.name IN $names AND (n:Entity OR n:Event)
        RETURN n.name AS name,
               CASE WHEN n:Event THEN 'event' ELSE 'entity' END AS type,
               n.content AS content,
               n.desc AS desc
        """
        info = {}
        with self._session() as s:
            for rec in s.run(cypher, names=list(names)):
                info[rec["name"]] = {
                    "type": rec["type"],
                    "content": rec["content"] or rec["desc"],
                }
        return info

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

    # ---------- 双层 schema(Phase 2):事件节点与三类边 ----------

    def upsert_event(self, name, content, chunk_id=None, order=None):
        """建/更新 Event 节点(厚节点,带情节 content)。"""
        cypher = """
        MERGE (e:Event {name:$name})
        SET e.content = $content, e.chunk_id = $chunk_id, e.order = $order
        """
        with self._session() as s:
            s.run(cypher, name=name, content=content,
                  chunk_id=chunk_id, order=order)

    def upsert_participation(self, entity, event):
        """(:Entity)-[:PARTICIPATES]->(:Event):谁参与了事件。"""
        cypher = """
        MERGE (n:Entity {name:$entity})
        MERGE (e:Event {name:$event})
        MERGE (n)-[:PARTICIPATES]->(e)
        """
        with self._session() as s:
            s.run(cypher, entity=entity, event=event)

    def upsert_event_sequence(self, prev_event, next_event):
        """(:Event)-[:NEXT]->(:Event):事件时序(续写必需)。"""
        cypher = """
        MATCH (a:Event {name:$prev}), (b:Event {name:$next})
        MERGE (a)-[:NEXT]->(b)
        """
        with self._session() as s:
            s.run(cypher, prev=prev_event, next=next_event)

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
