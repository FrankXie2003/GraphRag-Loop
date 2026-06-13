"""离线建图总编排 —— chunk → 抽实体关系 → 对齐 → 写 Neo4j + Qdrant。

# TODO(可选重构)
当前状态:占位。**实际建图逻辑分散在 ingest_graph_md.py 和 ingest_graph_md_v2.py 里**
        (script 风格,各自封装了完整流程)。

何时收回到这里:当出现第三种数据源(如真实小说接入、增量建图)时,把共享逻辑
        (chunker → extractor → alignment → graph_store.upsert)抽到此模块,
        三个入口脚本只配置参数。

不急做的原因:script 风格的好处是"一个文件读完就能跑",对建图这种**离线一次性任务**
        反而比抽象框架更直观。强行抽象 = 过早优化。
"""
