"""端到端编排器 —— 用 LangGraph 把 entry → loop → generation → IsSup → refeed 串成状态机。

# TODO(Phase 3)
当前状态:占位,未实现。
触发条件:做 tool calling 风格的多源 Agent(图 / 向量 / Web 联合检索)时启用。

为什么现在不需要:run_phase2.py 已经把整套链路用普通 Python 控制流串好了,跑得通且可调试。
        LangGraph 的价值在于"把检索循环作为一个 tool,被外层 Agent 调用",这是
        Agentic RAG 的标准范式(参见 think-about-node.md 讨论的双层 Agent 决策)。
        Phase 2 只有"图谱检索"一条 tool,没必要上 LangGraph。

实现要点(等 Phase 3 真有多源时):
  - 把 run_phase2.main 拆成 graph_search_loop(query) 工具函数
  - LangGraph node:L1 Router → graph_search_loop / vector_search / web_search → 答案聚合
  - 状态机 state 含: query, accumulated_evidence, tool_history, final_answer
  - 外层路由 LLM(stage='L1_route_decision')决定调哪个 tool

参考:think-about-node.md "Agentic RAG 工具封装" 一节;LangGraph 文档。
"""
