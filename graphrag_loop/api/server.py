"""FastAPI 服务 —— POST /query 调 orchestration,返回答案 + 证据 + 轨迹。

# TODO(Phase 3+,对外服务时)
当前状态:占位,未实现。
触发条件:需要把系统暴露给前端 / 其他服务消费时启用。开发期 run_phase2.py 命令行已足够。

实现要点:
  - FastAPI app + uvicorn
  - POST /query: { query, options } → { answer, evidence_subgraph, trace_url }
  - 复用 run_phase2.main 的核心逻辑(待先重构为可复用函数)
  - 复用 RunRecorder 落地轨迹,trace_url 指向 logs/*.md
  - 鉴权 / 速率限制按部署需求加
  - 健康检查 /healthz: 转调 check_connections.py 的 4 条线
"""
