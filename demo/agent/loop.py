# Demo 的检索主循环(核心文件)。
# for r in 1..D_max: 扩展 → 剪枝 → Beam 截断 → Agent 决策(停/继续)。
# 维护 state:frontier、已访问边、累积证据、置信度。
# 这条控制流的契约与 graphrag_loop/loop/controller.py 保持一致。
