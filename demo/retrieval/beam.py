# Beam 扩展与截断。
# expand(frontier) → 候选池;rank+剪枝(< tau_rel 砍掉)→ 取 top-N 作为新 frontier。
# 这是防"图爆炸"的核心宽度控制。
