# Agent 决策策略(规则模拟版)。
# 根据累积证据估计置信度 conf;conf>=theta_stop 或 r>=D_max 则停止。
# 完整架构里会被换成真 LLM 的 Self-RAG 决策(loop/decision.py)。
