"""外部服务连接配置 —— 全部从 .env / 环境变量读,绝不硬编码密钥。

用法:在仓库根放 .env(从 .env.example 复制),首次 import 时自动加载。
各 storage / models 模块从这里取配置,而不是自己读环境变量,保证单一来源。
"""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    # 仓库根目录的 .env(本文件在 graphrag_loop/config/ 下,上溯两级)
    _root = Path(__file__).resolve().parents[2]
    load_dotenv(_root / ".env")
except ImportError:
    # 未装 python-dotenv 时退回纯环境变量(CI / 容器场景)
    pass


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"缺少环境变量 {key};请在 .env 中配置(参考 .env.example)")
    return val


@dataclass(frozen=True)
class DashScopeConfig:
    api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    model: str = os.getenv("LLM_MODEL", "qwen-plus")
    model_small: str = os.getenv("LLM_MODEL_SMALL", "qwen-turbo")


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str = os.getenv("EMBEDDING_BACKEND", "local")   # local | dashscope
    model_local: str = os.getenv("EMBEDDING_MODEL_LOCAL", "BAAI/bge-m3")
    model_dashscope: str = os.getenv("EMBEDDING_MODEL_DASHSCOPE", "text-embedding-v3")
    reranker: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "")


@dataclass(frozen=True)
class QdrantConfig:
    url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection: str = os.getenv("QDRANT_COLLECTION", "graphrag_chunks")


DASHSCOPE = DashScopeConfig()
EMBEDDING = EmbeddingConfig()
NEO4J = Neo4jConfig()
QDRANT = QdrantConfig()
