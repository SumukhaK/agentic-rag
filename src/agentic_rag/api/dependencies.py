from fastapi import Request
from qdrant_client import QdrantClient

from agentic_rag.config import Settings
from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.orchestration.semantic_cache import SemanticCache


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_qdrant_client(request: Request) -> QdrantClient:
    return request.app.state.qdrant_client


def get_embedding_cache(request: Request) -> EmbeddingCache:
    return request.app.state.embedding_cache


def get_semantic_cache(request: Request) -> SemanticCache:
    return request.app.state.semantic_cache
