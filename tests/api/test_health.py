from pathlib import Path

from fastapi.testclient import TestClient

from agentic_rag.api.app import create_app
from agentic_rag.config import Settings
from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.orchestration.semantic_cache import SemanticCache
from qdrant_client import QdrantClient


def _test_settings(tmp_path: Path) -> Settings:
    return Settings(
        watched_folder_path=tmp_path / "corpus",
        qdrant_storage_path=tmp_path / "qdrant",
        _env_file=None,
    )


def test_health_returns_ok(tmp_path):
    app = create_app(_test_settings(tmp_path))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lifespan_populates_shared_resources_on_app_state(tmp_path):
    app = create_app(_test_settings(tmp_path))

    with TestClient(app):
        assert isinstance(app.state.settings, Settings)
        assert isinstance(app.state.qdrant_client, QdrantClient)
        assert isinstance(app.state.embedding_cache, EmbeddingCache)
        assert isinstance(app.state.semantic_cache, SemanticCache)


def test_lifespan_creates_the_qdrant_collection(tmp_path):
    settings = _test_settings(tmp_path)
    app = create_app(settings)

    with TestClient(app):
        assert app.state.qdrant_client.collection_exists(settings.qdrant_collection_name)
