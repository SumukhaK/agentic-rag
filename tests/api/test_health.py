from pathlib import Path
from unittest.mock import MagicMock, patch

import requests
from fastapi.testclient import TestClient

from agentic_rag.api.app import create_app
from agentic_rag.api.dependencies import get_qdrant_client
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


# --- readiness ---------------------------------------------------------------


def test_readiness_returns_200_when_all_dependencies_are_reachable(tmp_path):
    app = create_app(_test_settings(tmp_path))

    with patch("agentic_rag.api.routers.health.requests.get") as mock_get:
        mock_get.return_value.raise_for_status.return_value = None
        with TestClient(app) as client:
            response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"] == {"qdrant": "ok", "ollama": "ok"}


def test_readiness_returns_503_when_qdrant_is_unreachable(tmp_path):
    app = create_app(_test_settings(tmp_path))
    broken_client = MagicMock()
    broken_client.collection_exists.side_effect = RuntimeError("storage locked")
    app.dependency_overrides[get_qdrant_client] = lambda: broken_client

    with patch("agentic_rag.api.routers.health.requests.get") as mock_get:
        mock_get.return_value.raise_for_status.return_value = None
        with TestClient(app) as client:
            response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert "storage locked" in body["checks"]["qdrant"]
    assert body["checks"]["ollama"] == "ok"


def test_readiness_returns_503_when_ollama_is_unreachable(tmp_path):
    app = create_app(_test_settings(tmp_path))

    with patch(
        "agentic_rag.api.routers.health.requests.get",
        side_effect=requests.ConnectionError("connection refused"),
    ):
        with TestClient(app) as client:
            response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["qdrant"] == "ok"
    assert "connection refused" in body["checks"]["ollama"]


def test_readiness_reports_both_failures_when_both_dependencies_are_down(tmp_path):
    app = create_app(_test_settings(tmp_path))
    broken_client = MagicMock()
    broken_client.collection_exists.side_effect = RuntimeError("storage locked")
    app.dependency_overrides[get_qdrant_client] = lambda: broken_client

    with patch(
        "agentic_rag.api.routers.health.requests.get",
        side_effect=requests.ConnectionError("connection refused"),
    ):
        with TestClient(app) as client:
            response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert "storage locked" in body["checks"]["qdrant"]
    assert "connection refused" in body["checks"]["ollama"]


def test_readiness_uses_the_configured_timeout_for_the_ollama_check(tmp_path):
    settings = _test_settings(tmp_path)
    app = create_app(settings)

    with patch("agentic_rag.api.routers.health.requests.get") as mock_get:
        mock_get.return_value.raise_for_status.return_value = None
        with TestClient(app) as client:
            client.get("/health/ready")

    assert mock_get.call_args.kwargs["timeout"] == settings.readiness_check_timeout_seconds


def test_health_liveness_is_unaffected_by_broken_dependencies(tmp_path):
    # /health must keep answering "the process is up" regardless of
    # downstream state - that's the whole reason it's a separate
    # endpoint from /health/ready.
    app = create_app(_test_settings(tmp_path))
    broken_client = MagicMock()
    broken_client.collection_exists.side_effect = RuntimeError("storage locked")
    app.dependency_overrides[get_qdrant_client] = lambda: broken_client

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
