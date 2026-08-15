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


def _app_with_broken_qdrant(tmp_path: Path, *, error: str = "storage locked"):
    """An app whose Qdrant client raises `error` from `collection_exists()`
    - the shared setup for every test exercising the readiness/liveness
    endpoints' behavior when Qdrant is unreachable."""
    app = create_app(_test_settings(tmp_path))
    broken_client = MagicMock()
    broken_client.collection_exists.side_effect = RuntimeError(error)
    app.dependency_overrides[get_qdrant_client] = lambda: broken_client
    return app


def _mock_ollama_healthy():
    """`patch(...)` context manager for a healthy `requests.get()` call to
    Ollama's `/api/tags` - the shared setup for every readiness test that
    doesn't care about Ollama's own state."""
    return patch("agentic_rag.api.routers.health.requests.get")


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

    with _mock_ollama_healthy() as mock_get:
        mock_get.return_value.raise_for_status.return_value = None
        with TestClient(app) as client:
            response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"] == {"qdrant": "ok", "ollama": "ok"}


def test_readiness_checks_ollamas_api_tags_endpoint_not_the_bare_base_url(tmp_path):
    # Every real caller in this codebase hits Ollama's structured /api/*
    # endpoints, not its root - a reverse proxy or unrelated service could
    # answer 200 at "/" while the actual API is down, producing a false
    # "ready" signal if this checked the bare base URL instead.
    settings = _test_settings(tmp_path)
    app = create_app(settings)

    with _mock_ollama_healthy() as mock_get:
        mock_get.return_value.raise_for_status.return_value = None
        with TestClient(app) as client:
            client.get("/health/ready")

    assert mock_get.call_args.args[0] == f"{settings.ollama_base_url}/api/tags"


def test_readiness_returns_503_when_qdrant_collection_does_not_exist(tmp_path):
    # Qdrant being reachable but the configured collection missing (a
    # misconfigured qdrant_collection_name, or a collection deleted at
    # runtime) is a real, distinct failure - collection_exists() returns
    # False without raising, so this must not be silently treated as "ok"
    # just because no exception was thrown.
    app = create_app(_test_settings(tmp_path))
    missing_client = MagicMock()
    missing_client.collection_exists.return_value = False
    app.dependency_overrides[get_qdrant_client] = lambda: missing_client

    with _mock_ollama_healthy() as mock_get:
        mock_get.return_value.raise_for_status.return_value = None
        with TestClient(app) as client:
            response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert "does not exist" in body["checks"]["qdrant"]
    assert body["checks"]["ollama"] == "ok"


def test_readiness_returns_503_when_qdrant_is_unreachable(tmp_path):
    app = _app_with_broken_qdrant(tmp_path)

    with _mock_ollama_healthy() as mock_get:
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


def test_readiness_reports_a_non_request_exception_from_ollama_gracefully(tmp_path):
    # The Ollama check must degrade to a reported checks entry for ANY
    # exception, not just requests.RequestException - an asymmetric catch
    # here would let an unusual failure mode 500 the whole endpoint
    # instead of honoring "every dependency is always checked."
    app = create_app(_test_settings(tmp_path))

    with patch(
        "agentic_rag.api.routers.health.requests.get",
        side_effect=ValueError("unexpected"),
    ):
        with TestClient(app) as client:
            response = client.get("/health/ready")

    assert response.status_code == 503
    assert "unexpected" in response.json()["checks"]["ollama"]


def test_readiness_reports_both_failures_when_both_dependencies_are_down(tmp_path):
    app = _app_with_broken_qdrant(tmp_path)

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

    with _mock_ollama_healthy() as mock_get:
        mock_get.return_value.raise_for_status.return_value = None
        with TestClient(app) as client:
            client.get("/health/ready")

    assert mock_get.call_args.kwargs["timeout"] == settings.readiness_check_timeout_seconds


def test_health_liveness_is_unaffected_by_broken_dependencies(tmp_path):
    # /health must keep answering "the process is up" regardless of
    # downstream state - that's the whole reason it's a separate
    # endpoint from /health/ready.
    app = _app_with_broken_qdrant(tmp_path)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
