import importlib
import sys

import pydantic
import pytest
from fastapi import FastAPI


def test_main_module_exposes_an_asgi_app(tmp_path, monkeypatch):
    # agentic_rag.api.main is the module a real deployment (or Docker's
    # CMD) points uvicorn at (`uvicorn agentic_rag.api.main:app`) -
    # without it, create_app() can only ever be reached from a test that
    # constructs its own Settings, with no way to actually run the app
    # as a standalone server.
    monkeypatch.setenv("WATCHED_FOLDER_PATH", str(tmp_path / "corpus"))
    monkeypatch.setenv("QDRANT_STORAGE_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("SYNC_SNAPSHOT_PATH", str(tmp_path / "sync_snapshot.json"))
    sys.modules.pop("agentic_rag.api.main", None)

    try:
        main = importlib.import_module("agentic_rag.api.main")

        assert isinstance(main.app, FastAPI)
        # .openapi() is the stable, public way to check route registration
        # in this codebase (already established in tests/api/test_app.py)
        # - main.app.routes' internal shape isn't a contract worth
        # depending on directly.
        registered_paths = main.app.openapi()["paths"]
        assert "/health" in registered_paths
        assert "/health/ready" in registered_paths
        assert "/query" in registered_paths
    finally:
        # Don't leak a Settings()-bound module into later tests that
        # import agentic_rag.api.main without these env vars set.
        sys.modules.pop("agentic_rag.api.main", None)


def test_main_module_fails_fast_without_a_watched_folder_path(monkeypatch):
    # watched_folder_path has no default (config.py) - a real deployment
    # missing this required setting should fail at import/startup time,
    # not with a confusing error the first time a request comes in. This
    # repo's own root has no committed .env (verified: gitignored, none
    # present), so an unset env var isn't masked by one here.
    monkeypatch.delenv("WATCHED_FOLDER_PATH", raising=False)
    sys.modules.pop("agentic_rag.api.main", None)

    try:
        with pytest.raises(pydantic.ValidationError):
            importlib.import_module("agentic_rag.api.main")
    finally:
        sys.modules.pop("agentic_rag.api.main", None)
