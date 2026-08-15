import pydantic
import pytest
from fastapi import FastAPI

from agentic_rag.api.main import create


def test_create_returns_a_working_asgi_app(tmp_path, monkeypatch):
    # agentic_rag.api.main:create is the factory a real deployment (or
    # Docker's CMD) points `uvicorn ... --factory` at - without it,
    # create_app() can only ever be reached from a test that constructs
    # its own Settings, with no way to actually run the app as a
    # standalone server.
    monkeypatch.setenv("WATCHED_FOLDER_PATH", str(tmp_path / "corpus"))
    monkeypatch.setenv("QDRANT_STORAGE_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("SYNC_SNAPSHOT_PATH", str(tmp_path / "sync_snapshot.json"))

    app = create()

    assert isinstance(app, FastAPI)
    # .openapi() is the stable, public way to check route registration in
    # this codebase (already established in tests/api/test_app.py) -
    # app.routes' internal shape isn't a contract worth depending on.
    registered_paths = app.openapi()["paths"]
    assert "/health" in registered_paths
    assert "/health/ready" in registered_paths
    assert "/query" in registered_paths


def test_create_fails_fast_without_a_watched_folder_path(monkeypatch):
    # watched_folder_path has no default (config.py) - a real deployment
    # missing this required setting should fail the moment create() runs
    # (at startup, before serving anything), not with a confusing error
    # the first time a request comes in. This repo's own root has no
    # committed .env (verified: gitignored, none present), so an unset
    # env var isn't masked by one here. Unlike a module-level
    # `app = create_app(Settings())` would, merely *importing* this
    # module (already done at the top of this file) has no such
    # side effect - only calling create() does, which is exactly the
    # point of using a factory function instead.
    monkeypatch.delenv("WATCHED_FOLDER_PATH", raising=False)

    with pytest.raises(pydantic.ValidationError):
        create()
