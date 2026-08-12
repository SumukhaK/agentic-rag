from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag.config import Settings


def test_settings_loads_watched_folder_path_from_env(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")

    settings = Settings()

    assert settings.watched_folder_path == Path("/tmp/corpus")


def test_settings_requires_watched_folder_path(monkeypatch):
    monkeypatch.delenv("WATCHED_FOLDER_PATH", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
