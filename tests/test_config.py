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


def test_settings_defaults_chunk_size_chars(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.delenv("CHUNK_SIZE_CHARS", raising=False)

    settings = Settings()

    assert settings.chunk_size_chars == 2000


def test_settings_chunk_size_chars_overridable_from_env(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.setenv("CHUNK_SIZE_CHARS", "500")

    settings = Settings()

    assert settings.chunk_size_chars == 500


def test_settings_defaults_access_tiers(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.delenv("ACCESS_TIERS", raising=False)

    settings = Settings()

    assert settings.access_tiers == ["tier-1", "tier-2", "tier-3"]


def test_settings_access_tiers_overridable_from_env(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.setenv("ACCESS_TIERS", '["developer", "manager", "director"]')

    settings = Settings()

    assert settings.access_tiers == ["developer", "manager", "director"]


def test_settings_defaults_ollama_base_url_and_embedding_model(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    settings = Settings()

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.embedding_model == "nomic-embed-text"


def test_settings_ollama_base_url_and_embedding_model_overridable_from_env(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:11434")
    monkeypatch.setenv("EMBEDDING_MODEL", "some-other-embedding-model")

    settings = Settings()

    assert settings.ollama_base_url == "http://ollama.internal:11434"
    assert settings.embedding_model == "some-other-embedding-model"


def test_settings_defaults_embedding_timeout_seconds(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.delenv("EMBEDDING_TIMEOUT_SECONDS", raising=False)

    settings = Settings()

    assert settings.embedding_timeout_seconds == 30


def test_settings_embedding_timeout_seconds_overridable_from_env(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "60")

    settings = Settings()

    assert settings.embedding_timeout_seconds == 60
