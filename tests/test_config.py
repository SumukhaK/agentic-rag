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


def test_settings_defaults_qdrant_settings(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    for key in ("EMBEDDING_DIMENSIONS", "QDRANT_STORAGE_PATH", "QDRANT_COLLECTION_NAME"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings()

    assert settings.embedding_dimensions == 768
    assert settings.qdrant_storage_path == Path("./data/qdrant")
    assert settings.qdrant_collection_name == "documents"


def test_settings_qdrant_settings_overridable_from_env(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setenv("QDRANT_STORAGE_PATH", "/data/prod-qdrant")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "football_docs")

    settings = Settings()

    assert settings.embedding_dimensions == 1024
    assert settings.qdrant_storage_path == Path("/data/prod-qdrant")
    assert settings.qdrant_collection_name == "football_docs"


def test_settings_defaults_sparse_embedding_model(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.delenv("SPARSE_EMBEDDING_MODEL", raising=False)

    settings = Settings()

    assert settings.sparse_embedding_model == "Qdrant/bm25"


def test_settings_sparse_embedding_model_overridable_from_env(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.setenv("SPARSE_EMBEDDING_MODEL", "Qdrant/other-sparse-model")

    settings = Settings()

    assert settings.sparse_embedding_model == "Qdrant/other-sparse-model"


def test_settings_defaults_retrieval_top_k_candidates(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.delenv("RETRIEVAL_TOP_K_CANDIDATES", raising=False)

    settings = Settings()

    assert settings.retrieval_top_k_candidates == 10


def test_settings_retrieval_top_k_candidates_overridable_from_env(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.setenv("RETRIEVAL_TOP_K_CANDIDATES", "20")

    settings = Settings()

    assert settings.retrieval_top_k_candidates == 20


def test_settings_defaults_reranker_settings(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    for key in ("RERANKER_MODEL", "RERANK_TOP_K"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings()

    assert settings.reranker_model == "BAAI/bge-reranker-base"
    assert settings.rerank_top_k == 4


def test_settings_reranker_settings_overridable_from_env(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.setenv("RERANKER_MODEL", "some-other-reranker-model")
    monkeypatch.setenv("RERANK_TOP_K", "6")

    settings = Settings()

    assert settings.reranker_model == "some-other-reranker-model"
    assert settings.rerank_top_k == 6


def test_settings_defaults_generation_settings(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    for key in ("GENERATION_MODEL", "GENERATION_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings()

    assert settings.generation_model == "mistral"
    assert settings.generation_timeout_seconds == 60


def test_settings_generation_settings_overridable_from_env(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.setenv("GENERATION_MODEL", "some-other-generation-model")
    monkeypatch.setenv("GENERATION_TIMEOUT_SECONDS", "90")

    settings = Settings()

    assert settings.generation_model == "some-other-generation-model"
    assert settings.generation_timeout_seconds == 90


def test_settings_defaults_max_retrieval_attempts(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.delenv("MAX_RETRIEVAL_ATTEMPTS", raising=False)

    settings = Settings()

    assert settings.max_retrieval_attempts == 5


def test_settings_max_retrieval_attempts_overridable_from_env(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.setenv("MAX_RETRIEVAL_ATTEMPTS", "3")

    settings = Settings()

    assert settings.max_retrieval_attempts == 3


def test_settings_defaults_semantic_cache_similarity_threshold(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.delenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", raising=False)

    settings = Settings()

    assert settings.semantic_cache_similarity_threshold == 0.95


def test_settings_semantic_cache_similarity_threshold_overridable_from_env(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.setenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", "0.9")

    settings = Settings()

    assert settings.semantic_cache_similarity_threshold == 0.9


def test_settings_defaults_semantic_cache_ttl_seconds(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.delenv("SEMANTIC_CACHE_TTL_SECONDS", raising=False)

    settings = Settings()

    assert settings.semantic_cache_ttl_seconds == 300.0


def test_settings_semantic_cache_ttl_seconds_overridable_from_env(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.setenv("SEMANTIC_CACHE_TTL_SECONDS", "60")

    settings = Settings()

    assert settings.semantic_cache_ttl_seconds == 60.0


def test_settings_defaults_judge_temperature(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.delenv("JUDGE_TEMPERATURE", raising=False)

    settings = Settings()

    assert settings.judge_temperature == 0.0


def test_settings_judge_temperature_overridable_from_env(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.setenv("JUDGE_TEMPERATURE", "0.1")

    settings = Settings()

    assert settings.judge_temperature == 0.1


def test_settings_defaults_generation_temperature(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.delenv("GENERATION_TEMPERATURE", raising=False)

    settings = Settings()

    assert settings.generation_temperature == 0.0


def test_settings_generation_temperature_overridable_from_env(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.setenv("GENERATION_TEMPERATURE", "0.3")

    settings = Settings()

    assert settings.generation_temperature == 0.3


def test_settings_defaults_rewrite_and_decompose_temperatures(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    for key in ("REWRITE_TEMPERATURE", "DECOMPOSE_TEMPERATURE", "DECOMPOSE_RETRY_TEMPERATURE"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings()

    assert settings.rewrite_temperature == 0.0
    assert settings.decompose_temperature == 0.0
    assert settings.decompose_retry_temperature == 0.4


def test_settings_rewrite_and_decompose_temperatures_overridable_from_env(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.setenv("REWRITE_TEMPERATURE", "0.1")
    monkeypatch.setenv("DECOMPOSE_TEMPERATURE", "0.2")
    monkeypatch.setenv("DECOMPOSE_RETRY_TEMPERATURE", "0.6")

    settings = Settings()

    assert settings.rewrite_temperature == 0.1
    assert settings.decompose_temperature == 0.2
    assert settings.decompose_retry_temperature == 0.6


def test_settings_defaults_sync_interval_seconds(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.delenv("SYNC_INTERVAL_SECONDS", raising=False)

    settings = Settings()

    assert settings.sync_interval_seconds == 60.0


def test_settings_rejects_a_non_positive_sync_interval_seconds(monkeypatch):
    # asyncio.sleep() silently treats 0/negative as "don't sleep at all" -
    # left unvalidated, a misconfigured 0 would turn the background sync
    # loop into an unthrottled busy-loop.
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")

    with pytest.raises(ValidationError):
        Settings(sync_interval_seconds=0, _env_file=None)

    with pytest.raises(ValidationError):
        Settings(sync_interval_seconds=-1, _env_file=None)


def test_settings_defaults_sync_snapshot_path(monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", "/tmp/corpus")
    monkeypatch.delenv("SYNC_SNAPSHOT_PATH", raising=False)

    settings = Settings()

    assert settings.sync_snapshot_path == Path("./data/sync_snapshot.json")
