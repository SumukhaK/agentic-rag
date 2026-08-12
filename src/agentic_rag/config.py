from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    watched_folder_path: Path
    chunk_size_chars: int = 2000
    access_tiers: list[str] = ["tier-1", "tier-2", "tier-3"]
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_timeout_seconds: int = 30
    embedding_dimensions: int = 768
    qdrant_storage_path: Path = Path("./data/qdrant")
    qdrant_collection_name: str = "documents"
    sparse_embedding_model: str = "Qdrant/bm25"
