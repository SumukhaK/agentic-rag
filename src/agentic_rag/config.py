from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    watched_folder_path: Path
    # gt=0, not >=0: asyncio.sleep() silently treats 0/negative as "don't
    # sleep at all", which would turn the background sync loop into an
    # unthrottled busy-loop hammering the filesystem/Ollama/Qdrant forever.
    sync_interval_seconds: float = Field(default=60.0, gt=0)
    sync_snapshot_path: Path = Path("./data/sync_snapshot.json")
    chunk_size_chars: int = 2000
    access_tiers: list[str] = ["employee", "manager", "director"]
    ollama_base_url: str = "http://localhost:11434"
    # Deliberately short and separate from embedding_timeout_seconds/
    # generation_timeout_seconds - a readiness probe exists to answer
    # "can this reach Ollama right now," not to wait as long as a real
    # embedding/generation call would; a slow readiness check defeats
    # its own purpose (a container orchestrator polling it frequently).
    # gt=0 matches sync_interval_seconds' own guard: requests.get(...,
    # timeout=0) has unpredictable behavior rather than failing fast.
    readiness_check_timeout_seconds: int = Field(default=3, gt=0)
    embedding_model: str = "nomic-embed-text"
    embedding_timeout_seconds: int = 30
    embedding_dimensions: int = 768
    qdrant_storage_path: Path = Path("./data/qdrant")
    qdrant_collection_name: str = "documents"
    qdrant_backup_path: Path = Path("./data/qdrant_backups")
    # Bounds backup I/O to a wall-clock cadence independent of
    # sync_interval_seconds - copying the whole Qdrant storage directory
    # on every ~60s sync cycle would be wasteful at any real corpus size;
    # hourly is a starting point, not a tuned value.
    qdrant_backup_interval_seconds: float = Field(default=3600.0, gt=0)
    # gt=0: a retention count of 0 would mean "prune every backup
    # immediately after creating it," which defeats the point of having
    # one at all - not a real configuration this system should silently
    # accept.
    qdrant_backup_retention_count: int = Field(default=3, gt=0)
    sparse_embedding_model: str = "Qdrant/bm25"
    retrieval_top_k_candidates: int = 10
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = 4
    generation_model: str = "mistral"
    generation_timeout_seconds: int = 60
    max_retrieval_attempts: int = 5
    semantic_cache_similarity_threshold: float = 0.95
    semantic_cache_ttl_seconds: float = 300.0
    judge_temperature: float = 0.0
    generation_temperature: float = 0.0
    rewrite_temperature: float = 0.0
    decompose_temperature: float = 0.0
    decompose_retry_temperature: float = 0.4
    evaluation_model: str = "qwen2.5:14b-instruct"
    evaluation_temperature: float = 0.0
    evaluation_timeout_seconds: int = 120
    evaluation_corpus_path: Path = Path("./eval/corpus")
    evaluation_questions_path: Path = Path("./eval/questions.json")
    evaluation_qdrant_storage_path: Path = Path("./eval/qdrant")
    evaluation_qdrant_collection_name: str = "eval_documents"
    evaluation_results_path: Path = Path("./eval/results")
    loadtest_corpus_staging_path: Path = Path("./loadtest/corpus_staging")
    loadtest_watched_folder_path: Path = Path("./loadtest/watched")
    loadtest_qdrant_storage_path: Path = Path("./loadtest/qdrant")
    loadtest_qdrant_collection_name: str = "loadtest_documents"
    loadtest_sync_snapshot_path: Path = Path("./loadtest/sync_snapshot.json")
    loadtest_results_path: Path = Path("./loadtest/results")
    # Bounds how much work a crash can lose to roughly one batch, not the
    # whole ~30-hour run - run_sync_cycle() only checkpoints (via
    # save_snapshot()) once per call, and at the ~10.6s/doc rate measured
    # in README.md's calibration run, 200 docs/batch is ~35 minutes of
    # exposure, not hours.
    loadtest_batch_size: int = Field(default=200, gt=0)
