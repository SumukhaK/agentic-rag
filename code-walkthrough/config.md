# `config.py`

**Purpose:** This file defines the single place where every configuration value the whole application needs lives: file paths, timeouts, model names, thresholds, and so on. Instead of scattering "magic numbers" and settings across the codebase, every other module asks for a `Settings` object and reads what it needs from it. The values can come from environment variables or from a `.env` file, which is what lets the same code run differently in development, testing, and a load-test environment without editing any source file. This matches the project's stated philosophy of using `pydantic-settings` instead of `os.environ` calls scattered throughout the code.

## Line-by-line walkthrough

### Lines 1-4 — Imports
```python
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
```
- `from pathlib import Path` — imports Python's standard, object-oriented way of representing filesystem paths (as opposed to plain strings), which is used throughout this file for anything that names a file or folder.
- `from pydantic import Field` — imports `Field`, a helper used to attach extra rules (like "must be greater than zero") and a default value to a settings entry, beyond just giving it a type.
- `from pydantic_settings import BaseSettings, SettingsConfigDict` — imports the special base class (`BaseSettings`) that knows how to read its field values from environment variables and `.env` files automatically, plus a helper (`SettingsConfigDict`) used to configure how that reading behaves.

### Lines 7-10 — The `Settings` class and its config
```python
class Settings(BaseSettings):
    """Central configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```
- `class Settings(BaseSettings):` — declares the `Settings` class, inheriting from `BaseSettings` so that every field defined below automatically becomes something that can be populated from an environment variable of the same name.
- `"""Central configuration, loaded from environment variables / .env."""` — a one-line docstring (a description string attached to the class) explaining the class's role: it's the one central source of configuration for the app.
- `model_config = SettingsConfigDict(env_file=".env", extra="ignore")` — configures two behaviors: `env_file=".env"` tells Pydantic to also read values from a file named `.env` in the working directory (in addition to real environment variables), and `extra="ignore"` tells it to silently ignore any extra keys found in the environment or `.env` file that don't match a field here, rather than raising an error.

### Line 12 — Required setting: the watched folder
```python
    watched_folder_path: Path
```
- `watched_folder_path: Path` — declares a required field (no default value is given, so this must be supplied via environment variable or `.env`, or the app will fail to start) holding the filesystem path to the folder that the ingestion system watches for new or changed documents.

### Lines 13-16 — Sync interval, with a guard against zero
```python
    # gt=0, not >=0: asyncio.sleep() silently treats 0/negative as "don't
    # sleep at all", which would turn the background sync loop into an
    # unthrottled busy-loop hammering the filesystem/Ollama/Qdrant forever.
    sync_interval_seconds: float = Field(default=60.0, gt=0)
```
- The comment explains a deliberate, non-obvious design choice: the constraint on this field is "strictly greater than zero" (`gt=0`) rather than "zero or more" (`>=0`). If zero were allowed, Python's `asyncio.sleep()` (used somewhere in the sync loop) would just skip sleeping entirely, so the background job that periodically re-syncs documents would spin in a tight loop with no pause, overloading the filesystem, the Ollama server, and the Qdrant database.
- `sync_interval_seconds: float = Field(default=60.0, gt=0)` — declares how many seconds to wait between sync cycles, defaulting to 60 seconds, and enforces via `Field(..., gt=0)` that any configured value must be a positive number.

### Line 17 — Sync snapshot path
```python
    sync_snapshot_path: Path = Path("./data/sync_snapshot.json")
```
- `sync_snapshot_path: Path = Path("./data/sync_snapshot.json")` — the file where the sync process saves a "snapshot" of what it has already indexed, so that on restart it doesn't need to reprocess everything from scratch. Defaults to a relative path under `./data/`.

### Line 18 — Chunk size
```python
    chunk_size_chars: int = 2000
```
- `chunk_size_chars: int = 2000` — the maximum number of characters used when splitting a source document into smaller "chunks" for embedding and retrieval, defaulting to 2000 characters.

### Line 19 — Access tiers
```python
    access_tiers: list[str] = ["employee", "manager", "director"]
```
- `access_tiers: list[str] = ["employee", "manager", "director"]` — the list of valid access levels a user (and a document) can belong to, used to enforce that a user can only see documents at or below their tier. This is the list validated against when a query specifies a `user_tier`.

### Line 20 — Ollama base URL
```python
    ollama_base_url: str = "http://localhost:11434"
```
- `ollama_base_url: str = "http://localhost:11434"` — the network address of the locally running Ollama server (the tool used to run language models locally), which the app calls for embeddings, generation, and judging tasks. Defaults to Ollama's standard local port.

### Lines 21-28 — Readiness check timeout, with reasoning for why it's separate and short
```python
    # Deliberately short and separate from embedding_timeout_seconds/
    # generation_timeout_seconds - a readiness probe exists to answer
    # "can this reach Ollama right now," not to wait as long as a real
    # embedding/generation call would; a slow readiness check defeats
    # its own purpose (a container orchestrator polling it frequently).
    # gt=0 matches sync_interval_seconds' own guard: requests.get(...,
    # timeout=0) has unpredictable behavior rather than failing fast.
    readiness_check_timeout_seconds: int = Field(default=3, gt=0)
```
- The comment explains why this timeout is its own separate setting rather than reusing `embedding_timeout_seconds` or `generation_timeout_seconds`: a readiness check (used by things like container orchestrators polling "is this service healthy") needs to answer quickly whether Ollama is reachable at all — it shouldn't wait as long as a real embedding or generation request would, because a slow health check undermines the entire point of having one.
- It also explains why `gt=0` (strictly positive) is required here too: passing `timeout=0` to the `requests.get()` call used for this check has unpredictable behavior, rather than cleanly failing fast, so zero must be disallowed just like `sync_interval_seconds`.
- `readiness_check_timeout_seconds: int = Field(default=3, gt=0)` — declares this setting with a short default of 3 seconds and enforces it must be positive.

### Lines 29-31 — Embedding settings
```python
    embedding_model: str = "nomic-embed-text"
    embedding_timeout_seconds: int = 30
    embedding_dimensions: int = 768
```
- `embedding_model: str = "nomic-embed-text"` — the name of the Ollama model used to turn text into embeddings (numeric vectors representing meaning), defaulting to `nomic-embed-text`.
- `embedding_timeout_seconds: int = 30` — how long, in seconds, to wait for an embedding request to Ollama before giving up.
- `embedding_dimensions: int = 768` — the length of the numeric vector the embedding model produces; this must match what the vector database (Qdrant) expects, since the collection is created with a fixed vector size.

### Lines 32-33 — Qdrant storage settings
```python
    qdrant_storage_path: Path = Path("./data/qdrant")
    qdrant_collection_name: str = "documents"
```
- `qdrant_storage_path: Path = Path("./data/qdrant")` — the on-disk location where the local, embedded Qdrant vector database stores its data.
- `qdrant_collection_name: str = "documents"` — the name of the "collection" (similar to a table) inside Qdrant that holds the indexed document chunks.

### Lines 34-37 — Retrieval and reranking settings
```python
    sparse_embedding_model: str = "Qdrant/bm25"
    retrieval_top_k_candidates: int = 10
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = 4
```
- `sparse_embedding_model: str = "Qdrant/bm25"` — the model used to produce "sparse" embeddings (a keyword-based, BM25-style representation) used alongside the dense semantic embeddings for hybrid search, which combines exact keyword matching with meaning-based matching.
- `retrieval_top_k_candidates: int = 10` — how many candidate chunks the initial hybrid search step retrieves before any reranking happens.
- `reranker_model: str = "BAAI/bge-reranker-base"` — the name of the model used to re-score and reorder those candidates by relevance, more precisely than the initial retrieval step.
- `rerank_top_k: int = 4` — how many of the reranked candidates are actually kept and passed on to be used as context for generating an answer.

### Lines 38-40 — Generation and retry settings
```python
    generation_model: str = "mistral"
    generation_timeout_seconds: int = 60
    max_retrieval_attempts: int = 5
```
- `generation_model: str = "mistral"` — the Ollama model used to generate the final answer text (and also reused for the judge/security checks, based on how it's passed around in other files).
- `generation_timeout_seconds: int = 60` — how long, in seconds, to wait for a generation call to Ollama before giving up.
- `max_retrieval_attempts: int = 5` — the maximum number of times the system will retry or reformulate a retrieval attempt (for example, if initial results seem insufficient) before giving up and returning a "cannot answer" response.

### Lines 41-42 — Semantic cache settings
```python
    semantic_cache_similarity_threshold: float = 0.95
    semantic_cache_ttl_seconds: float = 300.0
```
- `semantic_cache_similarity_threshold: float = 0.95` — how similar (on a 0-to-1 scale) a new query's embedding must be to a previously cached query before the cached answer is reused instead of running the full pipeline again.
- `semantic_cache_ttl_seconds: float = 300.0` — "time to live": how many seconds a cached answer remains valid before it's considered stale and no longer reused, defaulting to 300 seconds (5 minutes).

### Lines 43-46 — Temperature settings for different LLM calls
```python
    judge_temperature: float = 0.0
    generation_temperature: float = 0.0
    rewrite_temperature: float = 0.0
    decompose_temperature: float = 0.0
```
- These four fields each configure the "temperature" (a parameter controlling how random versus deterministic a language model's output is, where `0.0` means as deterministic/consistent as possible) for a different kind of LLM call: `judge_temperature` for the security/injection judges, `generation_temperature` for producing the final answer, `rewrite_temperature` for rewriting a user's query into a self-contained question, and `decompose_temperature` for breaking a complex query into sub-questions. All default to `0.0` so these steps behave as predictably and repeatably as possible, which matters for an application that's expected to be traceable and testable.

### Line 47 — Decompose retry temperature
```python
    decompose_retry_temperature: float = 0.4
```
- `decompose_retry_temperature: float = 0.4` — a separate, higher temperature used specifically when retrying a failed query-decomposition attempt; a small amount of randomness on retry gives the model a chance to produce a different (hopefully better) result than the deterministic first attempt did.

### Lines 48-50 — Evaluation model settings
```python
    evaluation_model: str = "qwen2.5:14b-instruct"
    evaluation_temperature: float = 0.0
    evaluation_timeout_seconds: int = 120
```
- `evaluation_model: str = "qwen2.5:14b-instruct"` — a separate, presumably more capable model used specifically for evaluating the quality of the system's own answers (rather than for answering user queries directly).
- `evaluation_temperature: float = 0.0` — kept deterministic for the same reproducibility reasons as the other temperature settings.
- `evaluation_timeout_seconds: int = 120` — a longer timeout (2 minutes) than ordinary generation, reflecting that evaluation calls may involve larger models or more complex prompts.

### Lines 51-55 — Evaluation data paths
```python
    evaluation_corpus_path: Path = Path("./eval/corpus")
    evaluation_questions_path: Path = Path("./eval/questions.json")
    evaluation_qdrant_storage_path: Path = Path("./eval/qdrant")
    evaluation_qdrant_collection_name: str = "eval_documents"
    evaluation_results_path: Path = Path("./eval/results")
```
- These five fields define an entirely separate set of paths and a separate Qdrant collection used just for running evaluations, distinct from the paths used for the live/production data. This keeps evaluation runs from interfering with or being mixed up with real ingested data: `evaluation_corpus_path` is where evaluation source documents live, `evaluation_questions_path` is the file listing evaluation questions, `evaluation_qdrant_storage_path`/`evaluation_qdrant_collection_name` are a dedicated Qdrant database and collection for evaluation, and `evaluation_results_path` is where evaluation output is written.

### Lines 56-61 — Load-test paths
```python
    loadtest_corpus_staging_path: Path = Path("./loadtest/corpus_staging")
    loadtest_watched_folder_path: Path = Path("./loadtest/watched")
    loadtest_qdrant_storage_path: Path = Path("./loadtest/qdrant")
    loadtest_qdrant_collection_name: str = "loadtest_documents"
    loadtest_sync_snapshot_path: Path = Path("./loadtest/sync_snapshot.json")
    loadtest_results_path: Path = Path("./loadtest/results")
```
- Similarly, these six fields define a third, fully separate set of paths and a Qdrant collection used only when running load tests (tests that simulate heavy usage to measure performance): a staging area for corpus files before they're fed into the watched folder, the load-test's own watched folder, its own Qdrant storage and collection name, its own sync snapshot file, and where load-test results get written. Keeping these isolated from both production and evaluation data means a load test can't accidentally corrupt real data or evaluation baselines.

### Lines 62-67 — Load-test batch size, with reasoning about crash exposure
```python
    # Bounds how much work a crash can lose to roughly one batch, not the
    # whole ~30-hour run - run_sync_cycle() only checkpoints (via
    # save_snapshot()) once per call, and at the ~10.6s/doc rate measured
    # in README.md's calibration run, 200 docs/batch is ~35 minutes of
    # exposure, not hours.
    loadtest_batch_size: int = Field(default=200, gt=0)
```
- The comment explains the reasoning behind the default value of 200: because the sync process only saves its progress (a "checkpoint") once per call to `run_sync_cycle()`, processing documents in batches of 200 means that if the process crashes partway through, at most about 35 minutes of work (at a measured rate of roughly 10.6 seconds per document) is lost and needs to be redone — not the entire multi-hour load test run.
- `loadtest_batch_size: int = Field(default=200, gt=0)` — declares the setting with that reasoned default of 200 documents per batch, and requires it be a positive number (a batch size of zero or less would be meaningless).
