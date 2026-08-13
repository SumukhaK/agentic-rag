# Project Tracker

Phased roadmap for the Agentic RAG system. Updated as each phase/feature is
completed and merged. See [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) for
the full spec behind each item, and [`.claude/CLAUDE.md`](.claude/CLAUDE.md)
for how work gets done (TDD, one-feature-per-PR, Kodus.io review).

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done and merged

---

## Phase 0 — Project Foundations

- [x] Repo scaffolding (git init, `.gitignore`, initial README)
- [x] `.claude/CLAUDE.md` — working agreement / agent rules
- [x] `docs/REQUIREMENTS.md` — product & system requirements
- [x] `PROJECT_TRACKER.md` — this file
- [x] `README.md` — high-level architecture diagram
- [x] `.env.example` + central config module skeleton (built incrementally
      alongside the Phase 1 features that needed each setting)

## Phase 1 — Data Ingestion & Processing

- [x] Document source-of-truth decided: watched folder/filesystem (see
      `docs/REQUIREMENTS.md` §3, §13)
- [x] Folder watcher: deterministic snapshot/diff of a folder to detect
      created/modified/deleted files (`src/agentic_rag/ingestion/watcher.py`)
- [x] `markitdown` conversion wrapper: any supported file → Markdown text
      (`src/agentic_rag/ingestion/converter.py`)
- [x] Wire watcher → converter into a single ingestion pipeline step
      (`src/agentic_rag/ingestion/pipeline.py::process_changes`)
- [x] Hybrid chunking: fixed target size (`CHUNK_SIZE_CHARS`, default 2000)
      with boundary-aware extension for oversized blocks
      (`src/agentic_rag/ingestion/chunker.py::chunk_markdown`)
- [x] Wire chunker into the ingestion pipeline (`process_changes` now
      returns each document with its `chunks`)
- [x] Access-level tagging: folder-per-tier convention, validated against the
      configured `ACCESS_TIERS` list, wired into `process_changes`
      (`src/agentic_rag/ingestion/tagger.py`)
- [x] Edit/delete propagation: `sync_folder()` ties `watcher.snapshot`/
      `diff_snapshots` and `pipeline.process_changes` into one ingestion-cycle
      entrypoint, returning documents, failures, deleted paths, and the
      snapshot to persist for the next cycle
      (`src/agentic_rag/ingestion/sync.py`). Scheduling how often this runs
      (the "minutes"/"immediately" timing from FR4) is Phase 7's job.
- [x] Schema + validation for processed documents: `IngestedDocument`/`Chunk`
      (dataclasses) are the schema; `validate_document()` enforces the
      invariants before a document is considered indexable - non-empty
      chunk list, non-empty chunk text, non-empty access tier. A failing
      document becomes an `IngestionFailure`, same as a tagging or
      conversion error, rather than silently entering the index
      (`src/agentic_rag/ingestion/validation.py`)

**Phase 1 complete.**

## Phase 2 — Indexing Layer

- [x] Qdrant setup with HNSW indexing (dense vectors): `get_client()`/
      `ensure_collection()` — local/embedded mode (on-disk storage, no
      Docker in this environment), swappable for a real server later.
      Qdrant indexes dense vectors with HNSW by default, so creating a
      standard collection already satisfies this; verified live (default
      HNSW config: m=16, ef_construct=100). Collection created with named
      dense (`"dense"`) + sparse (`"sparse"`) vectors from the start, since
      Qdrant can't add a sparse field to an existing collection later, and
      raises `CollectionSchemaMismatchError` instead of silently no-opping
      if the dense vector size doesn't match what's requested
      (`src/agentic_rag/indexing/qdrant_setup.py`)
- [x] Qdrant native hybrid search enabled (sparse + dense):
      `embed_sparse_texts()` — BM25 via `fastembed` (`Qdrant/bm25`),
      returns `qdrant_client`'s own `SparseVector` type directly
      (`src/agentic_rag/embedding/sparse_client.py`). `index_document()` /
      `delete_document()` (`src/agentic_rag/indexing/upsert.py`) embed
      each chunk with both dense (Ollama) and sparse (BM25) vectors and
      upsert them as Qdrant points, with citation/access-control payload
      (`relative_path`, `chunk_index`, `text`, `access_tier`). Deterministic
      point IDs (uuid5 of path+chunk index) make re-indexing idempotent;
      existing points for a document are deleted before its current chunk
      set is inserted, so an edit that reduces the chunk count doesn't
      leave stale points behind. Verified end-to-end against the real
      Ollama server and a real local Qdrant collection.
- [x] Embedding generation via `nomic-embed-text` (Ollama):
      `embed_texts()` calls Ollama's batch-capable `/api/embed` endpoint
      (one HTTP round-trip for many chunks), with `embed_text()` as a
      single-item convenience wrapper. Tested with mocked HTTP (including
      malformed-response cases) and smoke-tested against the real local
      Ollama server (`src/agentic_rag/embedding/ollama_client.py`)
- [x] Embedding cache: `EmbeddingCache` + `embed_with_cache()` — in-memory,
      keyed on `(model, text)`, generic over dense and sparse.
      Wired into `index_document()` via a required `embedding_cache` param
      shared across calls, so repeated chunk text across different
      documents in the same run skips re-embedding. Verified live: cache
      hit went from ~6.6s to ~0.006s. Deliberately in-memory only for this
      first version — persistence is an explicit open question, not
      designed speculatively now (`src/agentic_rag/embedding/cache.py`)

**Phase 2 complete.**

## Phase 3 — Retrieval Pipeline

- [x] Parallel hybrid search (dense + keyword) against Qdrant, **and**
      Fusion of results → top 10 candidates: these are one Qdrant
      operation, not two — `hybrid_search()` uses `prefetch` (dense +
      sparse) + `FusionQuery(fusion=Fusion.RRF)` in a single call, up to
      `RETRIEVAL_TOP_K_CANDIDATES` (default 10) results. Prefetch
      over-fetches 4× `top_k` per leg (RRF only ranks over what was
      already fetched, so equal limits would silently drop competitive
      candidates ranked just outside `top_k` on both legs). Dense and
      sparse query embedding run concurrently (thread pool), not
      sequentially, since dense is a blocking Ollama call and this is the
      hottest path in the system
      (`src/agentic_rag/retrieval/search.py`)
- [x] Access-control filtering applied before fusion/reranking (FR3):
      `allowed_tiers_for()` (`src/agentic_rag/retrieval/access.py`)
      resolves which tiers a user may see; `hybrid_search()` applies it as
      a Qdrant filter on *both* the dense and sparse `Prefetch` legs, so a
      disallowed chunk never enters the candidate pool fusion ranks over.
      Verified live: a tier-2-only chunk never appeared in a tier-1 user's
      results, with correct relevance ranking (RRF) on top
- [x] Reranker (local cross-encoder) → top 4 chunks: `rerank()` via
      `fastembed`'s `TextCrossEncoder` (`BAAI/bge-reranker-base` —
      substituted for the originally-named `bge-reranker-v2-m3`, which
      `fastembed` doesn't support; same model family, avoids adding
      `sentence-transformers`/PyTorch as a new dependency). Configurable
      via `RERANKER_MODEL`/`RERANK_TOP_K`. Verified live end-to-end with
      `hybrid_search()`: sharper relevance separation than the fused RRF
      score alone (`src/agentic_rag/retrieval/rerank.py`)

**Phase 3 complete.** Semantic cache was originally listed here but is
moved to Phase 5 (see below) — it caches *answers*, and there's no answer
to cache until generation exists. Listing it under Retrieval was a
sequencing mistake made when this roadmap was first drafted; not
building unwired infrastructure to fill the slot.

## Phase 4 — Orchestration & Multi-Turn Chat

- [x] `mistral` pulled and verified working via Ollama — needed now, not
      just for Phase 5, since rewriting/decomposition are LLM reasoning
      tasks. `generate()` (`src/agentic_rag/generation/llm_client.py`)
      wraps Ollama's `/api/generate` endpoint; shared building block that
      Phase 5's final answer generation will reuse with a different
      prompt. See `docs/REQUIREMENTS.md` §10 for why `mistral` over
      Mixtral (~4.1GB vs. ~26GB)
- [x] Orchestrator: history + query rewriting on every turn: `rewrite_query()`
      (`src/agentic_rag/orchestration/rewrite.py`) — given prior
      `ConversationTurn`s + the new query, prompts `generate()` for a
      single standalone question. Returns the query unchanged with no LLM
      call when there's no history (first turn is already standalone).
      Verified live: correctly resolved "them"/"it" pronouns from a prior
      turn into a fully self-contained retrieval query
- [x] Sub-question decomposition: `decompose_query()`
      (`src/agentic_rag/orchestration/decompose.py`) — one sub-question
      per line of the LLM's response, list markers stripped. Raises
      `GenerationError` on a result with no usable sub-questions.
      Verified live — worth knowing: `mistral` doesn't reliably return an
      already-simple question unchanged despite being instructed to (see
      `docs/REQUIREMENTS.md` §10 for the observed example); a genuinely
      complex question decomposed cleanly into one sub-question per clause
- [x] Retry/replanning loop on insufficient evidence: `plan_and_retrieve()`
      (`src/agentic_rag/orchestration/planning.py`) — decomposes, retrieves,
      and reranks per sub-question; if any sub-question comes back with no
      reranked candidates, retries by re-decomposing from scratch (fresh
      LLM phrasing is the only thing that can plausibly change results
      against a deterministic corpus). Attempt budget is
      `Settings.max_retrieval_attempts` (default 5) — configurable, not
      hardcoded, per explicit instruction. "Sufficient" is deliberately a
      coarse retrieval-only signal (candidates non-empty after rerank), not
      an answer-quality judgment: a fixed cutoff on the reranker's own
      score was tried and rejected after live testing showed relevant and
      irrelevant candidates produce overlapping score ranges for short,
      generic questions (a genuinely relevant candidate scored -5.88,
      worse than a genuinely irrelevant one at -4.44). Real answerability
      judgment needs the LLM to reason over retrieved text, which belongs
      to generation (Phase 5), not a retrieval-time score threshold
- [x] Canonical "I do not know" fallback: `CANNOT_ANSWER_MESSAGE`
      (`src/agentic_rag/orchestration/planning.py`) — single constant
      reused for both the direct no-match path (insufficient on the first
      attempt) and the exhausted-retry path, since both collapse to the
      same `PlanningResult(sufficient=False, ...)` outcome; there was never
      a need for two separate code paths. Actually wired onto the result
      (not just defined): `PlanningResult.message` carries the constant
      when `sufficient=False`, `None` otherwise — self-review on PR #24
      caught that the first version defined the constant but never
      attached it to anything a caller could read
- [x] Self-review hardening (PR #24): `plan_and_retrieve()`'s attempt loop
      had no exception handling — a single transient failure from
      `decompose_query`/`hybrid_search`/`rerank` (e.g. a dropped Ollama
      connection) propagated straight out and aborted the entire retry
      budget on the spot, defeating the loop's purpose. Three independent
      review angles converged on this same bug. Fixed: `GenerationError`,
      `EmbeddingError`, `SparseEmbeddingError`, and `RerankError` are now
      caught per attempt and treated as "this attempt found nothing, try
      again"; `UnknownAccessTierError` is deliberately left uncaught since
      a bad `user_tier` is a config error no retry can fix. The
      per-sub-question retrieve+rerank logic was also extracted into
      `_retrieve_outcome()` to give the try/except a single, clear
      boundary. (One review finding — a claimed 40-line function-length
      CLAUDE.md violation — was checked against this repo's actual
      `.claude/CLAUDE.md` and found to not exist; not acted on.)

## Phase 5 — Generation & Grounding

- [x] Prompt assembly + generation via `mistral`: `generate_answer()`
      (`src/agentic_rag/orchestration/answer.py`) — takes a `PlanningResult`
      straight from Phase 4's `plan_and_retrieve()`. If insufficient,
      returns `planning_result.message` (the canonical fallback) directly
      with **no LLM call** — nothing to ground an answer in, so calling the
      model would only risk it reaching for outside knowledge. Otherwise
      flattens+deduplicates candidates across every sub-question's outcome
      (the same chunk can be evidence for more than one sub-question — a
      dedup by `(relative_path, chunk_index)` keeps it out of the prompt
      twice) into a citation-numbered source list (`[1]`, `[2]`, ...,
      each labelled with its source path, **chunk index**, and access tier)
      and calls `generate()`. Reconciles the original architecture's "top 4
      chunks" framing with Phase 4's decomposition: with N sub-questions the
      evidence set is N × `rerank_top_k` before dedup, not a fixed 4 — the
      original figure described the single-question case before
      decomposition existed
- [x] Citation enforcement + no-outside-knowledge enforcement (§8 rules 1
      and 3): encoded in the generation prompt (cite `[N]` per claim, never
      use knowledge beyond the numbered sources) **and validated after the
      fact** — self-review on PR #25 correctly called out that prompt-only
      enforcement can't satisfy a rule stated as having "no exceptions",
      since prompt-following is probabilistic. `_is_grounded()` now checks
      every returned answer: it must be the canonical fallback verbatim, or
      cite at least one source number that's actually in range
      (`1..len(candidates)`). An answer with zero citations, or a citation
      to a source that doesn't exist (worse than none — it carries false
      authority), is replaced with `CANNOT_ANSWER_MESSAGE` rather than
      returned as-is. Citations now also identify the exact chunk (not just
      the file), per §8 rule 1's literal wording — the first version's
      citation label only had the file path, missed by every finder angle
      except one on the first review pass
- [x] "I do not know" enforcement (§8 rule 2), **verified live as a working
      second line of defense**: the retrieval-only `sufficient` signal
      (Phase 4) is coarse and can be `True` even when nothing relevant was
      actually found (a 1-document corpus returns *something* for any
      query). Live-tested against the real Ollama/Qdrant stack: asking
      "What is the capital of France?" against a football-only corpus
      still resolved `sufficient=True` from `plan_and_retrieve`, but
      `generate_answer()`'s prompt-level instruction caught what the
      retrieval signal missed — `mistral` correctly returned
      `CANNOT_ANSWER_MESSAGE` verbatim instead of fabricating an answer
      from the irrelevant chunk. This is exactly the deferred-to-Phase-5
      answerability judgment Phase 4's docstring anticipated
- [x] Self-review hardening (PR #25): `PlanningResult.message` was typed
      `str | None` with nothing enforcing it's non-`None` when
      `sufficient=False` — `generate_answer()`'s `-> str` signature could
      silently return `None` if a future caller constructed a mismatched
      `PlanningResult` by hand. Fixed with a `__post_init__` invariant on
      `PlanningResult` itself (the right layer to enforce it, not every
      consumer) raising `ValueError` on construction if `sufficient` and
      `message` disagree. Also added a defensive fallback in
      `generate_answer()` for `sufficient=True` with zero actual candidates
      (not reachable via `plan_and_retrieve` today, but not guaranteed by
      `answer.py` itself either) — returns the canonical fallback rather
      than firing a pointless LLM call over an empty source list.
      **Known, deliberately deferred**: no bound on assembled prompt size
      as sub-question count grows (flagged by the efficiency review angle;
      needs a token-budgeting design decision, not a quick fix — noted here
      rather than guessed at)
- [ ] Claude-as-evaluator wiring (offline eval, not in the live answer path)
- [ ] Semantic cache (query-meaning-keyed answer cache) — moved from Phase 3;
      needs a decision on cache backend and similarity threshold, see
      `docs/REQUIREMENTS.md` §14

## Phase 6 — Access Control & Security

- [ ] Configurable linear access-tier model (§11) wired end-to-end
- [ ] Prompt-injection LLM judge — **blocked on model choice**, see
      `docs/REQUIREMENTS.md` §14
- [ ] Output/citation validation before returning an answer
- [ ] Foul-language refusal
- [ ] Secrets/config hygiene audit

## Phase 7 — API & Delivery

- [ ] FastAPI backend exposing chat/query endpoints
- [ ] OpenAPI docs kept accurate
- [ ] Background sync job for near-real-time index freshness (FR4)

## Phase 8 — Evaluation, Observability & Production Readiness

- [ ] Structured evaluation: retrieval precision, faithfulness, hallucination
      rate (Claude as judge)
- [ ] Logging/tracing across the pipeline
- [ ] Load test at target scale (10,000 docs × ~50 pages)
- [ ] Deployment hardening (containerization, health checks)

---

## Initial Prompt (recorded verbatim)

> we are building a production grade agentic rag. you act like a a senior software architect, go through these and extract whats suitable for claude.md from them and add them there and take rest of the informations as project requirements and implementation guidelines. once done reading, analyse it deeply, plan the project and features to implement in each phases of the project and record them in a md file in project folder and update them as we accomplish implemntation of each phases. record this initial prompt as well in that tracker md. update the project readme with a high level of architecture diagram of what we are planning to do and update the same readme with short details as we implement each phase succesfully. Think and analyse before acting, read the existing code before making edits or updates. if there s a bug, read the code, understand and then plan before fixing it. fixing one bug should not break existing and working parts of the project.
>
> use TDD, write unit tests before writing actaul code. divide tasks into features and write code accordingly and push them to git with approriate commmit messages. each feature should first open a PR in git. Kodus.io will review the code and merge if its fine, if it changes the status to "Request changes" act on them accordingly. asll the code should be simple, human radable and human understandable format. do not assume anything, do not make up anything. ask if confused. keep all the api keys and sensitive data in env and do not push them. keep all configurations related settings in one file and at one place.
> questions can be divided into sub questions and if not sufficinet evidences are cited go to planning mode again and start the process again for upto 5 turns, if not good answer is obtained say "Couldn't find suitable answer based on docs available".
>
> roughly our user's query journey looks like this : from browser/UI user quesry - orchastrator- orchastrator re writes history and the query - then it goes to embedding - then vecotr db search plus keyword baed search happens in parallel trying to find top 10 candidates - re ranker ranks the combined serach results - top 4 chunks are selected after re ranking- finally top 4 chunks + rules + user query is passed as prompt to LLM for final answer generation.
>
> rules :
> 1. every fatual answer cites its source and user access level.
> 2. if the source do not contain answer, say "I do not know the answer based on indexed documents".
> 3. never use knowldge outside the sourced documents to generate answers.
>
> now few insturctions related to rag we are building. we are assuming that our system should be fast and reliable. we will target it should be able to process at least 10,000 docs each of on average page size 50. use https://github.com/microsoft/markitdown to convert document of any type to md file.
> functional requirements :
> 1. answer ther question from the corpus with citations to the eaxct source chunks.
> 2. Must inherit contxet from previous turn to support multi turn chat.
> 3. Respect document permission per user. if someone is at developer they cant get answers from docs reservred for managers, but higher level user can get info from his/her below level roles.
> 4. reflect the edits made in docs in minutes and deletion should reflect immedietly.
> 5. say i dont know when corpus doesnt have an answer.
>
> other instructions :
> 1.use hybrid chunking, follow a regular chunking size but when information spills over use hybrid chunking to avoid losing context.
> 2.use HNSW indexing for searhing in verctor.
> 3. use indexing.
> 4. use both vector db searchg and keyword bsed search in parallel and combine results. after fusion select upto top 10 candidates and then use reranking to select at max top 4 chunks to pass it on to LLM for the final output generation.
> 5. use a hybrid search pattern, merging lexical (BM25) and semantic (dense vector) results to maximize accuracy.
> 6. use both embedding cache and symanctic cache to improve retrieval speed.
> 7. saftey checks : apart from access control via groups as already mentioned, use LLM to judge any injections at peompting level and check citation links and final output chunks before printing for security threats or malfunctioning. refuse to entertain foul language at any level.
> once user query is enetered, rewrite history.
