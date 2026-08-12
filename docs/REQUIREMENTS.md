# Agentic RAG — Requirements & Implementation Guidelines

Product and system specification. This is the source of truth for *what* the
system does. For *how* work is carried out in this repo (TDD, branching, PR
review, secrets, config), see [`.claude/CLAUDE.md`](../.claude/CLAUDE.md).

The original prompt this document was distilled from is recorded verbatim in
[`PROJECT_TRACKER.md`](../PROJECT_TRACKER.md).

---

## 1. Vision

A production-grade agentic RAG system that answers questions grounded strictly
in an indexed document corpus, with per-user access control, source citations,
and multi-turn conversation support. The system must be fast, reliable, and
honest about the limits of what it knows.

## 2. Scale & Non-Functional Targets

- Must handle a corpus of **at least 10,000 documents**, averaging **~50 pages**
  each (≈500,000 pages).
- Must be **fast and reliable** — retrieval and generation latency are first-class
  design constraints, not an afterthought (this is why caching, ANN indexing,
  and a bounded top-k pipeline are required — see §7–§9).
- Index freshness: document **edits reflected within minutes**; document
  **deletions reflected immediately**.

## 3. Ingestion

- Source documents may be of any file type. All documents are converted to
  Markdown using [`markitdown`](https://github.com/microsoft/markitdown)
  before any downstream processing (chunking, embedding, indexing).
- Raw source files are treated as immutable inputs to the conversion step.
- **Source-of-truth: a watched folder/filesystem.** A configured directory
  (path set in the central config module, see `.claude/CLAUDE.md` §5) is the
  corpus's source of truth. New, modified, and deleted files in that folder
  drive ingestion — there is no separate upload API or external-system sync
  in this phase.
- **Schema + validation.** `IngestedDocument`/`Chunk` define the schema of a
  processed document — the shape everything downstream (indexing) can rely
  on. `validate_document()` enforces the invariants that shape alone doesn't
  guarantee at runtime: a non-empty chunk list, non-empty chunk text, and a
  non-empty access tier. A document that fails validation (e.g. a blank
  file that converted to zero usable chunks) is reported as an
  `IngestionFailure` — the same loud-error, per-file-isolated path as a
  tagging or conversion failure — rather than silently entering the index
  with nothing useful in it.

## 4. Chunking Strategy

- **Hybrid chunking**: chunk at a regular, fixed target size by default. When a
  semantic unit (e.g. a section, list, or table) would otherwise be split
  across a chunk boundary and lose context, extend/adjust the chunk to keep
  that unit intact rather than cutting it mid-way.
- **Implemented as**: Markdown is split into blocks on blank lines (the
  standard Markdown separator between paragraphs, list groups, and tables),
  then blocks are greedily packed into a chunk up to `chunk_size_chars`
  (config, default **2000**, overridable via `.env`). A single block larger
  than `chunk_size_chars` is never split — it becomes its own oversized
  chunk. No character-level overlap between chunks is applied; keeping
  semantic units intact was the stated goal, not overlap, so overlap was not
  added. If retrieval quality later shows overlap is needed, that's a
  follow-up decision, not an assumption baked in now.

## 5. Indexing

- **Vector store: Qdrant**, using **HNSW** indexing for approximate nearest
  neighbor search over dense embeddings.
- **Hybrid search backend: Qdrant native hybrid search** (sparse + dense
  vectors in the same database), rather than standing up a separate keyword
  search engine (e.g. Elasticsearch/OpenSearch). Chosen to minimize the number
  of stateful services that must be run and kept in sync.
- **Sparse vectors: BM25 via `fastembed`** (`Qdrant/bm25`), Qdrant's own
  recommended sparse embedder — runs locally, no server, deterministic per
  text regardless of what else is in a batch (fixed term statistics, not
  corpus-fitted IDF), which is what makes it safe to reindex without churn.
  Tested for real rather than mocked, to actually verify that determinism.
  **Correction from an earlier PR**: this was initially described as "the
  same precedent as `markitdown`," which doesn't hold up — `markitdown`
  needs no network for plain-text conversion, but `fastembed` downloads a
  tokenizer/vocab bundle on first use, and caches it in the **OS temp
  directory**, not a stable location, so this can recur on any environment
  where that cache was cleared. The test suite now has a module-scoped
  `autouse` fixture that skips these tests with a clear reason if the model
  can't be loaded, rather than failing confusingly or silently depending on
  network access every run.
- Every indexed chunk carries metadata required for downstream filtering:
  source document ID, exact source location (for citation), and the
  **access-level/role tag(s)** required to view it (see §11). Implemented
  as the Qdrant point payload: `relative_path`, `chunk_index`, `text` (the
  chunk's own content, for citation without a second lookup), and
  `access_tier`.
- **Upsert is idempotent and edit-safe.** `index_document()` embeds a
  document's full chunk set (dense + sparse) *before* touching the index,
  and only then deletes existing points for that `relative_path` and
  inserts the fresh set. Embedding first — not deleting first — matters: a
  transient embedding failure (Ollama momentarily unreachable, a timeout
  under load) must leave an already-indexed document exactly as it was,
  not silently vanish it from search results with nothing to replace it.
  A length mismatch between the embedded vectors and the chunk count
  (which `zip()` would otherwise truncate to silently, dropping chunks
  with no error) raises loudly instead of partially indexing a document.
  Point IDs are deterministic (`uuid5` of `relative_path` + chunk index),
  so re-running `index_document()` for the same document converges to the
  same points instead of accumulating duplicates — this matters once
  Phase 7 puts ingestion on a schedule that may retry.
- Embedding model: `nomic-embed-text`, served locally via Ollama (pulled and
  verified working — 768-dimensional vectors — during Phase 2). Embedding
  calls use Ollama's batch-capable `/api/embed` endpoint (many texts in one
  HTTP round-trip) rather than the older single-prompt `/api/embeddings`,
  since indexing will need to embed every chunk of every document.
- **Qdrant deployment: local/embedded mode for now.** Docker isn't available
  in this dev environment, so Qdrant runs via `qdrant-client`'s built-in
  local mode (on-disk storage, no server process) rather than a container.
  This is swappable for a real Qdrant server later via config (a URL vs. a
  local path) — the indexing code itself doesn't need to change.
- **HNSW**: Qdrant indexes dense vectors with HNSW by default — there's no
  alternative index to opt into, so `ensure_collection()` creating a
  collection with standard vector params already satisfies this
  requirement, with no manual HNSW tuning needed unless retrieval quality
  later calls for it.
- **Collection schema set up for hybrid search from the start.** Qdrant
  can't add a sparse vector field to a collection after creation — only
  recreate it. `ensure_collection()` therefore creates a named dense vector
  (`"dense"`) *and* a named sparse vector (`"sparse"`) together, even
  though sparse vectors aren't populated until the native hybrid search
  item ships. Confirmed empirically: attempting to add a sparse vector via
  `update_collection()` to an existing dense-only collection fails with
  `ValueError: Vector sparse does not exist in the collection`.
- **Vector-size mismatch is a loud error, not a silent no-op.**
  `ensure_collection()` checks an existing collection's dense vector size
  against what's requested and raises `CollectionSchemaMismatchError` on a
  mismatch (e.g. `EMBEDDING_DIMENSIONS` changed without migrating the
  collection), rather than silently leaving a stale, mismatched collection
  in place until a later upsert fails with an opaque dimension error.

## 6. Retrieval Pipeline (query journey)

End-to-end path from a user query to an answer. See `README.md` for the
visual diagram; this is the authoritative step list:

1. **User query** arrives from the browser/UI.
2. **Orchestrator** rewrites conversation history and the incoming query into
   a single, self-contained query (contextualization for multi-turn chat —
   see §10). This happens on every new user turn.
3. **Embedding**: the rewritten query is embedded (`nomic-embed-text`).
4. **Parallel hybrid search**: dense vector search and sparse/keyword (BM25)
   search run in parallel against Qdrant, each contributing candidates.
5. **Fusion**: the two result sets are merged/fused into a single ranked list;
   the **top 10** combined candidates are kept.
6. **Reranking**: a **local open-source cross-encoder** (e.g.
   `BAAI/bge-reranker-v2-m3`, run locally) reranks the top 10 and selects the
   **top 4** chunks.
7. **Generation**: the top 4 chunks + the grounding rules (§8) + the
   (rewritten) user query are assembled into the final prompt and sent to the
   generation LLM (Mistral/Mixtral via Ollama) to produce the answer.

All retrieval (step 4 onward) is subject to the access-control filter in §11
— a candidate the user isn't permitted to see must never reach step 5, let
alone be cited in an answer.

## 7. Caching

Two caching layers, both required, to meet the speed/reliability target:

- **Embedding cache**: avoid recomputing embeddings for text (queries or
  chunks) that has already been embedded. **Implemented as** `EmbeddingCache`
  + `embed_with_cache()` (`src/agentic_rag/embedding/cache.py`) — an
  in-memory dict keyed on `(model, sha256(text))`, generic over both dense
  and sparse embeddings (the model name already discriminates between
  them, so one shared cache instance covers both). `index_document()`
  takes a required `embedding_cache` parameter rather than creating one
  internally — the cache only pays off when **one instance is shared
  across many `index_document()` calls** (e.g. one per sync cycle), so
  repeated content across *different* documents (boilerplate, headers,
  disclaimers) skips re-embedding. Verified live: an identical chunk
  embedded twice with a shared cache went from ~6.6s (real Ollama call) to
  ~0.006s (cache hit, no network call).
  **Deliberately in-memory only for this first version** — scoped to the
  process's lifetime, not persisted across restarts. A persistent backend
  (what store, eviction policy) is an open question for whenever that
  matters in practice, not designed speculatively now.
- **Semantic cache**: cache answers keyed on query *meaning*, so
  semantically-similar repeat questions can be served without re-running the
  full retrieval + generation pipeline. Not yet implemented — a Phase 3/5
  concern (this is about caching final *answers*, not embeddings).

Cache backend, eviction policy, and semantic-similarity threshold for the
semantic cache are still open — see §14.

## 8. Grounding & Answer Rules

These rules apply to every answer the system produces, with no exceptions:

1. Every factual answer **cites its source** (document + exact chunk) **and**
   the access level that source requires.
2. If the retrieved sources don't contain the answer, the system responds:
   **"I do not know the answer based on indexed documents."** — this is the
   single canonical fallback message, used everywhere the system cannot
   ground an answer (including when the sub-question decomposition loop in
   §10 is exhausted).
3. The system **never** uses knowledge outside the retrieved/sourced
   documents to generate an answer — no reliance on the LLM's parametric
   knowledge for factual claims.

## 9. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | Answer questions from the corpus with citations to the exact source chunk(s). |
| FR2 | Support multi-turn chat: each turn inherits context (history) from previous turns in the same conversation. |
| FR3 | Enforce per-user document permissions (see §11); a user must never see content, or an answer derived from content, above their access level. |
| FR4 | Reflect document edits within minutes and document deletions immediately in retrieval results. |
| FR5 | Say "I do not know" (§8, rule 2) when the corpus has no answer, rather than guessing. |

## 10. Multi-Turn Chat & Query Decomposition

- **History rewriting**: on every new user query, the orchestrator rewrites
  the conversation history plus the new query into one self-contained query
  before it enters the retrieval pipeline (§6, step 2). This is what makes
  FR2 (multi-turn context) work.
- **Sub-question decomposition**: a complex question may be split into
  sub-questions, each run through the retrieval pipeline independently.
- **Retry/replanning loop**: if the evidence retrieved for a (sub-)question is
  insufficient to answer it, the system returns to planning and retries —
  up to **5 turns** total.
- If, after 5 turns, no sufficiently-evidenced answer was found, the system
  returns the canonical fallback message from §8 rule 2. (A single message is
  used for all "couldn't answer" cases — see the design-decisions log in §13.)

## 11. Access Control

- Every document/chunk carries an access-level tag.
- **Model: simple linear tiers.** Access levels form a single ordered list
  (lowest → highest); a user at a given tier can see content tagged at their
  tier **or any tier below it**, matching the stated example ("developer"
  can't see "manager"-only docs; a "manager" can see "developer"-level docs).
- The ordered tier list itself is **configuration, not hardcoded** — it lives
  in the central config module (`.claude/CLAUDE.md` §5) so the real
  organizational role names can be supplied without a code change. A
  placeholder tier list (e.g. `tier-1 < tier-2 < tier-3`) is used for
  development/tests until the real list is provided.
- Access filtering happens **at retrieval time** (§6, step 4/5) — a chunk the
  user isn't permitted to see must be excluded before fusion/reranking, not
  filtered out after the fact.
- **Tagging mechanism (ingestion side): folder-per-tier.** A document's tier
  is its top-level subfolder under `WATCHED_FOLDER_PATH`, e.g.
  `tier-2/report.txt` (nesting further inside that folder is fine —
  `tier-1/quarterly/report.txt` is still tier-1). A document placed directly
  under the watched root, with no tier subfolder, is rejected loudly
  (`UntaggedDocumentError`) rather than silently defaulting to a tier — same
  for a subfolder name that isn't in the configured `ACCESS_TIERS` list
  (`UnknownAccessTierError`). No sidecar files or manifest to keep in sync.
- **Failure isolation is per file, not per batch.** A file that fails at any
  ingestion step — unrecognized access tier, or a conversion error (corrupt
  file, unsupported format, vanished between being detected and being read)
  — is reported as an `IngestionFailure` (relative path + reason) alongside
  the successfully-processed `IngestedDocument`s from the same watcher cycle.
  One bad file never discards everything else that succeeded in the same
  run, and — critically once Phase 7 puts this on a schedule — a single
  persistently-bad file (e.g. a corrupt PDF that will never convert) cannot
  permanently stall ingestion of every other document in the corpus by
  raising on every cycle. This matters directly at the target scale (10,000+
  docs, §2): a batch-abort-on-first-error design would turn one bad file
  into a reliability outage, not just a data-quality blemish.

## 12. Safety & Security Controls

- **Access control** — covered by §11; the first line of defense.
- **Prompt injection detection**: incoming user queries are screened by an
  LLM-based judge for injection attempts before being used in retrieval or
  generation.
- **Output/citation validation**: before an answer is returned, its citation
  links and the underlying chunks are checked for security threats or
  malfunction (e.g. a citation pointing to a chunk the user isn't permitted
  to see, or content indicating a successful injection).
- **Foul language refusal**: the system refuses to engage with foul/abusive
  language at any stage of the conversation.
- Which model performs the injection judge and output-validation checks (the
  local generation model vs. Claude) is an open item — see §14.

## 13. Resolved Design Decisions

Log of decisions made explicitly during planning, for traceability:

| Decision | Choice | Rationale |
|---|---|---|
| Access control model | Simple linear tiers (configurable list) | Simplest to implement/reason about; matches the stated example |
| Keyword search backend | Qdrant native hybrid (sparse+dense) | One database instead of two stateful services |
| Reranker | Local open-source cross-encoder (`bge-reranker-v2-m3`-class) | Consistent with local-first/open-source generation stack |
| Fallback message | Single canonical message everywhere | Simpler to test and guarantee consistency of |
| Document source-of-truth | Watched folder/filesystem | No separate upload service to build; fits documents already managed as files |
| Chunk size | 2000 chars, no overlap, block-based (blank-line-separated) boundary detection | Simple, dependency-free, keeps semantic units intact per §4; overlap wasn't part of the stated requirement so it was left out rather than assumed |
| Access-tier tagging mechanism | Folder-per-tier under the watched root | No extra file format to maintain; matches the watched-folder source of truth |

## 14. Open Items (need a decision before the relevant phase starts)

- **Injection judge / output validation model**: local model (fast, no
  external cost) vs. Claude (likely higher judgment quality, adds latency +
  external API dependency). Needs a decision before Phase 6.
- **Real access-tier names**: the linear tier list is a placeholder (§11)
  until the actual organizational roles are provided.
- **Cache backend and semantic-similarity threshold** for the semantic cache
  (§7).
