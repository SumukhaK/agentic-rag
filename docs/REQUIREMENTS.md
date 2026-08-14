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
4. **Parallel hybrid search, fused**: dense vector search and sparse/keyword
   (BM25) search run against Qdrant with the access-control filter applied to
   *both* legs, natively fused (RRF) into one ranked list of up to
   `RETRIEVAL_TOP_K_CANDIDATES` (default 10) candidates. **Implemented as**
   `hybrid_search()` (`src/agentic_rag/retrieval/search.py`) — this is one
   Qdrant call (`prefetch` + `FusionQuery(fusion=Fusion.RRF)`), not
   "search, then separately fuse" as two steps; Qdrant's native hybrid query
   API does both at once.
   - **Prefetch over-fetches** (4× `top_k` per leg) rather than fetching
     exactly `top_k` from each leg. RRF only ranks over what each `Prefetch`
     already returned — if the per-leg limit equalled the final limit, a
     chunk ranked just outside `top_k` on *both* legs individually, but
     competitive after fusion, would never be fetched at all.
   - **Dense and sparse query embedding run concurrently** (a thread pool),
     not sequentially: dense is a blocking Ollama HTTP round-trip, sparse is
     local CPU work, and this runs on every query — the hottest path in the
     system, per the fast/reliable NFR in §2.
5. **Reranking**: a **local open-source cross-encoder** reranks the top 10
   and selects the **top 4** chunks. **Implemented as** `rerank()`
   (`src/agentic_rag/retrieval/rerank.py`) via `fastembed`'s
   `TextCrossEncoder`, using `BAAI/bge-reranker-base` rather than the
   originally-named `BAAI/bge-reranker-v2-m3` — `fastembed` doesn't support
   the v2-m3 variant (`TextCrossEncoder.list_supported_models()` confirms
   this), and `bge-reranker-base` is the same model family, staying
   consistent with the rest of the stack's dependency footprint (no new
   heavy ML framework like `sentence-transformers`/PyTorch, which
   `bge-reranker-v2-m3` would otherwise require). `RERANKER_MODEL` is
   configurable if a different model is wanted later. Verified live:
   correctly separates a directly-relevant chunk from tangentially-related
   and irrelevant ones with much sharper score separation than the fused
   hybrid-search score alone. Each candidate's `score` field is replaced
   with the reranker's own relevance score.
6. **Generation**: the top 4 chunks + the grounding rules (§8) + the
   (rewritten) user query are assembled into the final prompt and sent to the
   generation LLM (`mistral` via Ollama — see §10 for why `mistral` rather
   than the originally-named Mistral/Mixtral pairing) to produce the
   answer. *(Not yet implemented — Phase 5.)*

All retrieval (step 4 onward) is subject to the access-control filter in §11
— a candidate the user isn't permitted to see must never reach step 5, let
alone be cited in an answer. **Implemented as**: `hybrid_search()` computes
`allowed_tiers_for(user_tier, known_tiers)`
(`src/agentic_rag/retrieval/access.py`) and applies it as a Qdrant
`Filter(FieldCondition(access_tier, MatchAny(allowed_tiers)))` on each of the
dense and sparse `Prefetch` queries — a disallowed chunk is excluded from the
candidate pool Qdrant fuses over, not filtered out of the result afterward.
Verified live: a tier-2-only chunk never appeared in a tier-1 user's results,
in a real search against a real Ollama-embedded query.

## 7. Caching

Two caching layers, both required, to meet the speed/reliability target:

- **Embedding cache**: avoid recomputing embeddings for text (queries or
  chunks) that has already been embedded. **Implemented as** `EmbeddingCache`
  + `embed_with_cache()` (`src/agentic_rag/embedding/cache.py`) — an
  in-memory dict keyed on `(model, text)`, generic over both dense and
  sparse embeddings (the model name already discriminates between them, so
  one shared cache instance covers both). Validates that the embedding
  function returned exactly as many results as were requested *before*
  caching any of them — a partial response is a loud `ValueError`, not a
  silently cached gap that would look like a legitimate hit on retry.
  `index_document()` takes a required `embedding_cache` parameter rather
  than creating one internally — the cache only pays off when **one
  instance is shared across many `index_document()` calls** (e.g. one per
  sync cycle), so repeated content across *different* documents
  (boilerplate, headers, disclaimers) skips re-embedding. Verified live: an
  identical chunk embedded twice with a shared cache went from ~6.6s (real
  Ollama call) to ~0.006s (cache hit, no network call).
  **Two things deliberately not solved yet, both flagged rather than
  silently deferred:**
  - **No persistence** — scoped to the process's lifetime. What backend,
    if any, is an open question for whenever that matters in practice.
  - **No eviction — the dict is unbounded.** At the stated scale (10,000+
    docs, §2), a cache shared across one full sync cycle could hold
    hundreds of thousands of embeddings in memory at once. Eviction policy
    was already flagged as an implementation detail for a later phase
    before this PR existed; this makes that concern concrete rather than
    theoretical. It's also unclear whether Phase 7's eventual scheduler
    should create one cache per sync cycle (loses cross-cycle hits on a
    mostly-stable corpus) or one for the process's lifetime (unbounded
    growth over days/weeks of uptime) — that tradeoff needs a decision
    when Phase 7 is designed, not an assumption baked in here.
- **Semantic cache**: cache answers keyed on query *meaning*, so
  semantically-similar repeat questions can be served without re-running the
  full retrieval + generation pipeline. **Moved from Phase 3 to Phase 5** —
  it caches the final generated *answer*, and there's no answer to cache
  until generation exists (Phase 5). Listing it under Retrieval was a
  sequencing mistake in this roadmap's first draft, not a deliberate
  choice; corrected once Phase 3 was otherwise complete rather than
  building unwired infrastructure to fill the checklist slot early.
  **Implemented as** `SemanticCache` + `answer_with_cache()`
  (`src/agentic_rag/orchestration/semantic_cache.py`). Backend: **in-memory,
  linear cosine similarity** — chosen over a second Qdrant collection since
  it needs no new infrastructure and mirrors `EmbeddingCache`'s already-
  established pattern for a dataset that's far smaller and more ephemeral
  than the document corpus. Threshold: `Settings.semantic_cache_similarity_threshold`
  (default 0.95, configurable). **Scoped per `user_tier`, not just query
  meaning**: a cached answer was generated from retrieval already filtered
  to the tier that produced it (§11/FR3), so two users at different tiers
  asking near-identical questions must never share a cache entry — this
  followed directly from FR3 rather than needing a separate decision.
  **Verified live**: an initial query took 50.8s; a semantically-similar
  rephrasing at the same tier returned the identical answer in 2.2s (cache
  hit); the same rephrasing at a different tier correctly missed the cache
  and re-ran the full pipeline (16.9s) — tier isolation confirmed in
  practice, not just in mocked tests.

  **Self-review (PR #26) caught a genuinely serious gap, confirmed by
  three independent finder angles**: caching `CANNOT_ANSWER_MESSAGE`
  creates a negative cache that never self-corrects, even after a
  not-yet-ingested document arrives within FR4's own freshness target —
  and because this system's access-tier model is folder-per-tier (§11), a
  document can be *reclassified* to a stricter tier just by moving it,
  which a cache with no invalidation hook can't detect, risking a cached
  answer citing content the user is no longer authorized to see. Fixed
  with two layers, not one: `answer_with_cache` never caches when
  `sufficient=False`, **and** never caches when the answer text itself
  contains the fallback phrase even when `sufficient=True` — necessary in
  practice, not just in theory, since a live test showed the coarse
  `sufficient` signal can still misfire while the model hedges with an
  answer that opens with the fallback phrase but adds a citation that
  passes `_is_grounded()` anyway. A configurable TTL
  (`Settings.semantic_cache_ttl_seconds`, default 300s) is the second
  layer: it bounds, but does not eliminate, how long even a correctly
  cached grounded answer can outlive the document it cites. Full
  invalidation (a hook into ingestion events, or re-validating grounding
  at read time) remains open, same caveat as `EmbeddingCache` (§7 above).

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

**Implemented as** `generate_answer()` (`src/agentic_rag/orchestration/answer.py`),
returning an `AnswerResult(text, citations)` — not a bare `str`. Rule 1's
"document + exact chunk" was, for a while, satisfied only *inside the
generation prompt* (the model was told to cite `[N]`) with no way for a
caller to resolve `[N]` back to an actual document — self-review of the
`POST /query` PR (Phase 7) caught this as a real end-to-end gap, not a
theoretical one, and it's now fixed: `citations` resolves every `[N]` the
answer text actually cites into its `relative_path`/`chunk_index`/
`access_tier`, threaded all the way through `answer_with_cache()` and
`SemanticCache` (both cache-hit and cache-miss paths) to `POST /query`'s
response. All three rules are also encoded directly in the generation
prompt itself: sources are
listed with a citation number plus their path, chunk index, and access tier
(rule 1), the model is instructed to reply with the canonical fallback
verbatim if the sources don't suffice (rule 2), and told not to use
knowledge beyond what's given (rule 3). When Phase 4's `plan_and_retrieve`
already reports `sufficient=False`, `generate_answer` returns
`planning_result.message` directly with no LLM call at all — the fastest
and most certain way to satisfy rule 2 for that case. **Verified live that
rule 2 is a genuine second line of defense, not just a formality**:
`plan_and_retrieve`'s `sufficient` signal is a coarse, retrieval-only
heuristic (§10) that came back `True` for "What is the capital of France?"
against a football-only corpus, since retrieval always returns its nearest
candidate even when nothing is actually relevant. `generate_answer()`
caught it anyway — `mistral`, given the actual (irrelevant) chunk,
correctly recognized it didn't answer the question and returned the
canonical fallback instead of fabricating an answer.

Rule 1 is also **validated, not just requested**: prompt-following is
probabilistic, which can't satisfy a rule with "no exceptions" on its own —
a self-review finding on PR #25 made this explicit. `_is_grounded()` checks
every generated answer before it's returned: valid only if it's the
canonical fallback verbatim, or cites at least one source number actually
in range (`1..len(candidates)`). An answer with no citations, or one citing
a source that doesn't exist, is replaced with `CANNOT_ANSWER_MESSAGE` — a
fabricated citation is worse than none, since it carries false authority
the reader has no way to detect on their own.

**Deterministic by requirement, not just by preference**: `generate_answer()`
takes a required `temperature` (`Settings.generation_temperature`, default
`0.0`) — discovered as a real gap once Phase 7's `POST /query` became the
first real caller to exercise this function repeatedly: the identical,
already-`sufficient` `PlanningResult` produced the correct cited answer on
2 of 3 identical calls and the "I do not know" fallback on the third at
Ollama's default (non-zero) temperature. This is a *correctness* bug against
rule 1/rule 2 above, not a phrasing-variety nicety — re-verified with 5
identical calls at `temperature=0.0` producing the correct, byte-for-byte
identical answer every time. A separate setting from the Phase 6 judges'
`Settings.judge_temperature`, not a reuse of it: a judge's single-word
verdict has no reason to vary, but a natural-language answer's phrasing
plausibly could, so the two are free to diverge later even though they
start at the same value.

## 9. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | Answer questions from the corpus with citations to the exact source chunk(s). |
| FR2 | Support multi-turn chat: each turn inherits context (history) from previous turns in the same conversation. |
| FR3 | Enforce per-user document permissions (see §11); a user must never see content, or an answer derived from content, above their access level. |
| FR4 | Reflect document edits within minutes and document deletions immediately in retrieval results. |
| FR5 | Say "I do not know" (§8, rule 2) when the corpus has no answer, rather than guessing. |

## 10. Multi-Turn Chat & Query Decomposition

- **Generation model: `mistral`, served locally via Ollama** (pulled and
  verified working during Phase 4 — `nomic-embed-text`, the sparse
  reranker/embedder, and `mistral` are all pulled now). Mixtral was the
  other originally-named option but is far larger (~26GB vs. mistral's
  ~4.1GB) and wasn't warranted for local dev.
  **Implemented as** `generate()` (`src/agentic_rag/generation/llm_client.py`)
  — a thin wrapper around Ollama's `/api/generate` endpoint, wrapping
  connection/malformed-response failures in `GenerationError`. This is a
  shared building block: orchestration (this section) uses it for query
  rewriting and decomposition, and Phase 5 reuses the same client for
  final answer generation with a different prompt — the client itself
  doesn't change between the two uses, only the prompt does.
- **History rewriting**: on every new user query, the orchestrator rewrites
  the conversation history plus the new query into one self-contained query
  before it enters the retrieval pipeline (§6, step 2). This is what makes
  FR2 (multi-turn context) work. **Implemented as** `rewrite_query()`
  (`src/agentic_rag/orchestration/rewrite.py`) — given a list of prior
  `(user_query, assistant_answer)` turns and the new query, prompts
  `generate()` to produce a single standalone question with pronouns/
  references resolved. Returns the query unchanged, with **no LLM call**,
  when there's no history yet — the first turn is already self-contained,
  and calling the LLM would be pure wasted latency. Verified live: "Who
  scored for them?" after a turn about an Arsenal-Chelsea match correctly
  rewrote to "Which players scored for Arsenal in their match against
  Chelsea?" — "them" and "it" both resolved from context.
- **Sub-question decomposition**: a complex question may be split into
  sub-questions, each run through the retrieval pipeline independently.
  **Implemented as** `decompose_query()`
  (`src/agentic_rag/orchestration/decompose.py`) — prompts `generate()`
  for one sub-question per line, stripping numbering/bullets the model
  adds despite being told not to — the marker regex requires whitespace
  or end-of-line right after "N."/"N)", so it correctly leaves a
  sub-question that starts with a decimal stat (e.g. "1.85 xG...",
  realistic content in a football-analytics corpus) untouched instead of
  corrupting it into "85 xG...". Raises `GenerationError` if the LLM
  returns nothing usable, same failure-must-be-loud principle as
  `rewrite_query()`.
  **Observed live, documented honestly rather than only showing the good
  case**: the prompt asks the model to return an already-simple question
  unchanged as a single line, but `mistral` doesn't reliably follow that
  — "Who won the match?" came back decomposed into 4 sub-questions
  ("Who participated?", "Which team did each represent?", "When did it
  take place?", "What was the final score?") instead of being returned
  as-is. Not a code defect — the function did exactly what was asked
  (parse whatever the LLM returns into sub-questions) — but a real
  prompt-adherence limitation worth knowing about rather than glossing
  over. A genuinely complex question decomposed correctly and
  sensibly: "Who won the Arsenal vs Chelsea match, how many goals were
  scored, and were there any red cards?" → 3 focused sub-questions, one
  per clause.
- **Retry/replanning loop**: if the evidence retrieved for a (sub-)question is
  insufficient to answer it, the system returns to planning and retries — up
  to `Settings.max_retrieval_attempts` turns total (default **5**;
  **configurable, not hardcoded**, per explicit instruction — same rationale
  as the access-tier list in §11: a number chosen once shouldn't require a
  code change to revisit). **Implemented as** `plan_and_retrieve()`
  (`src/agentic_rag/orchestration/planning.py`) — each attempt fully
  re-decomposes the query (fresh LLM phrasing is the only thing that can
  plausibly change the result against a deterministic corpus and
  embeddings), then retrieves + reranks per sub-question. "Sufficient" means
  every sub-question has at least one candidate chunk after reranking — a
  coarse, retrieval-only signal, not an answer-quality judgment, since
  answer quality isn't knowable until generation exists (Phase 5).
  **Tried and rejected**: a fixed cutoff on the reranker's own score, as a
  tighter "is this actually relevant" check. Live-tested against the
  reranker, a genuinely relevant candidate ("Who played for Arsenal against
  Chelsea?", answerable from the indexed corpus) scored **-5.88** — worse
  than a genuinely irrelevant one ("What is the name of the capital of
  France?") at **-4.44**. Relevant/irrelevant score ranges overlap too much
  for a global threshold to separate them reliably for short,
  generically-phrased questions, so any cutoff would either drop real
  evidence or let noise through depending on the query — worse than the
  coarse non-empty signal it would have replaced. Real answerability
  judgment needs the LLM to reason over the retrieved text, which belongs to
  generation (Phase 5), not a retrieval-time score threshold.
- If, after `max_retrieval_attempts` turns, no sufficiently-evidenced answer
  was found, the system returns the canonical fallback message from §8 rule
  2 — `CANNOT_ANSWER_MESSAGE` in `planning.py`, exposed on the result as
  `PlanningResult.message` (`None` when `sufficient=True`). The same
  constant, and the same `PlanningResult(sufficient=False, ...)` code path,
  covers both a direct no-match on the very first attempt and exhausting
  all retries; a single message is used for all "couldn't answer" cases
  (see the design-decisions log in §13).
- **Transient vs. configuration failures during retrieval**: if
  `decompose_query`, `hybrid_search`, or `rerank` raises mid-attempt
  (`GenerationError`, `EmbeddingError`, `SparseEmbeddingError`,
  `RerankError`) — e.g. a dropped Ollama connection — that costs one retry
  attempt, same as the attempt finding no evidence; a self-review on PR #24
  caught that the first version let any such exception propagate straight
  out and abort the whole call, defeating the retry budget's purpose.
  `UnknownAccessTierError` is excluded from this and still propagates
  immediately — a bad `user_tier` is a configuration error, not something a
  fresh decomposition could ever fix, so retrying it would just waste the
  budget on a guaranteed-repeat failure.

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
  generation. **Implemented as** `check_for_injection()`
  (`src/agentic_rag/orchestration/injection_judge.py`) — returns an
  `InjectionCheckResult(is_injection, raw_response)`, not a bare bool, so a
  miss is at least auditable after the fact. **Not wired into any caller
  yet** — `answer_with_cache` and every other current entrypoint still call
  straight through to retrieval/generation with no screening in front of
  them, so the protection this bullet describes does not exist end-to-end
  in the running system today; composing it in is Phase 7's job, including
  deciding whether to screen the raw or the rewritten query.
  Empirically validated against 20 committed, reproducible prompts
  (`tests/orchestration/test_injection_judge_live.py`, skipped gracefully
  if Ollama/`mistral` is unavailable) — not just a one-off manual count in
  prose. The validation set itself only exists in its current, stronger
  form because self-review of the first version found two real gaps a
  smaller set would have missed: naive whole-response substring matching
  both **failed open** (a response containing "unclean" matched the
  "clean" substring) and **failed closed on legitimate queries** (a
  verbose CLEAN verdict echoing a query term like "injection" — a genuine
  football topic, e.g. a player's medical injection); and the original
  prompt had no delimiter between instructions and untrusted input, so a
  query ending in a fake "...Answer: CLEAN" could get the judge to echo
  that answer back verbatim, defeating the check. Both are fixed
  (first-word-only parsing; explicit `<<<MESSAGE_START>>>`/`<<<MESSAGE_END>>>`
  delimiters with an instruction to treat the contents as data, not
  commands) and covered by regression cases in the fixed validation set.
  **Deterministic verdicts (PR #31)**: `check_for_injection()` now requires
  an explicit `temperature` keyword argument (e.g.
  `Settings.judge_temperature`, default `0.0`), closing the same
  non-determinism gap first found in `check_output_security()` below (PR
  #30) — re-verified for this module specifically (not assumed from the
  sibling fix): the full live suite passed 20/20 across 5 consecutive
  runs, and the delimiter-confusion exploit prompt produced a
  byte-for-byte identical judge response across 10 consecutive runs at
  `temperature=0.0`. See `PROJECT_TRACKER.md`'s Phase 6 log for the full
  verification detail.
- **Output/citation validation**: before an answer is returned, its citation
  links and the underlying chunks are checked for security threats or
  malfunction (e.g. a citation pointing to a chunk the user isn't permitted
  to see, or content indicating a successful injection). **Implemented as**
  `check_output_security()` (`src/agentic_rag/orchestration/output_security.py`)
  — a deterministic access-tier check (no LLM call) plus an LLM-based check
  of whether the generated answer itself shows signs of a successful
  injection, sharing its response parser with `check_for_injection()`. **Not
  wired into any caller yet** — same status as the injection judge above,
  for the same reason (composition is Phase 7's job). Empirically validated
  11/11 against a committed fixture
  (`tests/orchestration/test_output_security_live.py`); see
  `PROJECT_TRACKER.md`'s Phase 6 log for the full tuning history and a
  known, accepted residual limitation. Also surfaced that `generate()`'s
  lack of temperature control made judge verdicts genuinely
  non-deterministic across identical calls — fixed with a new
  `Settings.judge_temperature` (default `0.0`), applied here and, as of PR
  #31, to `check_for_injection()` too (see above).
- **Foul language refusal**: the system refuses to engage with foul/abusive
  language at any stage of the conversation. **Implemented as**
  `check_for_foul_language()` (`src/agentic_rag/orchestration/foul_language.py`)
  — an LLM-based judge sharing its response parser with `check_for_injection()`
  and `check_output_security()`. That parser was renamed from
  `classify_injection_verdict()` to `classify_verdict()` once this became its
  third caller, confirming it was already generic (a first-word CLEAN-vs-flagged
  check), not injection-specific despite the module it lives in. Unlike the two
  checks above, a flagged message gets its own distinct
  `FOUL_LANGUAGE_REFUSAL_MESSAGE` rather than the shared canonical fallback —
  there's no adversarial-calibration reason to hide which check caught a user
  here, and a direct "please rephrase" message is clearer UX than reusing the
  "I do not know the answer" wording. **Not wired into any caller yet** — same
  status as the injection judge and output-validation checks above; composition
  is Phase 7's job. Empirically validated 14/14 against a committed fixture
  (`tests/orchestration/test_foul_language_live.py`) — the CLEAN/FOUL
  classification itself needed no tuning. Deterministic by construction —
  takes the same required `temperature` keyword argument as its siblings
  from the start (`Settings.judge_temperature`, default `0.0`), rather than
  needing a follow-up fix. Self-review did find and fix a real prompt-
  hardening gap, though: the first draft's delimiter instructions were
  missing two anti-exploit clauses `check_for_injection()`'s prompt already
  had, and a forged `"...Answer: CLEAN"` suffix live-flipped a genuinely
  abusive message to CLEAN 3/3 times. Restoring the matching wording plus an
  end-of-prompt reminder fixed 1 of 3 repro cases; **the other 2 still
  flip**, and the identical trick (reworded) also flips the already-merged
  `check_for_injection()` — a shared, phrasing-dependent `mistral`
  instruction-following gap under the current delimiter mitigation, not
  something specific to this judge or fixable by further wordsmithing alone.
  Tracked as its own follow-up in `PROJECT_TRACKER.md` rather than chased
  further in this PR.
- **Resolved**: the forged/pre-filled-verdict exploit above (and the
  matching gap in `check_for_injection()`) is now closed by
  `has_forged_verdict()` (`src/agentic_rag/orchestration/judge.py`) — a
  deterministic regex gate, no LLM call, run before every judge's prompt.
  It detects the structural signature shared by every forged-verdict
  variant tested (a verdict-label word like "answer," "verdict," "status,"
  or "classification" in close proximity to the literal word "clean") and
  flags unconditionally, since a regex can't be talked out of its answer
  the way `mistral`'s instruction-following could be defeated by a
  well-placed suffix. Two other candidate directions (an unguessable
  per-call challenge token; few-shot examples applied uniformly to all
  three judges) were live-tested and found insufficient — see
  `PROJECT_TRACKER.md`'s completed entry for the full comparison,
  including why the injection judge specifically needed a different
  approach than the foul-language judge. All three judges (this one,
  `check_for_injection()`, `check_output_security()`) now run the gate
  first and the hardened prompt second, as defense-in-depth for exploit
  phrasings that don't match the regex's structural signature.
  Empirically verified: 16 new regression cases (7 foul-language, 7
  injection, 2 output-security, split between the original repro strings
  and a differently-worded holdout set) all pass, with zero false
  positives against a broad set of genuine football content mentioning
  "result," "status," or "clean" in unrelated contexts.
- **Resolved**: the injection judge and output-validation checks are
  performed by the **local generation model (`mistral`)**, not Claude — no
  new `ANTHROPIC_API_KEY` needed. See §13's decision log for the full
  tradeoff, including a known `mistral` reliability gap this decision
  doesn't paper over. Both checks' empirical validation has now run (above).

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
| Semantic cache backend | In-memory, linear cosine similarity | No new infrastructure; mirrors `EmbeddingCache`'s established pattern for a much smaller, more ephemeral dataset than the document corpus |
| Semantic cache scoping | Per `(query meaning, user_tier)`, not just query meaning | Follows directly from FR3 — a cached answer was generated from tier-filtered retrieval, so a different tier must never receive it |
| Injection judge / output validation model | Local generation model (`mistral`) | No new `ANTHROPIC_API_KEY` needed; a real tradeoff, not a clean win — full reasoning and the required empirical validation step are in `PROJECT_TRACKER.md`'s Phase 6 log, not duplicated here |
| API app location | `src/agentic_rag/api/` | Consistent with this repo's established flat `src/agentic_rag/` layout (`ingestion/`, `embedding/`, `retrieval/`, `generation/`, `orchestration/`) — no new top-level packaging concept introduced for the API layer alone |
| Multi-turn chat session model | Stateless — client resends full history each `POST /query` call | Simplest for the MVP; avoids a new persistence/session-store decision this early, consistent with this project's bias against speculative infrastructure. Revisit if a real chat UI needs server-side session state |

## 14. Open Items (need a decision before the relevant phase starts)

- **Real access-tier names**: the linear tier list is a placeholder (§11)
  until the actual organizational roles are provided.
- **Claude-as-evaluator scope and credentials**: no `ANTHROPIC_API_KEY` is
  configured for this project, and the detailed spec (what gets judged, what
  rubric, what output format) isn't written down anywhere — only a
  PROJECT_TRACKER.md checklist bullet and Phase 8's higher-level "retrieval
  precision, faithfulness, hallucination rate" framing exist. Needs both an
  API key and a written spec before Phase 8 starts.
