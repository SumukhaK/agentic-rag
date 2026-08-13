from unittest.mock import patch

import pytest

from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.embedding.sparse_client import SparseEmbeddingError, embed_sparse_texts
from agentic_rag.indexing.qdrant_setup import ensure_collection, get_client
from agentic_rag.indexing.upsert import index_document
from agentic_rag.ingestion.chunker import Chunk
from agentic_rag.ingestion.pipeline import IngestedDocument
from agentic_rag.retrieval.access import UnknownAccessTierError
from agentic_rag.retrieval.search import hybrid_search

SPARSE_MODEL = "Qdrant/bm25"
COLLECTION = "documents"
KNOWN_TIERS = ["tier-1", "tier-2", "tier-3"]


@pytest.fixture(scope="module", autouse=True)
def _require_sparse_model():
    try:
        embed_sparse_texts(["warmup"], model_name=SPARSE_MODEL)
    except SparseEmbeddingError as exc:
        pytest.skip(f"sparse embedding model unavailable: {exc}")


@pytest.fixture(autouse=True)
def _mock_dense_embeddings():
    # index_document() (upsert.py) and hybrid_search() (search.py) each
    # import their own reference to embed_texts, so both need mocking -
    # patching only one leaves the other hitting the real (768-dim) Ollama
    # server against a test collection sized for the mocked (3-dim) vector.
    with patch("agentic_rag.indexing.upsert.embed_texts") as mock_upsert, patch(
        "agentic_rag.retrieval.search.embed_texts"
    ) as mock_search:
        mock_upsert.return_value = [[0.1, 0.2, 0.3]]
        mock_search.return_value = [[0.1, 0.2, 0.3]]
        yield


@pytest.fixture
def client(tmp_path):
    client = get_client(tmp_path)
    ensure_collection(client, collection_name=COLLECTION, vector_size=3)
    return client


def _index(client, relative_path, text, access_tier):
    doc = IngestedDocument(
        relative_path=relative_path,
        markdown=text,
        chunks=[Chunk(text=text, index=0)],
        access_tier=access_tier,
    )
    index_document(
        client,
        collection_name=COLLECTION,
        document=doc,
        embedding_model="nomic-embed-text",
        ollama_base_url="http://localhost:11434",
        sparse_model=SPARSE_MODEL,
        embedding_timeout_seconds=30,
        embedding_cache=EmbeddingCache(),
    )


def _search(client, query, user_tier, top_k=10, known_tiers=None):
    return hybrid_search(
        client,
        collection_name=COLLECTION,
        query=query,
        embedding_model="nomic-embed-text",
        ollama_base_url="http://localhost:11434",
        embedding_timeout_seconds=30,
        sparse_model=SPARSE_MODEL,
        embedding_cache=EmbeddingCache(),
        user_tier=user_tier,
        known_tiers=known_tiers or KNOWN_TIERS,
        top_k=top_k,
    )


def test_hybrid_search_returns_a_matching_candidate(client):
    _index(client, "tier-1/a.txt", "Arsenal drew 1-1 against Chelsea.", "tier-1")

    results = _search(client, "Arsenal Chelsea draw", user_tier="tier-1")

    assert len(results) == 1
    candidate = results[0]
    assert candidate.relative_path == "tier-1/a.txt"
    assert candidate.chunk_index == 0
    assert candidate.text == "Arsenal drew 1-1 against Chelsea."
    assert candidate.access_tier == "tier-1"
    assert candidate.score > 0


def test_hybrid_search_excludes_candidates_above_the_users_tier(client):
    _index(client, "tier-1/a.txt", "Arsenal drew 1-1 against Chelsea.", "tier-1")
    _index(client, "tier-2/b.txt", "Arsenal internal transfer budget report.", "tier-2")

    results = _search(client, "Arsenal", user_tier="tier-1")

    assert [c.relative_path for c in results] == ["tier-1/a.txt"]


def test_hybrid_search_includes_candidates_at_or_below_the_users_tier(client):
    _index(client, "tier-1/a.txt", "Arsenal drew 1-1 against Chelsea.", "tier-1")
    _index(client, "tier-2/b.txt", "Arsenal internal transfer budget report.", "tier-2")

    results = _search(client, "Arsenal", user_tier="tier-2")

    assert {c.relative_path for c in results} == {"tier-1/a.txt", "tier-2/b.txt"}


def test_hybrid_search_respects_top_k(client):
    for i in range(5):
        _index(client, f"tier-1/{i}.txt", f"Arsenal match report number {i}.", "tier-1")

    results = _search(client, "Arsenal", user_tier="tier-1", top_k=3)

    assert len(results) <= 3


def test_hybrid_search_raises_for_an_unknown_user_tier(client):
    with pytest.raises(UnknownAccessTierError):
        _search(client, "Arsenal", user_tier="not-a-tier")
