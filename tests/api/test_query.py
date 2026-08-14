from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from agentic_rag.api.app import create_app
from agentic_rag.config import Settings
from agentic_rag.orchestration.answer import AnswerResult, Citation
from agentic_rag.orchestration.rewrite import ConversationTurn
from agentic_rag.retrieval.access import UnknownAccessTierError


def _test_settings(tmp_path: Path) -> Settings:
    return Settings(
        watched_folder_path=tmp_path / "corpus",
        qdrant_storage_path=tmp_path / "qdrant",
        _env_file=None,
    )


def _client(tmp_path: Path) -> TestClient:
    app = create_app(_test_settings(tmp_path))
    return TestClient(app)


def test_query_returns_the_answer_from_answer_with_cache(tmp_path):
    with _client(tmp_path) as client:
        with (
            patch("agentic_rag.api.routers.query.rewrite_query", return_value="rewritten?"),
            patch(
                "agentic_rag.api.routers.query.answer_with_cache",
                return_value=AnswerResult(text="the answer", citations=[]),
            ),
        ):
            response = client.post(
                "/query", json={"query": "who won?", "user_tier": "tier-1", "history": []}
            )

    assert response.status_code == 200
    assert response.json() == {"answer": "the answer", "citations": []}


def test_query_returns_citations_resolving_each_source(tmp_path):
    citation = Citation(
        number=1, relative_path="tier-1/derby.md", chunk_index=2, access_tier="tier-1"
    )
    with _client(tmp_path) as client:
        with (
            patch("agentic_rag.api.routers.query.rewrite_query", return_value="rewritten?"),
            patch(
                "agentic_rag.api.routers.query.answer_with_cache",
                return_value=AnswerResult(text="Arsenal won [1].", citations=[citation]),
            ),
        ):
            response = client.post(
                "/query", json={"query": "who won?", "user_tier": "tier-1", "history": []}
            )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Arsenal won [1].",
        "citations": [
            {
                "number": 1,
                "relative_path": "tier-1/derby.md",
                "chunk_index": 2,
                "access_tier": "tier-1",
            }
        ],
    }


def test_query_rewrites_history_before_retrieval(tmp_path):
    with _client(tmp_path) as client:
        with (
            patch(
                "agentic_rag.api.routers.query.rewrite_query", return_value="rewritten?"
            ) as mock_rewrite,
            patch("agentic_rag.api.routers.query.answer_with_cache", return_value=AnswerResult(text="the answer", citations=[])),
        ):
            client.post(
                "/query",
                json={
                    "query": "and the second leg?",
                    "user_tier": "tier-1",
                    "history": [{"user_query": "who won the first leg?", "assistant_answer": "Arsenal, 2-1."}],
                },
            )

    history_arg, query_arg = mock_rewrite.call_args.args
    assert history_arg == [ConversationTurn("who won the first leg?", "Arsenal, 2-1.")]
    assert query_arg == "and the second leg?"


def test_query_passes_the_rewritten_query_and_user_tier_to_answer_with_cache(tmp_path):
    with _client(tmp_path) as client:
        with (
            patch(
                "agentic_rag.api.routers.query.rewrite_query",
                return_value="who won the second leg of the Arsenal tie?",
            ),
            patch(
                "agentic_rag.api.routers.query.answer_with_cache", return_value=AnswerResult(text="the answer", citations=[])
            ) as mock_answer,
        ):
            client.post(
                "/query",
                json={"query": "and the second leg?", "user_tier": "tier-2", "history": []},
            )

    query_arg, tier_arg = mock_answer.call_args.args
    assert query_arg == "who won the second leg of the Arsenal tie?"
    assert tier_arg == "tier-2"


def test_query_rejects_an_empty_query(tmp_path):
    with _client(tmp_path) as client:
        response = client.post("/query", json={"query": "", "user_tier": "tier-1", "history": []})

    assert response.status_code == 422


def test_query_rejects_a_whitespace_only_query(tmp_path):
    with _client(tmp_path) as client:
        response = client.post(
            "/query", json={"query": "   ", "user_tier": "tier-1", "history": []}
        )

    assert response.status_code == 422


def test_query_returns_422_for_an_unknown_user_tier(tmp_path):
    with _client(tmp_path) as client:
        with (
            patch("agentic_rag.api.routers.query.rewrite_query", return_value="who won?"),
            patch(
                "agentic_rag.api.routers.query.answer_with_cache",
                side_effect=UnknownAccessTierError("'admin' is not a known access tier"),
            ),
        ):
            response = client.post(
                "/query", json={"query": "who won?", "user_tier": "admin", "history": []}
            )

    assert response.status_code == 422
    assert "admin" in response.json()["detail"]


def test_query_rejects_a_missing_user_tier(tmp_path):
    with _client(tmp_path) as client:
        response = client.post("/query", json={"query": "who won?", "history": []})

    assert response.status_code == 422


def test_query_defaults_history_to_empty(tmp_path):
    with _client(tmp_path) as client:
        with (
            patch(
                "agentic_rag.api.routers.query.rewrite_query", return_value="who won?"
            ) as mock_rewrite,
            patch("agentic_rag.api.routers.query.answer_with_cache", return_value=AnswerResult(text="the answer", citations=[])),
        ):
            response = client.post("/query", json={"query": "who won?", "user_tier": "tier-1"})

    assert response.status_code == 200
    history_arg, _query_arg = mock_rewrite.call_args.args
    assert history_arg == []
