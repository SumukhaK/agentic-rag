from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from agentic_rag.api.app import create_app
from agentic_rag.config import Settings
from agentic_rag.orchestration.rewrite import ConversationTurn


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
            patch("agentic_rag.api.routers.query.answer_with_cache", return_value="the answer"),
        ):
            response = client.post(
                "/query", json={"query": "who won?", "user_tier": "tier-1", "history": []}
            )

    assert response.status_code == 200
    assert response.json() == {"answer": "the answer"}


def test_query_rewrites_history_before_retrieval(tmp_path):
    with _client(tmp_path) as client:
        with (
            patch(
                "agentic_rag.api.routers.query.rewrite_query", return_value="rewritten?"
            ) as mock_rewrite,
            patch("agentic_rag.api.routers.query.answer_with_cache", return_value="the answer"),
        ):
            client.post(
                "/query",
                json={
                    "query": "and the second leg?",
                    "user_tier": "tier-1",
                    "history": [{"user_query": "who won the first leg?", "assistant_answer": "Arsenal, 2-1."}],
                },
            )

    args, kwargs = mock_rewrite.call_args
    history_arg = args[0] if args else kwargs["history"]
    query_arg = args[1] if len(args) > 1 else kwargs["query"]
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
                "agentic_rag.api.routers.query.answer_with_cache", return_value="the answer"
            ) as mock_answer,
        ):
            client.post(
                "/query",
                json={"query": "and the second leg?", "user_tier": "tier-2", "history": []},
            )

    args, kwargs = mock_answer.call_args
    query_arg = args[0] if args else kwargs["query"]
    tier_arg = args[1] if len(args) > 1 else kwargs["user_tier"]
    assert query_arg == "who won the second leg of the Arsenal tie?"
    assert tier_arg == "tier-2"


def test_query_rejects_an_empty_query(tmp_path):
    with _client(tmp_path) as client:
        response = client.post("/query", json={"query": "", "user_tier": "tier-1", "history": []})

    assert response.status_code == 422


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
            patch("agentic_rag.api.routers.query.answer_with_cache", return_value="the answer"),
        ):
            response = client.post("/query", json={"query": "who won?", "user_tier": "tier-1"})

    assert response.status_code == 200
    args, kwargs = mock_rewrite.call_args
    history_arg = args[0] if args else kwargs["history"]
    assert history_arg == []
