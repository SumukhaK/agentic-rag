from unittest.mock import Mock, patch

import pytest
import requests

from agentic_rag.embedding.ollama_client import EmbeddingError, embed_text


@patch("agentic_rag.embedding.ollama_client.requests.post")
def test_embed_text_returns_the_embedding_vector(mock_post):
    mock_post.return_value = Mock(
        status_code=200,
        json=lambda: {"embedding": [0.1, 0.2, 0.3]},
        raise_for_status=lambda: None,
    )

    result = embed_text(
        "Arsenal drew 1-1.",
        model="nomic-embed-text",
        base_url="http://localhost:11434",
    )

    assert result == [0.1, 0.2, 0.3]


@patch("agentic_rag.embedding.ollama_client.requests.post")
def test_embed_text_calls_the_ollama_embeddings_endpoint_with_model_and_prompt(
    mock_post,
):
    mock_post.return_value = Mock(
        status_code=200,
        json=lambda: {"embedding": [0.1]},
        raise_for_status=lambda: None,
    )

    embed_text(
        "Arsenal drew 1-1.", model="nomic-embed-text", base_url="http://localhost:11434"
    )

    mock_post.assert_called_once_with(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": "Arsenal drew 1-1."},
        timeout=30,
    )


@patch("agentic_rag.embedding.ollama_client.requests.post")
def test_embed_text_raises_embedding_error_when_ollama_is_unreachable(mock_post):
    mock_post.side_effect = requests.ConnectionError("connection refused")

    with pytest.raises(EmbeddingError):
        embed_text(
            "Arsenal drew 1-1.",
            model="nomic-embed-text",
            base_url="http://localhost:11434",
        )


@patch("agentic_rag.embedding.ollama_client.requests.post")
def test_embed_text_raises_embedding_error_on_a_non_200_response(mock_post):
    response = Mock(status_code=404)
    response.raise_for_status.side_effect = requests.HTTPError("not found")
    mock_post.return_value = response

    with pytest.raises(EmbeddingError):
        embed_text(
            "Arsenal drew 1-1.",
            model="nomic-embed-text",
            base_url="http://localhost:11434",
        )
