from unittest.mock import Mock, patch

import pytest
import requests

from agentic_rag.embedding.ollama_client import (
    EmbeddingError,
    embed_text,
    embed_texts,
)


def _mock_response(status_code=200, body=None, json_side_effect=None):
    response = Mock(status_code=status_code)
    if json_side_effect is not None:
        response.json.side_effect = json_side_effect
    else:
        response.json.return_value = body
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(
            f"{status_code} error"
        )
    else:
        response.raise_for_status.return_value = None
    return response


@patch("agentic_rag.embedding.ollama_client.requests.post")
def test_embed_texts_returns_an_embedding_per_input(mock_post):
    mock_post.return_value = _mock_response(
        body={"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
    )

    result = embed_texts(
        ["Arsenal drew 1-1.", "Chelsea won 3-0."],
        model="nomic-embed-text",
        base_url="http://localhost:11434",
    )

    assert result == [[0.1, 0.2], [0.3, 0.4]]


@patch("agentic_rag.embedding.ollama_client.requests.post")
def test_embed_texts_calls_the_ollama_embed_endpoint_with_model_and_input_list(
    mock_post,
):
    mock_post.return_value = _mock_response(body={"embeddings": [[0.1]]})

    embed_texts(
        ["Arsenal drew 1-1."],
        model="nomic-embed-text",
        base_url="http://localhost:11434",
        timeout=45,
    )

    mock_post.assert_called_once_with(
        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text", "input": ["Arsenal drew 1-1."]},
        timeout=45,
    )


@patch("agentic_rag.embedding.ollama_client.requests.post")
def test_embed_texts_raises_embedding_error_when_ollama_is_unreachable(mock_post):
    mock_post.side_effect = requests.ConnectionError("connection refused")

    with pytest.raises(EmbeddingError):
        embed_texts(
            ["Arsenal drew 1-1."],
            model="nomic-embed-text",
            base_url="http://localhost:11434",
        )


@patch("agentic_rag.embedding.ollama_client.requests.post")
def test_embed_texts_raises_embedding_error_on_a_non_200_response(mock_post):
    mock_post.return_value = _mock_response(status_code=404)

    with pytest.raises(EmbeddingError):
        embed_texts(
            ["Arsenal drew 1-1."],
            model="nomic-embed-text",
            base_url="http://localhost:11434",
        )


@patch("agentic_rag.embedding.ollama_client.requests.post")
def test_embed_texts_raises_embedding_error_when_embeddings_key_is_missing(mock_post):
    # A real Ollama failure mode: a 200 response carrying an error payload
    # instead of embeddings, e.g. the model wasn't pulled.
    mock_post.return_value = _mock_response(body={"error": "model not found"})

    with pytest.raises(EmbeddingError):
        embed_texts(
            ["Arsenal drew 1-1."],
            model="nomic-embed-text",
            base_url="http://localhost:11434",
        )


@patch("agentic_rag.embedding.ollama_client.requests.post")
def test_embed_texts_raises_embedding_error_on_a_non_json_body(mock_post):
    mock_post.return_value = _mock_response(
        json_side_effect=requests.exceptions.JSONDecodeError("bad json", "", 0)
    )

    with pytest.raises(EmbeddingError):
        embed_texts(
            ["Arsenal drew 1-1."],
            model="nomic-embed-text",
            base_url="http://localhost:11434",
        )


@patch("agentic_rag.embedding.ollama_client.requests.post")
def test_embed_texts_reports_a_non_json_body_as_an_unexpected_response_not_unreachable(
    mock_post,
):
    # requests.exceptions.JSONDecodeError is BOTH a RequestException and a
    # ValueError - if the except clauses are ordered wrong, this gets
    # miscategorized as "failed to reach Ollama" when Ollama actually
    # responded, just with a body that couldn't be parsed.
    mock_post.return_value = _mock_response(
        json_side_effect=requests.exceptions.JSONDecodeError("bad json", "", 0)
    )

    with pytest.raises(EmbeddingError, match="unexpected response"):
        embed_texts(
            ["Arsenal drew 1-1."],
            model="nomic-embed-text",
            base_url="http://localhost:11434",
        )


@patch("agentic_rag.embedding.ollama_client.requests.post")
def test_embed_text_returns_a_single_embedding_vector(mock_post):
    mock_post.return_value = _mock_response(body={"embeddings": [[0.1, 0.2, 0.3]]})

    result = embed_text(
        "Arsenal drew 1-1.", model="nomic-embed-text", base_url="http://localhost:11434"
    )

    assert result == [0.1, 0.2, 0.3]
