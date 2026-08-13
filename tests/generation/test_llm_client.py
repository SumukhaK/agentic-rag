from unittest.mock import Mock, patch

import pytest
import requests

from agentic_rag.generation.llm_client import GenerationError, generate


def _mock_response(status_code=200, body=None, json_side_effect=None):
    response = Mock(status_code=status_code)
    if json_side_effect is not None:
        response.json.side_effect = json_side_effect
    else:
        response.json.return_value = body
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(f"{status_code} error")
    else:
        response.raise_for_status.return_value = None
    return response


@patch("agentic_rag.generation.llm_client.requests.post")
def test_generate_returns_the_response_text(mock_post):
    mock_post.return_value = _mock_response(body={"response": "Hello there!"})

    result = generate(
        "Say hello.", model="mistral", base_url="http://localhost:11434", timeout=30
    )

    assert result == "Hello there!"


@patch("agentic_rag.generation.llm_client.requests.post")
def test_generate_calls_the_ollama_generate_endpoint_with_model_and_prompt(mock_post):
    mock_post.return_value = _mock_response(body={"response": "ok"})

    generate("Say hello.", model="mistral", base_url="http://localhost:11434", timeout=45)

    mock_post.assert_called_once_with(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": "Say hello.", "stream": False},
        timeout=45,
    )


@patch("agentic_rag.generation.llm_client.requests.post")
def test_generate_raises_generation_error_when_ollama_is_unreachable(mock_post):
    mock_post.side_effect = requests.ConnectionError("connection refused")

    with pytest.raises(GenerationError):
        generate("Say hello.", model="mistral", base_url="http://localhost:11434", timeout=30)


@patch("agentic_rag.generation.llm_client.requests.post")
def test_generate_raises_generation_error_on_a_non_200_response(mock_post):
    mock_post.return_value = _mock_response(status_code=404)

    with pytest.raises(GenerationError):
        generate("Say hello.", model="mistral", base_url="http://localhost:11434", timeout=30)


@patch("agentic_rag.generation.llm_client.requests.post")
def test_generate_raises_generation_error_when_response_key_is_missing(mock_post):
    # A real Ollama failure mode: a 200 with an error payload instead of a
    # generation result, e.g. the model wasn't pulled.
    mock_post.return_value = _mock_response(body={"error": "model not found"})

    with pytest.raises(GenerationError):
        generate("Say hello.", model="mistral", base_url="http://localhost:11434", timeout=30)


@patch("agentic_rag.generation.llm_client.requests.post")
def test_generate_raises_generation_error_on_a_non_json_body(mock_post):
    mock_post.return_value = _mock_response(
        json_side_effect=requests.exceptions.JSONDecodeError("bad json", "", 0)
    )

    with pytest.raises(GenerationError):
        generate("Say hello.", model="mistral", base_url="http://localhost:11434", timeout=30)


@patch("agentic_rag.generation.llm_client.requests.post")
def test_generate_reports_a_non_json_body_as_an_unexpected_response_not_unreachable(
    mock_post,
):
    # requests.exceptions.JSONDecodeError is BOTH a RequestException and a
    # ValueError - if the except clauses are ordered wrong, this gets
    # miscategorized as "failed to reach Ollama" when Ollama actually
    # responded, just with a body that couldn't be parsed. The message
    # should say so, not blame connectivity.
    mock_post.return_value = _mock_response(
        json_side_effect=requests.exceptions.JSONDecodeError("bad json", "", 0)
    )

    with pytest.raises(GenerationError, match="unexpected response"):
        generate("Say hello.", model="mistral", base_url="http://localhost:11434", timeout=30)
