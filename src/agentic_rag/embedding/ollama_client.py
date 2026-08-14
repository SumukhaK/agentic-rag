import requests


class EmbeddingError(Exception):
    """Raised whenever embedding fails: Ollama is unreachable, returns a
    non-2xx response, or returns a 200 with a body that isn't a usable
    embeddings payload."""


def embed_texts(
    texts: list[str], model: str, base_url: str, timeout: int
) -> list[list[float]]:
    """Embed one or more texts in a single call via Ollama's /api/embed
    endpoint, returning one vector per input text, in order."""
    try:
        response = requests.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": texts},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["embeddings"]
    except (ValueError, KeyError) as exc:
        # requests.exceptions.JSONDecodeError (raised by response.json() on
        # a malformed body) is BOTH a ValueError and a RequestException, so
        # this branch must be checked first - Ollama did respond, it just
        # sent something unparseable; that's not the same failure as being
        # unreachable, and the message shouldn't conflate the two.
        raise EmbeddingError(f"unexpected response from Ollama: {exc}") from exc
    except requests.RequestException as exc:
        raise EmbeddingError(f"failed to reach Ollama: {exc}") from exc
