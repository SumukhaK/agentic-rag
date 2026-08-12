import requests


class EmbeddingError(Exception):
    """Raised whenever embedding fails: Ollama is unreachable, returns a
    non-2xx response, or returns a 200 with a body that isn't a usable
    embeddings payload."""


def embed_texts(
    texts: list[str], model: str, base_url: str, timeout: int = 30
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
    except requests.RequestException as exc:
        raise EmbeddingError(f"failed to reach Ollama: {exc}") from exc
    except (ValueError, KeyError) as exc:
        raise EmbeddingError(f"unexpected response from Ollama: {exc}") from exc


def embed_text(text: str, model: str, base_url: str, timeout: int = 30) -> list[float]:
    """Embed a single text. A thin convenience wrapper over embed_texts()."""
    return embed_texts([text], model, base_url, timeout)[0]
