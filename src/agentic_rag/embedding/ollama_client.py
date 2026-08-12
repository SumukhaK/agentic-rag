import requests


class EmbeddingError(Exception):
    """Raised when Ollama can't be reached, or refuses to embed the text."""


def embed_text(text: str, model: str, base_url: str) -> list[float]:
    """Embed `text` via Ollama's /api/embeddings endpoint."""
    try:
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise EmbeddingError(f"failed to embed text via Ollama: {exc}") from exc

    return response.json()["embedding"]
