import requests


class GenerationError(Exception):
    """Raised whenever generation fails: Ollama is unreachable, returns a
    non-2xx response, or returns a 200 with a body that isn't a usable
    generation payload."""


def generate(prompt: str, model: str, base_url: str, timeout: int) -> str:
    """Generate text from `prompt` via Ollama's /api/generate endpoint
    (non-streaming - the full response in one call)."""
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["response"]
    except (ValueError, KeyError) as exc:
        # requests.exceptions.JSONDecodeError (raised by response.json() on
        # a malformed body) is BOTH a ValueError and a RequestException, so
        # this branch must be checked first - Ollama did respond, it just
        # sent something unparseable; that's not the same failure as being
        # unreachable, and the message shouldn't conflate the two.
        raise GenerationError(f"unexpected response from Ollama: {exc}") from exc
    except requests.RequestException as exc:
        raise GenerationError(f"failed to reach Ollama: {exc}") from exc
