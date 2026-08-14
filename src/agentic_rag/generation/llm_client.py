import requests


class GenerationError(Exception):
    """Raised whenever generation fails: Ollama is unreachable, returns a
    non-2xx response, or returns a 200 with a body that isn't a usable
    generation payload."""


def generate(
    prompt: str, model: str, base_url: str, timeout: int, temperature: float | None = None
) -> str:
    """Generate text from `prompt` via Ollama's /api/generate endpoint
    (non-streaming - the full response in one call).

    `temperature` is omitted by default, leaving Ollama's own sampling
    default in place - fine for open-ended generation, where some
    response variation is harmless or even desirable. Callers doing
    classification/judgment, where the same input should reliably produce
    the same verdict, should pass a low value (e.g. `0.0`) explicitly -
    discovered as a real gap, not a theoretical one: the injection judge's
    own live validation suite caught the identical adversarial prompt
    passing on one run and failing on a re-run with no code change,
    confirming Ollama's default (non-zero) sampling temperature made a
    security-relevant verdict genuinely inconsistent, not just a flaky
    test.
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    if temperature is not None:
        payload["options"] = {"temperature": temperature}
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
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
