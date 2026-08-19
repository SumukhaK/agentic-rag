# `embedding/ollama_client.py`

**Purpose:** This file is the low-level piece of code responsible for actually talking to Ollama (a locally-running server that hosts language models, including embedding models) over the network to turn text into "dense embeddings" — long lists of numbers that represent the meaning of a piece of text, used elsewhere in the system to compare texts for similarity. It hides the details of the HTTP request and response format behind one simple function, and translates the various ways that request can fail (server unreachable, bad response, unparseable data) into a single, well-defined error type that the rest of the codebase can catch and handle consistently.

## Line-by-line walkthrough

### Line 1 — Import
```python
import requests
```
- Imports the `requests` library, a widely-used Python package for making HTTP requests. This file uses it to send the actual network call to Ollama's API.

### Lines 4-7 — Custom error type
```python
class EmbeddingError(Exception):
    """Raised whenever embedding fails: Ollama is unreachable, returns a
    non-2xx response, or returns a 200 with a body that isn't a usable
    embeddings payload."""
```
- `class EmbeddingError(Exception):` — defines a custom exception type, built on Python's base `Exception` class. Instead of letting callers deal with several different low-level error types from the `requests` library (connection errors, timeouts, JSON parsing errors, etc.), this file catches all of those internally and re-raises them as this one consistent `EmbeddingError`, so any code calling `embed_texts` only needs to handle one kind of failure.
- The docstring lists the three broad situations that map to this error: the network request never reaching Ollama at all, Ollama responding with an HTTP error status (a "non-2xx response" — HTTP status codes starting with 2, like 200, mean success, so "non-2xx" means something went wrong), or Ollama responding successfully (status 200) but with a body that doesn't actually contain usable embeddings.

### Lines 10-14 — `embed_texts` signature and docstring
```python
def embed_texts(
    texts: list[str], model: str, base_url: str, timeout: int
) -> list[list[float]]:
    """Embed one or more texts in a single call via Ollama's /api/embed
    endpoint, returning one vector per input text, in order."""
```
- Defines the main function of this file. It takes a list of strings to embed (`texts`), the name of the embedding model to use (`model`), the base URL of the Ollama server (`base_url`, e.g. `http://localhost:11434`), and a `timeout` in seconds (how long to wait for a response before giving up). It returns a list of embeddings — one `list[float]` (a vector of numbers) per input text, in the same order they were given.
- The docstring notes that all texts are sent in a single network call to Ollama's `/api/embed` endpoint (rather than one call per text), which is more efficient.

### Lines 15-22 — Making the request and returning the result
```python
    try:
        response = requests.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": texts},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["embeddings"]
```
- `try:` — begins a block that watches for errors during the network call, so they can be translated into `EmbeddingError` rather than leaking out as raw `requests` exceptions.
- `response = requests.post(...)` — sends an HTTP POST request to Ollama's embedding endpoint, built by joining `base_url` with `/api/embed`.
- `json={"model": model, "input": texts}` — the request body: tells Ollama which model to use and gives it the list of texts to embed, sent as JSON (a common structured text format for APIs) — `requests` automatically serializes this dictionary into the request body.
- `timeout=timeout` — caps how long the code will wait for Ollama to respond before giving up, preventing the program from hanging indefinitely if Ollama is slow or stuck.
- `response.raise_for_status()` — checks the HTTP status code of the response, and raises a `requests.HTTPError` (a subclass of `RequestException`) if it indicates failure (any non-2xx code), which is caught further down.
- `return response.json()["embeddings"]` — parses the response body as JSON and pulls out the `"embeddings"` field, which Ollama's API returns as a list of vectors, one per input text — this becomes the function's return value on success.

### Lines 23-29 — Handling a malformed but received response
```python
    except (ValueError, KeyError) as exc:
        # requests.exceptions.JSONDecodeError (raised by response.json() on
        # a malformed body) is BOTH a ValueError and a RequestException, so
        # this branch must be checked first - Ollama did respond, it just
        # sent something unparseable; that's not the same failure as being
        # unreachable, and the message shouldn't conflate the two.
        raise EmbeddingError(f"unexpected response from Ollama: {exc}") from exc
```
- `except (ValueError, KeyError) as exc:` — catches two specific error types: `ValueError` (which covers `response.json()` failing to parse the body as JSON at all) and `KeyError` (which covers the JSON parsing successfully but not containing an `"embeddings"` key, so the dictionary lookup fails).
- The comment explains a subtle but important ordering reason: `requests`' own JSON-decoding error class is defined to be *both* a `ValueError` and a `requests.RequestException` (the broader network-error family caught in the next block). Because this `except (ValueError, KeyError)` clause appears first in the code, Python will match a JSON decode failure here rather than falling through to the more generic `RequestException` handler below — which matters because the situation is meaningfully different: Ollama *did* respond (so it's not "unreachable"), it just sent back something the code couldn't understand. Keeping these as separate error messages avoids describing an unreachable server and a badly-formed response as the same problem.
- `raise EmbeddingError(f"unexpected response from Ollama: {exc}") from exc` — raises the custom error type with a message describing this specific case, while `from exc` preserves the original exception as the documented "cause," so anyone debugging later can still see the underlying `ValueError`/`KeyError` that triggered it.

### Lines 30-31 — Handling an unreachable server or other request-level failure
```python
    except requests.RequestException as exc:
        raise EmbeddingError(f"failed to reach Ollama: {exc}") from exc
```
- `except requests.RequestException as exc:` — catches the broader family of errors `requests` can raise for network-level problems (e.g. connection refused, DNS failure, timeout expiring, or the `raise_for_status()` call above signaling an HTTP error status). Because the more specific `ValueError`/`KeyError` case was already handled above, this block only catches problems that mean the request itself didn't complete successfully — genuinely "couldn't reach or get a good response from Ollama" situations.
- `raise EmbeddingError(f"failed to reach Ollama: {exc}") from exc` — again wraps the failure in the shared `EmbeddingError` type with a message that specifically says the server couldn't be reached, keeping this distinct from the "got a response but couldn't use it" message above, and again preserving the original exception via `from exc` for debugging.
