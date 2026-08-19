# `generation/llm_client.py`

**Purpose:** This file is the single place in the codebase that actually talks to Ollama (the local large-language-model server) to generate text. Anything in the system that needs an LLM to write an answer, rewrite a search query, or make a judgment call routes through the `generate()` function here. It exists so that the HTTP details of calling Ollama's `/api/generate` endpoint, and the error handling around network failures or malformed responses, live in exactly one function instead of being copy-pasted everywhere an LLM call is needed.

## Line-by-line walkthrough

### Line 1 — Import
```python
import requests
```
- `import requests` — brings in the `requests` library, which is used to make the HTTP POST call to Ollama's local web server. Ollama exposes its functionality over HTTP, so this is the mechanism used to reach it.

### Lines 4-7 — Custom exception type
```python
class GenerationError(Exception):
    """Raised whenever generation fails: Ollama is unreachable, returns a
    non-2xx response, or returns a 200 with a body that isn't a usable
    generation payload."""
```
- `class GenerationError(Exception):` — defines a dedicated exception type for this module. Instead of letting callers deal with raw `requests` exceptions or JSON parsing errors, every possible failure mode (can't reach Ollama, Ollama returned an HTTP error code, or Ollama returned a 200 OK but with a body that doesn't actually contain usable text) gets normalized into this one exception type. That way, code elsewhere in the system that calls `generate()` only needs to catch one thing.
- The docstring documents the three distinct failure scenarios this exception is meant to represent, so a future reader understands the exception isn't just "something went wrong with an HTTP request" but specifically covers unreachability, bad status codes, and unusable response bodies.

### Lines 10-31 — Function signature and docstring
```python
def generate(
    prompt: str, model: str, base_url: str, timeout: int, *, temperature: float
) -> str:
    """Generate text from `prompt` via Ollama's /api/generate endpoint
    (non-streaming - the full response in one call).

    `temperature` is required, not defaulted - every caller must decide
    explicitly whether this call should be deterministic (`0.0`, for
    classification/judgment work where the same input should reliably
    produce the same verdict) or benefits from sampling variance (a
    caller-chosen higher value, for open-ended generation or a
    deliberately-exploratory retry). Leaving it optional previously let
    the same "forgot to pin it" bug get reintroduced independently three
    times (`generate_answer`, `rewrite_query`, `decompose_query`) -
    discovered as a real gap, not a theoretical one: the injection judge's
    own live validation suite caught the identical adversarial prompt
    passing on one run and failing on a re-run with no code change,
    confirming Ollama's default (non-zero) sampling temperature made a
    security-relevant verdict genuinely inconsistent, not just a flaky
    test. A required parameter closes the gap for every future caller,
    not just the ones that exist today.
    """
```
- `def generate(prompt: str, model: str, base_url: str, timeout: int, *, temperature: float) -> str:` — declares the function's public interface. `prompt` is the text sent to the model, `model` is which Ollama model to use (e.g. an installed model name), `base_url` is the address of the Ollama server (so it can point at localhost or elsewhere depending on config), and `timeout` bounds how long to wait for a response before giving up. The `*` before `temperature` makes it a keyword-only argument — callers must write `temperature=...` explicitly rather than passing it positionally, which makes call sites self-documenting and impossible to get wrong by argument order. The function returns a plain `str` (the generated text).
- Making `temperature` required with no default (rather than defaulting to, say, `0.0` or `0.7`) is a deliberate design decision explained in the docstring: temperature controls how random/deterministic the model's output is. Low (`0.0`) makes the model always produce the same answer for the same input — important for tasks like classification or safety judgments where consistency matters. Higher values introduce randomness, useful for creative or exploratory generation. The docstring recounts that this wasn't a hypothetical concern — leaving temperature optional let three different call sites each independently forget to pin it, and a live security test (an "injection judge" checking whether the model correctly flags adversarial prompts) actually caught a verdict flip-flopping between runs with no code change, because Ollama's default temperature isn't zero. Making the parameter mandatory forces every future caller to consciously choose, closing that class of bug permanently rather than patching each occurrence.
- The docstring's first line also clarifies this is a *non-streaming* call — Ollama can stream tokens back incrementally, but this function waits for and returns the entire response in one shot, which is simpler for callers that just want the final text.

### Lines 32-37 — Building the request payload
```python
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
```
- `payload = {...}` — assembles the JSON body that will be sent to Ollama, matching the shape Ollama's `/api/generate` API expects.
- `"model": model` — tells Ollama which model to run the prompt through.
- `"prompt": prompt` — the actual text/question being sent to the model.
- `"stream": False` — explicitly disables streaming responses, matching the "non-streaming" behavior described in the docstring; the whole answer comes back in a single HTTP response rather than a series of chunks.
- `"options": {"temperature": temperature}` — nests the temperature setting inside Ollama's `options` object, which is where Ollama expects generation-tuning parameters like temperature to live.

### Lines 38-45 — Making the request and returning the result
```python
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["response"]
```
- `try:` — opens a block that wraps the network call so different kinds of failures can be caught and translated into `GenerationError` below.
- `response = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)` — sends the HTTP POST request. The URL is built by appending Ollama's generate endpoint path to the configured `base_url`. Passing `json=payload` tells `requests` to serialize the dict to JSON and set the appropriate content-type header automatically. `timeout=timeout` ensures the call doesn't hang forever if Ollama is slow or unresponsive.
- `response.raise_for_status()` — if Ollama responded with an HTTP error status code (4xx or 5xx), this raises a `requests.HTTPError` (a subclass of `requests.RequestException`), which gets caught further down and converted to a `GenerationError`.
- `return response.json()["response"]` — parses the response body as JSON and pulls out the `"response"` key, which is where Ollama puts the generated text. This is the function's normal, successful return path.

### Lines 46-54 — Translating failures into `GenerationError`
```python
    except (ValueError, KeyError) as exc:
        # requests.exceptions.JSONDecodeError (raised by response.json() on
        # a malformed body) is BOTH a ValueError and a RequestException, so
        # this branch must be checked first - Ollama did respond, it just
        # sent something unparseable; that's not the same failure as being
        # unreachable, and the message shouldn't conflate the two.
        raise GenerationError(f"unexpected response from Ollama: {exc}") from exc
    except requests.RequestException as exc:
        raise GenerationError(f"failed to reach Ollama: {exc}") from exc
```
- `except (ValueError, KeyError) as exc:` — catches two distinct problems with a response that Ollama *did* successfully send: `ValueError` covers the case where `response.json()` fails to parse the body as JSON at all (a malformed body), and `KeyError` covers the case where the JSON parsed fine but didn't contain the expected `"response"` field.
- The inline comment explains a subtle ordering requirement: `requests`' own JSON-decoding error class inherits from *both* `ValueError` and `requests.RequestException`. Because Python checks `except` clauses in order and uses the first match, this `except (ValueError, KeyError)` clause must come *before* the broader `except requests.RequestException` clause below it — otherwise a JSON parsing failure would be caught by the generic "failed to reach Ollama" branch and produce a misleading error message, even though Ollama was perfectly reachable and simply sent back something unusable.
- `raise GenerationError(f"unexpected response from Ollama: {exc}") from exc` — wraps the original exception in a `GenerationError` with a message that specifically says the response was unexpected/unparseable (not a connectivity problem). `from exc` preserves the original exception as the visible "cause" in tracebacks, which helps with debugging without hiding the root error.
- `except requests.RequestException as exc:` — catches everything else that `requests` can raise for network-level problems (connection refused, DNS failure, timeout, an HTTP error status from `raise_for_status()`, etc.) — i.e., Ollama genuinely could not be reached or responded with an error status.
- `raise GenerationError(f"failed to reach Ollama: {exc}") from exc` — re-raises as a `GenerationError` with a message that specifically says Ollama couldn't be reached, again preserving the original exception via `from exc` for traceability. Together, the two `except` blocks ensure callers of `generate()` get one exception type (`GenerationError`) but with a message that accurately distinguishes "couldn't reach the server" from "reached it but got garbage back."
