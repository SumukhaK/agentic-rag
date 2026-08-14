import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

from agentic_rag.api.app import create_app
from agentic_rag.config import Settings

_PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _test_settings(tmp_path: Path) -> Settings:
    return Settings(
        watched_folder_path=tmp_path / "corpus",
        qdrant_storage_path=tmp_path / "qdrant",
        _env_file=None,
    )


def _openapi_schema(tmp_path: Path) -> dict:
    app = create_app(_test_settings(tmp_path))
    with TestClient(app) as client:
        return client.get("/openapi.json").json()


def test_openapi_schema_has_a_description(tmp_path):
    schema = _openapi_schema(tmp_path)

    assert schema["info"]["description"]


def test_openapi_schema_version_matches_pyproject(tmp_path):
    # FastAPI defaults info.version to the literal string "0.1.0" when none
    # is given - matching this project's actual version today by pure
    # coincidence. Asserting against pyproject.toml directly, rather than a
    # second hardcoded "0.1.0" in this test, turns a future version bump
    # that forgets to update app.py into a test failure instead of a
    # silently stale OpenAPI schema.
    pyproject = tomllib.loads(_PYPROJECT_PATH.read_text())
    schema = _openapi_schema(tmp_path)

    assert schema["info"]["version"] == pyproject["project"]["version"]


def test_query_422_response_documents_the_unknown_tier_case(tmp_path):
    # /query can return a 422 two different ways with two different bodies:
    # FastAPI's own request-validation failure (an HTTPValidationError -
    # a `detail` array of field errors), or an UnknownAccessTierError
    # raised deliberately for a `user_tier` FastAPI's own validation never
    # sees, which comes back as a plain `{"detail": "<message>"}` string.
    # The default auto-generated schema only documents the first shape -
    # a client trusting it to always be an array would break parsing the
    # second. The description must call out that the shapes differ.
    schema = _openapi_schema(tmp_path)

    description = schema["paths"]["/query"]["post"]["responses"]["422"]["description"]
    assert "user_tier" in description


def test_query_422_response_still_documents_the_validation_error_schema(tmp_path):
    # Supplying a custom `responses={422: {...}}` on the route replaces
    # FastAPI's own auto-added 422 entry rather than merging into it - a
    # naive fix that only adds a `description` would silently drop the
    # `content`/schema reference for the ordinary request-validation-failure
    # case, leaving it undocumented even though it's still a real, common
    # response shape.
    schema = _openapi_schema(tmp_path)

    response_422 = schema["paths"]["/query"]["post"]["responses"]["422"]
    schema_ref = response_422["content"]["application/json"]["schema"]["$ref"]
    assert schema_ref == "#/components/schemas/HTTPValidationError"
    assert "HTTPValidationError" in schema["components"]["schemas"]


def test_health_response_schema_reflects_the_actual_fixed_body(tmp_path):
    # The endpoint only ever returns {"status": "ok"} - the previous
    # untyped `dict[str, str]` return annotation documented it as any
    # object with arbitrary string keys/values, which is broader than
    # what the endpoint can actually produce.
    schema = _openapi_schema(tmp_path)

    response_schema = schema["paths"]["/health"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    component_name = response_schema["$ref"].split("/")[-1]
    health_schema = schema["components"]["schemas"][component_name]

    assert health_schema["required"] == ["status"]
    assert "additionalProperties" not in health_schema
