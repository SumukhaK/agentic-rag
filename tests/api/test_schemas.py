from agentic_rag.api.schemas import (
    CitationModel,
    ConversationTurnModel,
    QueryRequest,
    QueryResponse,
)

_MODELS = (QueryRequest, QueryResponse, CitationModel, ConversationTurnModel)


def _field_descriptions(model) -> dict[str, str | None]:
    return {
        name: prop.get("description")
        for name, prop in model.model_json_schema()["properties"].items()
    }


def test_every_schema_class_has_a_docstring():
    # A bare BaseModel with no docstring shows up in Swagger UI as a
    # schema with no explanation of what it represents or when it's used.
    for model in _MODELS:
        assert model.__doc__


def test_query_request_fields_are_all_documented():
    descriptions = _field_descriptions(QueryRequest)
    assert all(descriptions.values())


def test_query_response_fields_are_all_documented():
    descriptions = _field_descriptions(QueryResponse)
    assert all(descriptions.values())


def test_citation_model_fields_are_all_documented():
    descriptions = _field_descriptions(CitationModel)
    assert all(descriptions.values())


def test_conversation_turn_model_fields_are_all_documented():
    descriptions = _field_descriptions(ConversationTurnModel)
    assert all(descriptions.values())
