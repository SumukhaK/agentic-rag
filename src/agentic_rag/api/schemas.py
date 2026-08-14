from pydantic import BaseModel, Field, field_validator


class ConversationTurnModel(BaseModel):
    user_query: str
    assistant_answer: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    user_tier: str = Field(min_length=1)
    history: list[ConversationTurnModel] = []

    @field_validator("query")
    @classmethod
    def _reject_whitespace_only_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class QueryResponse(BaseModel):
    answer: str
