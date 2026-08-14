from pydantic import BaseModel, Field


class ConversationTurnModel(BaseModel):
    user_query: str
    assistant_answer: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    user_tier: str = Field(min_length=1)
    history: list[ConversationTurnModel] = []


class QueryResponse(BaseModel):
    answer: str
