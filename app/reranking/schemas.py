from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import ErrorDetails


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1)
    top_n: int | None = Field(default=None, gt=0)


class RerankResult(BaseModel):
    index: int
    document: dict[str, str]
    relevance_score: float


class RerankResponse(BaseModel):
    object: Literal["list"] = "list"
    model: str
    results: list[RerankResult]
    request_id: str
    provider: str
    provider_model: str
    provider_endpoint: str
    latency_ms: int


class RerankErrorResponse(BaseModel):
    request_id: str
    error: ErrorDetails
