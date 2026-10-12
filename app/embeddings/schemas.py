from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import AttemptInfo, ErrorDetails


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    input: str | list[str] = Field(min_length=1)
    encoding_format: Literal["float"] = "float"
    user: str | None = None


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingItem]
    model: str
    usage: EmbeddingUsage
    request_id: str
    provider: str
    provider_model: str
    provider_endpoint: str
    fallback_used: bool
    attempts: list[AttemptInfo]
    latency_ms: int


class EmbeddingErrorResponse(BaseModel):
    request_id: str
    error: ErrorDetails
    attempts: list[AttemptInfo]
