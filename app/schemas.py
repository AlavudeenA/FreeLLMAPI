from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProviderId(str, Enum):
    NVIDIA = "nvidia"
    GROQ = "groq"
    INTERN_AI = "intern_ai"
    COHERE = "cohere"
    CLOUDFLARE_WORKERS_AI = "cloudflare_workers_ai"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderId | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    user: str | None = None
    stream: bool = False

    @field_validator("stream")
    @classmethod
    def streaming_is_not_supported(cls, value: bool) -> bool:
        if value:
            raise ValueError("Streaming is not supported by this endpoint")
        return value


class AttemptInfo(BaseModel):
    provider: str
    model: str
    endpoint: str
    outcome: Literal["success", "rate_limited", "unavailable", "error", "skipped"]
    status_code: int | None = None
    latency_ms: int
    error_code: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created_at: datetime
    request_id: str
    provider: str
    model: str
    provider_model: str
    provider_endpoint: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any] | None = None
    fallback_used: bool
    attempts: list[AttemptInfo]
    latency_ms: int


class ErrorDetails(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorDetails
    attempts: list[AttemptInfo]