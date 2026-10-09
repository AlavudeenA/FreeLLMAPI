import time
from datetime import datetime, timezone
from uuid import uuid4

import httpx

from app.providers.base import ProviderModel
from app.schemas import AttemptInfo, ChatCompletionRequest, ChatCompletionResponse


class UpstreamExhaustedError(Exception):
    def __init__(self, attempts: list[AttemptInfo]):
        self.attempts = attempts
        self.status_code = (
            503
            if all(attempt.outcome in {"skipped", "rate_limited", "unavailable"} for attempt in attempts)
            else 502
        )
        self.code = "providers_unavailable" if self.status_code == 503 else "upstream_failure"
        super().__init__("All configured model providers failed")


class LLMGateway:
    def __init__(self, routes: tuple[ProviderModel, ...]):
        self._routes = routes

    async def complete(self, request: ChatCompletionRequest, request_id: str) -> ChatCompletionResponse:
        started_at = time.perf_counter()
        attempts: list[AttemptInfo] = []
        payload = request.model_dump(exclude_none=True, exclude={"provider", "stream"})
        routes = self._routes
        if request.provider is not None:
            routes = tuple(route for route in routes if route.provider.provider_id == request.provider.value)

        for route in routes:
            provider = route.provider
            if not provider.is_configured:
                attempts.append(self._attempt(route, "skipped", 0, error_code="provider_not_configured"))
                continue

            attempt_started = time.perf_counter()
            try:
                response = await provider.complete({**payload, "model": route.model})
            except httpx.RequestError:
                attempts.append(
                    self._attempt(
                        route,
                        "unavailable",
                        self._elapsed_ms(attempt_started),
                        error_code="upstream_connection_error",
                    )
                )
                continue

            elapsed = self._elapsed_ms(attempt_started)
            if not response.is_success:
                status = response.status_code
                outcome = "rate_limited" if status == 429 else "unavailable" if status >= 500 or status in {408, 425} else "error"
                attempts.append(
                    self._attempt(route, outcome, elapsed, status_code=status, error_code=f"upstream_http_{status}")
                )
                continue

            try:
                data = response.json()
                choices = data.get("choices") if isinstance(data, dict) else None
                if not isinstance(choices, list) or not all(isinstance(choice, dict) for choice in choices):
                    raise ValueError("Missing choices in upstream response")
                if not choices or not self._has_assistant_output(choices[0]):
                    raise ValueError("Missing assistant content in upstream response")
            except (ValueError, TypeError):
                attempts.append(
                    self._attempt(route, "unavailable", elapsed, error_code="invalid_upstream_response")
                )
                continue

            attempts.append(self._attempt(route, "success", elapsed, status_code=response.status_code))
            reported_model = data.get("model")
            return ChatCompletionResponse(
                id=str(data.get("id") or uuid4()),
                created_at=datetime.now(timezone.utc),
                request_id=request_id,
                provider=provider.name,
                model=reported_model if isinstance(reported_model, str) else route.model,
                provider_model=route.model,
                provider_endpoint=provider.endpoint,
                choices=choices,
                usage=data.get("usage") if isinstance(data.get("usage"), dict) else None,
                fallback_used=sum(attempt.outcome != "skipped" for attempt in attempts) > 1,
                attempts=attempts,
                latency_ms=self._elapsed_ms(started_at),
            )

        raise UpstreamExhaustedError(attempts)

    @staticmethod
    def _has_assistant_output(choice: dict) -> bool:
        message = choice.get("message")
        if not isinstance(message, dict):
            return False

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return True
        if isinstance(content, list) and any(
            isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].strip()
            for block in content
        ):
            return True
        if isinstance(message.get("tool_calls"), list) and message["tool_calls"]:
            return True
        refusal = message.get("refusal")
        return isinstance(refusal, str) and bool(refusal.strip())

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return round((time.perf_counter() - started_at) * 1000)

    @classmethod
    def _attempt(
        cls,
        route: ProviderModel,
        outcome: str,
        latency_ms: int,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> AttemptInfo:
        return AttemptInfo(
            provider=route.provider.name,
            model=route.model,
            endpoint=route.provider.endpoint,
            outcome=outcome,
            status_code=status_code,
            latency_ms=latency_ms,
            error_code=error_code,
        )