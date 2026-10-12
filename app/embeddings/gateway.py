import math
import time

import httpx

from app.embeddings.providers.base import EmbeddingRoute
from app.embeddings.schemas import (
    EmbeddingItem,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from app.schemas import AttemptInfo


class EmbeddingUpstreamExhaustedError(Exception):
    def __init__(self, attempts: list[AttemptInfo]):
        self.attempts = attempts
        self.status_code = (
            503
            if all(attempt.outcome in {"skipped", "rate_limited", "unavailable"} for attempt in attempts)
            else 502
        )
        self.code = "embedding_providers_unavailable" if self.status_code == 503 else "embedding_upstream_failure"
        super().__init__("All configured embedding providers failed")


class EmbeddingModelNotFoundError(Exception):
    pass


class EmbeddingGateway:
    def __init__(self, routes: tuple[EmbeddingRoute, ...]):
        self._routes = routes

    async def embed(self, request: EmbeddingRequest, request_id: str) -> EmbeddingResponse:
        started_at = time.perf_counter()
        attempts: list[AttemptInfo] = []
        inputs = [request.input] if isinstance(request.input, str) else request.input
        routes = self._select_routes(request.model)
        if request.model is not None and not routes:
            raise EmbeddingModelNotFoundError(f"Unknown embedding model: {request.model}")

        for route in routes:
            provider = route.provider
            if not provider.is_configured:
                attempts.append(self._attempt(route, "skipped", 0, "provider_not_configured"))
                continue

            attempt_started = time.perf_counter()
            try:
                response = await provider.embed(inputs, route.model)
            except httpx.RequestError:
                attempts.append(
                    self._attempt(
                        route,
                        "unavailable",
                        self._elapsed_ms(attempt_started),
                        "upstream_connection_error",
                    )
                )
                continue

            elapsed = self._elapsed_ms(attempt_started)
            if not response.is_success:
                status = response.status_code
                outcome = "rate_limited" if status == 429 else "unavailable" if status >= 500 or status in {408, 425} else "error"
                attempts.append(
                    self._attempt(
                        route,
                        outcome,
                        elapsed,
                        f"upstream_http_{status}",
                        status,
                    )
                )
                continue

            try:
                data, usage = self._parse_response(response, len(inputs))
            except (ValueError, TypeError, OverflowError):
                attempts.append(
                    self._attempt(route, "unavailable", elapsed, "invalid_upstream_response")
                )
                continue

            attempts.append(self._attempt(route, "success", elapsed, status_code=response.status_code))
            return EmbeddingResponse(
                data=data,
                model=route.model,
                usage=usage,
                request_id=request_id,
                provider=provider.name,
                provider_model=route.model,
                provider_endpoint=provider.endpoint,
                fallback_used=sum(attempt.outcome != "skipped" for attempt in attempts) > 1,
                attempts=attempts,
                latency_ms=self._elapsed_ms(started_at),
            )

        raise EmbeddingUpstreamExhaustedError(attempts)

    def _select_routes(self, model: str | None) -> tuple[EmbeddingRoute, ...]:
        if model is None:
            return self._routes
        normalized = model.strip().lower()
        return tuple(route for route in self._routes if route.model.lower() == normalized)

    @staticmethod
    def _parse_response(
        response: httpx.Response,
        expected_count: int,
    ) -> tuple[list[EmbeddingItem], EmbeddingUsage]:
        result = response.json()
        raw_items = result.get("data") if isinstance(result, dict) else None
        if not isinstance(raw_items, list) or len(raw_items) != expected_count:
            raise ValueError("Invalid embedding response data")

        items: list[EmbeddingItem] = []
        dimensions: int | None = None
        for position, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, dict):
                raise ValueError("Invalid embedding item")
            embedding = raw_item.get("embedding")
            index = raw_item.get("index", position)
            if (
                not isinstance(embedding, list)
                or not embedding
                or not isinstance(index, int)
                or isinstance(index, bool)
                or index != position
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in embedding
                )
            ):
                raise ValueError("Invalid embedding vector")
            if dimensions is None:
                dimensions = len(embedding)
            elif len(embedding) != dimensions:
                raise ValueError("Embedding vector dimensions do not match")
            items.append(EmbeddingItem(index=index, embedding=embedding))

        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
        total_tokens = usage.get("total_tokens", prompt_tokens) if isinstance(usage, dict) else 0
        if (
            not isinstance(prompt_tokens, int)
            or isinstance(prompt_tokens, bool)
            or prompt_tokens < 0
            or not isinstance(total_tokens, int)
            or isinstance(total_tokens, bool)
            or total_tokens < 0
        ):
            raise ValueError("Invalid embedding usage")

        return items, EmbeddingUsage(prompt_tokens=prompt_tokens, total_tokens=total_tokens)

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return round((time.perf_counter() - started_at) * 1000)

    @classmethod
    def _attempt(
        cls,
        route: EmbeddingRoute,
        outcome: str,
        latency_ms: int,
        error_code: str | None = None,
        status_code: int | None = None,
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
