import math
import time

import httpx

from app.reranking.provider import JinaRerankerProvider
from app.reranking.schemas import RerankRequest, RerankResponse, RerankResult


class RerankModelNotFoundError(Exception):
    pass


class RerankUpstreamError(Exception):
    def __init__(self, message: str, status_code: int, code: str):
        self.status_code = status_code
        self.code = code
        super().__init__(message)


class RerankGateway:
    def __init__(self, provider: JinaRerankerProvider, model: str):
        self._provider = provider
        self._model = model

    async def rerank(self, request: RerankRequest, request_id: str) -> RerankResponse:
        if request.model is not None and request.model.strip().lower() != self._model.lower():
            raise RerankModelNotFoundError(f"Unknown reranker model: {request.model}")
        if not self._provider.is_configured:
            raise RerankUpstreamError(
                "Jina reranking is not configured; set JINA_API_KEY",
                503,
                "reranker_not_configured",
            )

        payload: dict[str, object] = {
            "model": self._model,
            "query": request.query,
            "documents": request.documents,
        }
        if request.top_n is not None:
            payload["top_n"] = min(request.top_n, len(request.documents))

        started_at = time.perf_counter()
        try:
            response = await self._provider.rerank(payload)
        except httpx.RequestError as error:
            raise RerankUpstreamError(
                "Could not connect to the Jina reranking service",
                503,
                "reranker_unavailable",
            ) from error

        if not response.is_success:
            status = response.status_code
            unavailable = status == 429 or status >= 500 or status in {408, 425}
            raise RerankUpstreamError(
                "Jina reranking service returned an error",
                503 if unavailable else 502,
                "reranker_unavailable" if unavailable else "reranker_upstream_failure",
            )

        try:
            results = self._parse_results(response, request.documents)
        except (ValueError, TypeError, OverflowError) as error:
            raise RerankUpstreamError(
                "Jina reranking service returned an invalid response",
                502,
                "invalid_reranker_response",
            ) from error

        return RerankResponse(
            model=self._model,
            results=results,
            request_id=request_id,
            provider=self._provider.name,
            provider_model=self._model,
            provider_endpoint=self._provider.endpoint,
            latency_ms=round((time.perf_counter() - started_at) * 1000),
        )

    @staticmethod
    def _parse_results(response: httpx.Response, documents: list[str]) -> list[RerankResult]:
        data = response.json()
        raw_results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(raw_results, list) or not raw_results or len(raw_results) > len(documents):
            raise ValueError("Invalid reranking results")

        results: list[RerankResult] = []
        seen_indices: set[int] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                raise ValueError("Invalid reranking result")
            index = item.get("index")
            score = item.get("relevance_score")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= len(documents)
                or index in seen_indices
                or not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
            ):
                raise ValueError("Invalid reranking result values")
            seen_indices.add(index)
            results.append(
                RerankResult(
                    index=index,
                    document={"text": documents[index]},
                    relevance_score=score,
                )
            )
        return results
