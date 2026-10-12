import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import app.main as app_module
from app.config import Settings
from app.dependencies import get_rerank_gateway
from app.reranking.factory import create_rerank_gateway
from app.reranking.gateway import RerankModelNotFoundError, RerankUpstreamError
from app.reranking.schemas import RerankRequest, RerankResponse


def test_jina_reranker_returns_ranked_documents():
    async def run_test():
        def respond(request: httpx.Request) -> httpx.Response:
            assert request.url == "https://api.jina.ai/v1/rerank"
            assert request.headers["Authorization"] == "Bearer jina-test"
            assert json.loads(request.read()) == {
                "model": "jina-reranker-v3",
                "query": "What is RAG?",
                "documents": ["irrelevant", "retrieval augmented generation"],
                "top_n": 1,
            }
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 1, "relevance_score": 0.98},
                    ]
                },
            )

        settings = Settings(_env_file=None, jina_api_key="jina-test")
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await create_rerank_gateway(settings, client).rerank(
                RerankRequest(
                    query="What is RAG?",
                    documents=["irrelevant", "retrieval augmented generation"],
                    top_n=1,
                ),
                request_id="rerank-test",
            )

        assert result.provider == "Jina AI"
        assert result.model == "jina-reranker-v3"
        assert result.request_id == "rerank-test"
        assert [item.index for item in result.results] == [1]
        assert result.results[0].document == {"text": "retrieval augmented generation"}
        assert result.results[0].relevance_score == 0.98

    asyncio.run(run_test())


def test_reranking_requires_jina_key_and_rejects_unknown_model():
    async def run_test():
        settings = Settings(_env_file=None, jina_api_key=None)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
            gateway = create_rerank_gateway(settings, client)
            with pytest.raises(RerankUpstreamError) as missing_key:
                await gateway.rerank(
                    RerankRequest(query="query", documents=["document"]),
                    request_id="missing-key-test",
                )
            assert missing_key.value.status_code == 503
            assert missing_key.value.code == "reranker_not_configured"

            with pytest.raises(RerankModelNotFoundError):
                await gateway.rerank(
                    RerankRequest(
                        model="unknown-model",
                        query="query",
                        documents=["document"],
                    ),
                    request_id="unknown-model-test",
                )

    asyncio.run(run_test())


def test_invalid_upstream_rerank_indices_are_rejected():
    async def run_test():
        settings = Settings(_env_file=None, jina_api_key="jina-test")
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={"results": [{"index": 4, "relevance_score": 0.9}]},
                )
            )
        ) as client:
            gateway = create_rerank_gateway(settings, client)
            with pytest.raises(RerankUpstreamError) as error:
                await gateway.rerank(
                    RerankRequest(query="query", documents=["document"]),
                    request_id="invalid-result-test",
                )

        assert error.value.status_code == 502
        assert error.value.code == "invalid_reranker_response"

    asyncio.run(run_test())


def test_rerank_endpoint_uses_gateway_auth(monkeypatch):
    class StubRerankGateway:
        async def rerank(self, request, request_id):
            return RerankResponse(
                model="jina-reranker-v3",
                results=[
                    {
                        "index": 0,
                        "document": {"text": "document"},
                        "relevance_score": 0.9,
                    }
                ],
                request_id=request_id,
                provider="Jina AI",
                provider_model="jina-reranker-v3",
                provider_endpoint="https://api.jina.ai/v1/rerank",
                latency_ms=1,
            )

    monkeypatch.setattr(app_module, "settings", Settings(_env_file=None, api_key="gateway-test"))
    app_module.app.dependency_overrides[get_rerank_gateway] = lambda: StubRerankGateway()
    try:
        with TestClient(app_module.app) as client:
            unauthorized = client.post(
                "/v1/rerank",
                json={"query": "query", "documents": ["document"]},
            )
            assert unauthorized.status_code == 401

            response = client.post(
                "/v1/rerank",
                json={"query": "query", "documents": ["document"]},
                headers={"Authorization": "Bearer gateway-test"},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["provider"] == "Jina AI"
        assert payload["results"][0]["index"] == 0
        assert payload["results"][0]["document"]["text"] == "document"
    finally:
        app_module.app.dependency_overrides.pop(get_rerank_gateway, None)
