import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import app.main as app_module
from app.config import Settings
from app.dependencies import get_embedding_gateway
from app.embeddings.factory import create_embedding_gateway
from app.embeddings.gateway import EmbeddingModelNotFoundError, EmbeddingUpstreamExhaustedError
from app.embeddings.schemas import EmbeddingRequest, EmbeddingResponse, EmbeddingUsage


def isolated_settings(**overrides) -> Settings:
    provider_settings = {
        "nvidia_api_key": None,
        "jina_api_key": None,
    }
    return Settings(
        _env_file=None,
        **(provider_settings | overrides),
    )


def test_rate_limited_nvidia_falls_back_to_jina():
    async def run_test():
        def respond(request: httpx.Request) -> httpx.Response:
            if request.url == "https://integrate.api.nvidia.com/v1/embeddings":
                assert request.headers["Authorization"] == "Bearer nvidia-test"
                assert json.loads(request.read())["model"] == "nvidia/nemotron-3-embed-1b"
                return httpx.Response(429, json={"error": "rate limited"})

            assert request.url == "https://api.jina.ai/v1/embeddings"
            assert request.headers["Authorization"] == "Bearer jina-test"
            payload = json.loads(request.read())
            assert payload["model"] == "jina-embeddings-v5-text-small"
            assert payload["input"] == ["first", "second"]
            assert payload["embedding_type"] == "float"
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                        {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                    ],
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                },
            )

        settings = isolated_settings(nvidia_api_key="nvidia-test", jina_api_key="jina-test")
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await create_embedding_gateway(settings, client).embed(
                EmbeddingRequest(input=["first", "second"]),
                request_id="embedding-test",
            )

        assert result.provider == "Jina AI"
        assert result.model == "jina-embeddings-v5-text-small"
        assert [item.index for item in result.data] == [0, 1]
        assert result.usage.prompt_tokens == 3
        assert result.fallback_used is True
        assert [attempt.outcome for attempt in result.attempts] == ["rate_limited", "success"]

    asyncio.run(run_test())


def test_failure_falls_through_all_five_embedding_models():
    async def run_test():
        attempted_models = []

        def respond(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.read())
            attempted_models.append(body.get("model"))
            return httpx.Response(429, json={"error": "provider unavailable"})

        settings = isolated_settings(nvidia_api_key="nvidia-test", jina_api_key="jina-test")
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            gateway = create_embedding_gateway(settings, client)
            with pytest.raises(EmbeddingUpstreamExhaustedError) as error:
                await gateway.embed(EmbeddingRequest(input="query"), request_id="all-models-test")

        assert attempted_models == [
            "nvidia/nemotron-3-embed-1b",
            "jina-embeddings-v5-text-small",
            "qwen3-embedding:0.6b",
            "bge-m3",
            "nomic-embed-text",
        ]
        assert [attempt.outcome for attempt in error.value.attempts] == ["rate_limited"] * 5

    asyncio.run(run_test())


def test_ollama_normalizes_local_qwen_response():
    async def run_test():
        def respond(request: httpx.Request) -> httpx.Response:
            assert request.url == "http://127.0.0.1:11434/api/embed"
            payload = json.loads(request.read())
            assert payload == {"model": "qwen3-embedding:0.6b", "input": ["query"]}
            return httpx.Response(200, json={"embeddings": [[0.5, 0.6]]})

        settings = isolated_settings()
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await create_embedding_gateway(settings, client).embed(
                EmbeddingRequest(input="query", model="Qwen/Qwen3-Embedding-0.6B"),
                request_id="local-embedding-test",
            )

        assert result.provider == "Ollama"
        assert result.model == "Qwen/Qwen3-Embedding-0.6B"
        assert result.data[0].embedding == [0.5, 0.6]
        assert result.fallback_used is False
        assert [attempt.outcome for attempt in result.attempts] == ["success"]

    asyncio.run(run_test())


def test_explicit_model_does_not_fall_back():
    async def run_test():
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(503, json={"error": "unavailable"})

        settings = isolated_settings(nvidia_api_key="nvidia-test", jina_api_key="jina-test")
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            gateway = create_embedding_gateway(settings, client)
            with pytest.raises(EmbeddingUpstreamExhaustedError) as error:
                await gateway.embed(
                    EmbeddingRequest(input="query", model="nvidia/nemotron-3-embed-1b"),
                    request_id="pinned-embedding-test",
                )

        assert len(requests) == 1
        assert error.value.status_code == 503
        assert [attempt.outcome for attempt in error.value.attempts] == ["unavailable"]

    asyncio.run(run_test())


def test_unknown_model_is_rejected_instead_of_silently_falling_back():
    async def run_test():
        settings = isolated_settings()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
            gateway = create_embedding_gateway(settings, client)
            with pytest.raises(EmbeddingModelNotFoundError):
                await gateway.embed(
                    EmbeddingRequest(input="query", model="not-a-configured-model"),
                    request_id="unknown-embedding-test",
                )

    asyncio.run(run_test())


def test_embeddings_endpoint_uses_gateway_auth_and_openai_response_shape(monkeypatch):
    class StubEmbeddingGateway:
        async def embed(self, request, request_id):
            return EmbeddingResponse(
                data=[{"index": 0, "embedding": [0.1, 0.2]}],
                model="jina-embeddings-v5-text-small",
                usage=EmbeddingUsage(prompt_tokens=2, total_tokens=2),
                request_id=request_id,
                provider="Jina AI",
                provider_model="jina-embeddings-v5-text-small",
                provider_endpoint="https://api.jina.ai/v1/embeddings",
                fallback_used=False,
                attempts=[],
                latency_ms=1,
            )

    monkeypatch.setattr(app_module, "settings", Settings(_env_file=None, api_key="gateway-test"))
    app_module.app.dependency_overrides[get_embedding_gateway] = lambda: StubEmbeddingGateway()
    try:
        with TestClient(app_module.app) as client:
            unauthorized = client.post("/v1/embeddings", json={"input": "hello"})
            assert unauthorized.status_code == 401

            response = client.post(
                "/v1/embeddings",
                json={"input": "hello"},
                headers={"Authorization": "Bearer " + "gateway-test"},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["object"] == "list"
        assert payload["data"] == [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}]
        assert payload["model"] == "jina-embeddings-v5-text-small"
    finally:
        app_module.app.dependency_overrides.pop(get_embedding_gateway, None)
