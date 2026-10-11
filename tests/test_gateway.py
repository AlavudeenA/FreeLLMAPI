import asyncio
import json

import httpx

from app.config import Settings
from app.providers.factory import create_gateway
from app.schemas import ChatCompletionRequest


def isolated_settings(**overrides) -> Settings:
    provider_settings = {
        "nvidia_api_key": None,
        "groq_api_key": None,
        "openrouter_api_key": None,
        "intern_api_key": None,
        "cohere_api_key": None,
        "cloudflare_api_token": None,
        "cloudflare_account_id": None,
    }
    return Settings(_env_file=None, **(provider_settings | overrides))


def test_rate_limit_falls_through_to_next_available_provider():
    async def run_test():
        def respond(request: httpx.Request) -> httpx.Response:
            if request.headers["Authorization"] == "Bearer nvidia-test":
                return httpx.Response(429, json={"error": {"message": "rate limit"}})
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "model": "openai/gpt-oss-120b",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ready"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        settings = isolated_settings(nvidia_api_key="nvidia-test", groq_api_key="groq-test")
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await create_gateway(settings, client).complete(
                ChatCompletionRequest(messages=[{"role": "user", "content": "Hello"}]),
                request_id="request-test",
            )

        assert result.provider == "Groq"
        assert result.model == "openai/gpt-oss-120b"
        assert result.provider_model == "openai/gpt-oss-120b"
        assert result.provider_endpoint == "https://api.groq.com/openai/v1/chat/completions"
        assert result.fallback_used is True
        assert [attempt.outcome for attempt in result.attempts] == ["rate_limited", "rate_limited", "success"]
        assert [attempt.provider for attempt in result.attempts] == ["NVIDIA NIM", "NVIDIA NIM", "Groq"]

    asyncio.run(run_test())


def test_empty_success_falls_through_to_next_available_provider():
    async def run_test():
        def respond(request: httpx.Request) -> httpx.Response:
            if request.headers["Authorization"] == "Bearer nvidia-test":
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"message": {"role": "assistant", "content": None, "reasoning_content": "thinking"}}
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={"model": "openai/gpt-oss-120b", "choices": [{"message": {"content": "ready"}}]},
            )

        settings = isolated_settings(
            nvidia_api_key="nvidia-test",
            groq_api_key="groq-test",
            intern_api_key=None,
            cohere_api_key=None,
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await create_gateway(settings, client).complete(
                ChatCompletionRequest(messages=[{"role": "user", "content": "Hello"}]),
                request_id="request-test",
            )

        assert result.provider == "Groq"
        assert result.choices[0]["message"]["content"] == "ready"
        assert result.fallback_used is True
        assert [attempt.outcome for attempt in result.attempts] == ["unavailable", "unavailable", "success"]

    asyncio.run(run_test())


def test_skipped_unconfigured_models_do_not_count_as_fallback():
    async def run_test():
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"model": "openai/gpt-oss-120b", "choices": [{"message": {"content": "ready"}}]},
            )

        settings = isolated_settings(groq_api_key="groq-test")
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await create_gateway(settings, client).complete(
                ChatCompletionRequest(messages=[{"role": "user", "content": "Hello"}]),
                request_id="request-test",
            )

        assert result.provider == "Groq"
        assert result.fallback_used is False
        assert [attempt.outcome for attempt in result.attempts] == ["skipped", "skipped", "success"]

    asyncio.run(run_test())


def test_explicit_model_path_selects_matching_provider_and_model():
    async def run_test():
        def respond(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.read())
            assert payload["model"] == "nvidia/nemotron-3-super-120b-a12b"
            return httpx.Response(
                200,
                json={
                    "model": "nvidia/nemotron-3-super-120b-a12b",
                    "choices": [{"message": {"role": "assistant", "content": "nvidia direct reply"}}],
                },
            )

        settings = isolated_settings(nvidia_api_key="nvidia-test", groq_api_key="groq-test")
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await create_gateway(settings, client).complete(
                ChatCompletionRequest(
                    model="nvidia/nemotron-3-super-120b-a12b",
                    messages=[{"role": "user", "content": "Hello"}],
                ),
                request_id="model-path-test",
            )

        assert result.provider == "NVIDIA NIM"
        assert result.model == "nvidia/nemotron-3-super-120b-a12b"
        assert result.provider_model == "nvidia/nemotron-3-super-120b-a12b"
        assert result.fallback_used is False

    asyncio.run(run_test())


def test_explicit_model_selection_only_calls_selected_model():
    async def run_test():
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            payload = json.loads(request.read())
            assert payload["model"] == "openai/gpt-oss-120b"
            return httpx.Response(
                200,
                json={
                    "model": "openai/gpt-oss-120b",
                    "choices": [{"message": {"role": "assistant", "content": "groq reply"}}],
                },
            )

        settings = isolated_settings(
            nvidia_api_key="nvidia-test",
            groq_api_key="groq-test",
            intern_api_key=None,
            cohere_api_key=None,
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await create_gateway(settings, client).complete(
                ChatCompletionRequest(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": "Hello"}],
                ),
                request_id="selected-model-test",
            )

        assert result.provider == "Groq"
        assert result.fallback_used is False
        assert len(requests) == 1
        assert requests[0].url == "https://api.groq.com/openai/v1/chat/completions"

    asyncio.run(run_test())


def test_openrouter_uses_configured_model_fallback_order():
    async def run_test():
        def respond(request: httpx.Request) -> httpx.Response:
            assert request.headers["Authorization"] == "Bearer openrouter-test-token"
            payload = json.loads(request.read())
            assert "model" not in payload
            assert payload["models"] == ["qwen/qwen3.8-27b:free", "openrouter/free"]
            return httpx.Response(
                200,
                json={
                    "model": "qwen/qwen3.8-27b:free",
                    "choices": [{"message": {"role": "assistant", "content": "ready"}}],
                },
            )

        settings = isolated_settings(openrouter_api_key="openrouter-test-token")
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await create_gateway(settings, client).complete(
                ChatCompletionRequest(
                    model="openrouter/free",
                    messages=[{"role": "user", "content": "Hello"}],
                ),
                request_id="openrouter-fallback-test",
            )

        assert result.provider == "OpenRouter"
        assert result.model == "qwen/qwen3.8-27b:free"
        assert result.provider_model == "openrouter/free"

    asyncio.run(run_test())


def test_cohere_native_chat_is_normalized_for_gateway_response():
    async def run_test():
        def respond(request: httpx.Request) -> httpx.Response:
            assert request.url == "https://api.cohere.com/v2/chat"
            assert request.headers["Authorization"] == "Bearer cohere-test-token"
            payload = json.loads(request.read())
            assert payload["model"] == "command-a-plus-05-2026"
            assert payload["messages"][0]["role"] == "system"
            assert payload["p"] == 0.4
            assert payload["stop_sequences"] == ["END"]
            assert "top_p" not in payload
            return httpx.Response(
                200,
                json={
                    "id": "cohere-chat-test",
                    "model": "command-a-plus-05-2026",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "ready"}],
                    },
                    "finish_reason": "COMPLETE",
                    "meta": {"tokens": {"input_tokens": 2, "output_tokens": 1}},
                },
            )

        settings = isolated_settings(
            nvidia_api_key=None,
            groq_api_key=None,
            intern_api_key=None,
            cohere_api_key="cohere-test-token",
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await create_gateway(settings, client).complete(
                ChatCompletionRequest(
                    messages=[
                        {"role": "developer", "content": "Be concise"},
                        {"role": "user", "content": "Hello"},
                    ],
                    top_p=0.4,
                    stop="END",
                ),
                request_id="cohere-request-test",
            )

        assert result.provider == "Cohere"
        assert result.model == "command-a-plus-05-2026"
        assert result.provider_model == "command-a-plus-05-2026"
        assert result.provider_endpoint == "https://api.cohere.com/v2/chat"
        assert result.choices[0]["message"]["content"] == "ready"
        assert result.usage == {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}

    asyncio.run(run_test())


def test_cloudflare_workers_ai_native_response_is_normalized():
    async def run_test():
        def respond(request: httpx.Request) -> httpx.Response:
            expected_endpoint = (
                "https://api.cloudflare.com/client/v4/accounts/account-test/ai/run/"
                "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
            )
            assert request.url == expected_endpoint
            assert request.headers["Authorization"] == "Bearer cloudflare-test-token"
            payload = json.loads(request.read())
            assert payload["messages"] == [{"role": "user", "content": "Hello"}]
            assert "model" not in payload
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "result": {
                        "response": "ready",
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                    },
                    "errors": [],
                    "messages": [],
                },
            )

        settings = isolated_settings(
            nvidia_api_key=None,
            groq_api_key=None,
            intern_api_key=None,
            cohere_api_key=None,
            cloudflare_api_token="cloudflare-test-token",
            cloudflare_account_id="account-test",
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await create_gateway(settings, client).complete(
                ChatCompletionRequest(messages=[{"role": "user", "content": "Hello"}]),
                request_id="cloudflare-request-test",
            )

        assert result.provider == "Cloudflare Workers AI"
        assert result.model == "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
        assert result.provider_endpoint == (
            "https://api.cloudflare.com/client/v4/accounts/account-test/ai/run/"
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
        )
        assert result.choices[0]["message"]["content"] == "ready"
        assert result.usage == {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}

    asyncio.run(run_test())


def test_intern_ai_chat_completion_uses_openai_compatible_endpoint():
    async def run_test():
        def respond(request: httpx.Request) -> httpx.Response:
            assert request.url == "https://chat.intern-ai.org.cn/api/v1/chat/completions"
            assert request.headers["Authorization"] == "Bearer intern-test-token"
            payload = json.loads(request.read())
            assert payload["model"] == "intern-s2"
            assert payload["messages"] == [{"role": "user", "content": "Hello"}]
            return httpx.Response(
                200,
                json={
                    "id": "intern-chat-test",
                    "model": "intern-s2",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ready"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        settings = isolated_settings(intern_api_key="intern-test-token")
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await create_gateway(settings, client).complete(
                ChatCompletionRequest(messages=[{"role": "user", "content": "Hello"}]),
                request_id="intern-request-test",
            )

        assert result.provider == "InternAI"
        assert result.model == "intern-s2"
        assert result.provider_model == "intern-s2"
        assert result.provider_endpoint == "https://chat.intern-ai.org.cn/api/v1/chat/completions"
        assert result.choices[0]["message"]["content"] == "ready"
        assert result.usage == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    asyncio.run(run_test())
