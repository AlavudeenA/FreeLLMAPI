from typing import Any

import httpx

from app.providers.base import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "OpenRouter"
    provider_id = "openrouter"
    fallback_models = ("qwen/qwen3.8-27b:free", "openrouter/free")

    async def complete(self, payload: dict[str, Any]) -> httpx.Response:
        request_payload = {key: value for key, value in payload.items() if key != "model"}
        request_payload["models"] = list(self.fallback_models)
        return await super().complete(request_payload)
