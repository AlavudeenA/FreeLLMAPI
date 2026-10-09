from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class LLMProvider(Protocol):
    name: str
    provider_id: str

    @property
    def endpoint(self) -> str: ...

    @property
    def is_configured(self) -> bool: ...

    async def complete(self, payload: dict[str, Any]) -> httpx.Response: ...


@dataclass(frozen=True)
class ProviderModel:
    provider: LLMProvider
    model: str


class OpenAICompatibleProvider:
    name: str

    def __init__(self, client: httpx.AsyncClient, base_url: str, api_key: str | None):
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    @property
    def endpoint(self) -> str:
        return f"{self._base_url}/chat/completions"

    @property
    def is_configured(self) -> bool:
        return self._api_key is not None

    async def complete(self, payload: dict[str, Any]) -> httpx.Response:
        return await self._client.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
        )