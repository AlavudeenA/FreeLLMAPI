from dataclasses import dataclass
from typing import Protocol

import httpx


class EmbeddingProvider(Protocol):
    name: str

    @property
    def endpoint(self) -> str: ...

    @property
    def is_configured(self) -> bool: ...

    async def embed(self, inputs: list[str], model: str) -> httpx.Response: ...


@dataclass(frozen=True)
class EmbeddingRoute:
    provider: EmbeddingProvider
    model: str
