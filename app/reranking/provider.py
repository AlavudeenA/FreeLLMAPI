import httpx


class JinaRerankerProvider:
    name = "Jina AI"

    def __init__(self, client: httpx.AsyncClient, base_url: str, api_key: str | None):
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    @property
    def endpoint(self) -> str:
        return f"{self._base_url}/rerank"

    @property
    def is_configured(self) -> bool:
        return self._api_key is not None

    async def rerank(self, payload: dict[str, object]) -> httpx.Response:
        return await self._client.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
        )
