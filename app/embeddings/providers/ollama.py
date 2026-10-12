import httpx


class OllamaEmbeddingProvider:
    name = "Ollama"

    def __init__(self, client: httpx.AsyncClient, base_url: str, model: str):
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._model = model

    @property
    def endpoint(self) -> str:
        return f"{self._base_url}/api/embed"

    @property
    def is_configured(self) -> bool:
        return True

    async def embed(self, inputs: list[str], model: str) -> httpx.Response:
        response = await self._client.post(
            self.endpoint,
            json={"model": self._model, "input": inputs},
        )
        if not response.is_success:
            return response

        try:
            result = response.json()
        except ValueError:
            return response
        if not isinstance(result, dict) or not isinstance(result.get("embeddings"), list):
            return response

        normalized = {
            "model": model,
            "data": [
                {"object": "embedding", "index": index, "embedding": embedding}
                for index, embedding in enumerate(result["embeddings"])
            ],
        }
        return httpx.Response(response.status_code, json=normalized, request=response.request)
