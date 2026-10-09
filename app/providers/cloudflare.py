from typing import Any

import httpx


class CloudflareWorkersAIProvider:
    name = "Cloudflare Workers AI"
    provider_id = "cloudflare_workers_ai"

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        account_id: str | None,
        api_token: str | None,
        model: str,
    ):
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._account_id = account_id.strip() if account_id else None
        self._api_token = api_token
        self._model = model

    @property
    def endpoint(self) -> str:
        account_id = self._account_id or "{account_id}"
        return f"{self._base_url}/accounts/{account_id}/ai/run/{self._model}"

    @property
    def is_configured(self) -> bool:
        return self._account_id is not None and self._api_token is not None

    async def complete(self, payload: dict[str, Any]) -> httpx.Response:
        body: dict[str, Any] = {"messages": payload["messages"]}
        for field in ("temperature", "max_tokens", "top_p", "seed"):
            if field in payload:
                body[field] = payload[field]

        response = await self._client.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self._api_token}"},
            json=body,
        )
        if not response.is_success:
            return response

        try:
            data = response.json()
        except ValueError:
            return response
        if not isinstance(data, dict) or data.get("success") is not True:
            return response

        result = data.get("result")
        if not isinstance(result, dict):
            return response

        text = result.get("response")
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": text if isinstance(text, str) else None,
        }
        tool_calls = result.get("tool_calls")
        if isinstance(tool_calls, list):
            assistant_message["tool_calls"] = tool_calls

        normalized: dict[str, Any] = {
            "id": result.get("id"),
            "model": self._model,
            "choices": [
                {
                    "index": 0,
                    "message": assistant_message,
                    "finish_reason": result.get("finish_reason", "stop"),
                }
            ],
        }
        usage = result.get("usage")
        if isinstance(usage, dict):
            normalized["usage"] = usage

        return httpx.Response(response.status_code, json=normalized, request=response.request)