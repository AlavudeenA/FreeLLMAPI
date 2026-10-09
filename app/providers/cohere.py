from typing import Any

import httpx


class CohereProvider:
    name = "Cohere"

    def __init__(self, client: httpx.AsyncClient, base_url: str, api_key: str | None):
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    @property
    def endpoint(self) -> str:
        return f"{self._base_url}/chat"

    @property
    def is_configured(self) -> bool:
        return self._api_key is not None

    async def complete(self, payload: dict[str, Any]) -> httpx.Response:
        messages = []
        for message in payload.get("messages", []):
            cohere_message = dict(message)
            if cohere_message.get("role") == "developer":
                cohere_message["role"] = "system"
            cohere_message.pop("name", None)
            messages.append(cohere_message)

        body: dict[str, Any] = {
            "model": payload["model"],
            "messages": messages,
            "stream": False,
        }
        for field in ("temperature", "max_tokens", "seed", "tools"):
            if field in payload:
                body[field] = payload[field]
        if "top_p" in payload:
            body["p"] = min(max(payload["top_p"], 0.01), 0.99)
        if "stop" in payload:
            stop = payload["stop"]
            body["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        tool_choice = payload.get("tool_choice")
        if isinstance(tool_choice, str):
            body["tool_choice"] = tool_choice.upper()

        response_format = payload.get("response_format")
        if isinstance(response_format, dict):
            format_type = response_format.get("type")
            if format_type == "json_schema":
                schema_config = response_format.get("json_schema", {})
                body["response_format"] = {"type": "json_object"}
                if isinstance(schema_config, dict) and isinstance(schema_config.get("schema"), dict):
                    body["response_format"]["json_schema"] = schema_config["schema"]
            elif format_type in {"json_object", "text"}:
                body["response_format"] = {"type": format_type}

        response = await self._client.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=body,
        )
        if not response.is_success:
            return response

        try:
            data = response.json()
        except ValueError:
            return response
        if not isinstance(data, dict):
            return response

        message = data.get("message")
        if not isinstance(message, dict):
            return response

        content = message.get("content")
        if isinstance(content, list):
            text = "".join(
                block["text"]
                for block in content
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
            )
        else:
            text = content if isinstance(content, str) else None

        assistant_message: dict[str, Any] = {"role": "assistant", "content": text}
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            assistant_message["tool_calls"] = tool_calls

        finish_reason = str(data.get("finish_reason", "COMPLETE")).upper()
        finish_reason = {
            "COMPLETE": "stop",
            "STOP_SEQUENCE": "stop",
            "MAX_TOKENS": "length",
            "TOOL_CALL": "tool_calls",
        }.get(finish_reason, "error")
        normalized: dict[str, Any] = {
            "id": data.get("id"),
            "model": data.get("model"),
            "choices": [{"index": 0, "message": assistant_message, "finish_reason": finish_reason}],
        }

        meta = data.get("meta")
        tokens = meta.get("tokens") if isinstance(meta, dict) else None
        if isinstance(tokens, dict):
            input_tokens = tokens.get("input_tokens", 0)
            output_tokens = tokens.get("output_tokens", 0)
            if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
                normalized["usage"] = {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }

        return httpx.Response(response.status_code, json=normalized, request=response.request)