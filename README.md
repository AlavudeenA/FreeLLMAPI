# FreeLLMAPI

A small OpenAI-compatible gateway for your agent projects. It tries the configured models in the order below, moving on when a provider is rate-limited, unavailable, or rejects a request. The response identifies the provider and model that answered and includes a record of every attempt.

## Model order

1. `nvidia/nemotron-3-super-120b-a12b` via NVIDIA NIM
2. `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` via NVIDIA NIM
3. `openai/gpt-oss-120b` via Groq
4. `z-ai/glm-5.3-flash` via NVIDIA NIM
5. `intern-s2` via InternAI
6. `command-a-plus-05-2026` via Cohere
7. `@cf/meta/llama-3.3-70b-instruct-fp8-fast` via Cloudflare Workers AI

Model names and provider API availability can change. Check each provider account for access to the exact model ID; override IDs and base URLs in `.env` if needed. Providers without a configured key are skipped automatically.

For Cohere, set `COHERE_API_KEY` in `.env`. The provider uses Cohere's native v2 chat endpoint at `https://api.cohere.com/v2/chat`.

For Cloudflare Workers AI, set both `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` in `.env`. Create a token with Workers AI Read and Workers AI Edit permissions. In the [Cloudflare Workers AI dashboard](https://dash.cloudflare.com/?to=/:account/ai/workers-ai), choose **Use REST API** to get the token and Account ID. The provider calls the account-specific `/ai/run/{model}` endpoint.

## Run

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and add the provider API keys you have.
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive API docs are at `http://127.0.0.1:8000/docs`; health check: `GET /health`.

## Call the endpoint

Send `POST /v1/chat/completions` with JSON. `messages` is required. Optional fields include `temperature`, `max_tokens`, `top_p`, `stop`, `tools`, `tool_choice`, `response_format`, `seed`, and `user`. Set `API_KEY` in `.env` to require `Authorization: Bearer <API_KEY>` on chat requests. Streaming is not supported.

```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Explain tool calling in one sentence." }
  ],
  "temperature": 0.2,
  "max_tokens": 300
}
```

PowerShell example:

```powershell
$body = @{
  messages = @(
    @{ role = "user"; content = "Hello" }
  )
  temperature = 0.2
  max_tokens = 200
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/chat/completions `
  -Method Post -ContentType "application/json" -Body $body
```

## Python client

Run `client.py` from another terminal. If the gateway has `API_KEY` configured, set `GATEWAY_API_KEY` to the same value; otherwise omit that variable.

```powershell
$env:GATEWAY_BASE_URL = "http://127.0.0.1:8000"
$env:GATEWAY_API_KEY = "<same value as API_KEY>"
python .\client.py "What can you help me with?"
```

For a client on another machine, set `GATEWAY_BASE_URL` to the gateway's reachable HTTPS address. Do not expose the development server directly to the internet.

## Response

Successful responses retain the familiar chat-completion `choices` and `usage` fields, and add gateway metadata:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created_at": "2026-10-09T12:00:00Z",
  "request_id": "...",
  "provider": "Groq",
  "provider_endpoint": "https://api.groq.com/openai/v1/chat/completions",
  "model": "openai/gpt-oss-120b",
  "provider_model": "openai/gpt-oss-120b",
  "choices": [],
  "usage": { "prompt_tokens": 12, "completion_tokens": 24, "total_tokens": 36 },
  "fallback_used": true,
  "attempts": [
    {
      "provider": "NVIDIA NIM",
      "model": "nvidia/nemotron-3-super-120b-a12b",
      "endpoint": "https://integrate.api.nvidia.com/v1/chat/completions",
      "outcome": "rate_limited",
      "status_code": 429,
      "latency_ms": 120,
      "error_code": "upstream_http_429"
    },
    {
      "provider": "Groq",
      "model": "openai/gpt-oss-120b",
      "outcome": "success",
      "status_code": 200,
      "latency_ms": 380,
      "error_code": null
    }
  ],
  "latency_ms": 510
}
```

`model` is the model name reported by the provider (or the configured ID if none is reported); `provider_model` is the ID sent upstream. Exhausting all providers returns HTTP `502` or `503` with an `error` object, `request_id`, and the same `attempts` detail. Each attempt omits provider error bodies so credentials and upstream internals are not leaked.

## Tests

```powershell
pytest
```

For local use, keep the server bound to localhost. Set `API_KEY` and use a protected deployment before making it reachable by other machines; never commit `.env` or expose provider keys to clients.
