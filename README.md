# FreeLLMAPI

A FastAPI gateway that returns chat-completion JSON from NVIDIA, Groq, InternAI, Cohere, or Cloudflare Workers AI. Choose a provider explicitly, or omit the choice to use the configured fallback order.

## Setup

From the project directory in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add the API keys for the providers you want to use to `.env`. Cloudflare also needs `CLOUDFLARE_ACCOUNT_ID`. Provider settings are listed in `.env.example`.

## Run

Start the gateway:

```powershell
python -m uvicorn app.main:app --reload
```

The API is at `http://127.0.0.1:8000`; interactive docs are at `http://127.0.0.1:8000/docs`.

In a second terminal, use the default fallback order:

```powershell
python .\client.py "What can you help me with?"
```

Or choose a provider:

```powershell
python .\client.py --provider cloudflare_workers_ai "What can you help me with?"
```

Valid choices: `nvidia`, `groq`, `intern_ai`, `cohere`, `cloudflare_workers_ai`. Without `--provider`, the gateway uses its configured fallback order. The response is JSON and includes the provider and endpoint that answered.
