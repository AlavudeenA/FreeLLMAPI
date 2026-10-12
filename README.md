# FreeLLMAPI

A FastAPI gateway that returns chat-completion JSON from NVIDIA, Groq, InternAI, Cohere, or Cloudflare Workers AI. Choose a provider explicitly, or omit the choice to use the configured fallback order.

The gateway also exposes an OpenAI-compatible embeddings endpoint at `/v1/embeddings`. It tries NVIDIA Nemotron 3 Embed 1B, Jina Embeddings v5 Text Small, Qwen3-Embedding-0.6B, BGE-M3, then Nomic Embed Text. NVIDIA and Jina use their hosted APIs; Qwen, BGE-M3, and Nomic run through a local Ollama server.

## Setup

From the project directory in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add the API keys for the providers you want to use to `.env`. Cloudflare also needs `CLOUDFLARE_ACCOUNT_ID`. Provider settings are listed in `.env.example`.

For the local fallback models, install and start Ollama, then pull the models you want available:

```powershell
ollama pull qwen3-embedding:0.6b
ollama pull bge-m3
ollama pull nomic-embed-text
```

Set `JINA_API_KEY` in `.env` to enable Jina. The NVIDIA route uses the existing `NVIDIA_API_KEY`. Ollama is expected at `http://127.0.0.1:11434` by default; configure `OLLAMA_BASE_URL` if needed. A missing cloud API key skips that route; a down or rate-limited provider falls through to the next model.

## Reranking

The `/v1/rerank` endpoint uses Jina Reranker v3 to sort retrieved candidate documents by relevance. It reuses `JINA_API_KEY` and can be pinned to a configured model using `model`.

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/rerank `
  -ContentType 'application/json' `
  -Body '{"query":"What is retrieval augmented generation?","documents":["A recipe for pancakes.","RAG combines retrieval with language generation."],"top_n":1}'
```

Python callers can send the query and retrieved documents directly:

```python
import json
from urllib.request import Request, urlopen


query = "What is retrieval augmented generation?"
documents = [
    "A recipe for pancakes.",
    "RAG combines retrieval with language generation.",
]
top_n = 1
request = Request(
    "http://127.0.0.1:8000/v1/rerank",
    data=json.dumps({
        "model": "jina-reranker-v3",
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
with urlopen(request) as response:
    print(json.load(response))
```

The response's `results` are ordered by relevance and include each document's original `index`, text, and `relevance_score`. Use the index to associate each result with the document supplied in the request.

## Run

Start the gateway:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Run this from the project directory after completing Setup and activating the virtual environment. `--reload` restarts the server when code changes. The API is at `http://127.0.0.1:8000`; interactive docs are at `http://127.0.0.1:8000/docs`.

In a second terminal, use the default fallback order:

```powershell
python .\client.py "What can you help me with?"
```

Or choose a provider:

```powershell
python .\client.py --provider cloudflare_workers_ai "What can you help me with?"
```

Valid choices: `nvidia`, `groq`, `intern_ai`, `cohere`, `cloudflare_workers_ai`. Without `--provider`, the gateway uses its configured fallback order. The response is JSON and includes the provider and endpoint that answered.

You can also call the chat-completion endpoint directly from Python:

```python
import json
from urllib.request import Request, urlopen


request = Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(
        {"messages": [{"role": "user", "content": "What can you help me with?"}]}
    ).encode(),
    headers={"Content-Type": "application/json"},
)
with urlopen(request) as response:
    print(json.load(response))
```

## Embeddings

Send one string or a list of strings to `/v1/embeddings`:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/embeddings `
  -ContentType 'application/json' `
  -Body '{"input":["A sample document","Another document"]}'
```

Or call the embeddings endpoint from Python:

```python
import json
from urllib.request import Request, urlopen


request = Request(
    "http://127.0.0.1:8000/v1/embeddings",
    data=json.dumps({"input": "Text to embed"}).encode(),
    headers={"Content-Type": "application/json"},
)
with urlopen(request) as response:
    print(json.load(response))
```

Pass a configured `model` value to pin a request to one model and disable failover. Responses include the selected provider, model, endpoint, and attempted routes. Since different embedding models produce different vector spaces, keep the returned model consistent for indexing and queries; changing models may require re-embedding the existing index.
