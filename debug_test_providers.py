from __future__ import annotations

import asyncio
import httpx

from app.config import Settings, secret_value
from app.providers.base import LLMProvider
from app.providers.cloudflare import CloudflareWorkersAIProvider
from app.providers.cohere import CohereProvider
from app.providers.groq import GroqProvider
from app.providers.intern import InternProvider
from app.providers.nvidia import NvidiaProvider
from app.providers.openrouter import OpenRouterProvider


async def call_provider(name: str, provider: LLMProvider, model: str) -> None:
    if not provider.is_configured:
        print(f'=== {name} ===')
        print('SKIPPED: missing API credentials or required account configuration')
        print()
        return

    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': f'Reply with exactly {name.replace(" ", "_").upper()}_OK and nothing else.'}],
        'temperature': 0,
    }

    try:
        resp = await provider.complete(payload)
        print(f'=== {name} ===')
        print('endpoint:', provider.endpoint)
        print('requested model:', model)
        print(f'status: {resp.status_code}')
        try:
            data = resp.json()
        except ValueError:
            print('error: provider returned invalid JSON')
            print()
            return

        if not resp.is_success:
            error = data.get('error', data.get('errors', '')) if isinstance(data, dict) else ''
            print('error:', str(error)[:500])
            print()
            return

        choices = data.get('choices') if isinstance(data, dict) else None
        message = choices[0].get('message', {}) if isinstance(choices, list) and choices else {}
        content = message.get('content') if isinstance(message, dict) else None
        print('returned model:', data.get('model') if isinstance(data, dict) else None)
        print('reply:', repr(content)[:500] if isinstance(content, str) else 'no visible text')
    except Exception as exc:
        print(f'=== {name} ===')
        print('request failed:', type(exc).__name__, str(exc)[:300])
    print()


async def main() -> None:
    settings = Settings()
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        providers = [
            (
                'NVIDIA NIM',
                NvidiaProvider(client, settings.nvidia_base_url, secret_value(settings.nvidia_api_key)),
                settings.nvidia_nemotron_super_model,
            ),
            (
                'Groq',
                GroqProvider(client, settings.groq_base_url, secret_value(settings.groq_api_key)),
                settings.groq_model,
            ),
            (
                'OpenRouter',
                OpenRouterProvider(client, settings.openrouter_base_url, secret_value(settings.openrouter_api_key)),
                settings.openrouter_model,
            ),
            (
                'InternAI',
                InternProvider(client, settings.intern_base_url, secret_value(settings.intern_api_key)),
                settings.intern_model,
            ),
            (
                'Cohere',
                CohereProvider(client, settings.cohere_base_url, secret_value(settings.cohere_api_key)),
                settings.cohere_model,
            ),
            (
                'Cloudflare Workers AI',
                CloudflareWorkersAIProvider(
                    client,
                    settings.cloudflare_base_url,
                    settings.cloudflare_account_id,
                    secret_value(settings.cloudflare_api_token),
                    settings.cloudflare_model,
                ),
                settings.cloudflare_model,
            ),
        ]
        for name, provider, model in providers:
            await call_provider(name, provider, model)


if __name__ == '__main__':
    asyncio.run(main())
