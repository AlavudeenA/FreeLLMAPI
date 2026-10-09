from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        values[key.strip()] = value.strip()
    return values


def pretty_text(data: dict) -> str:
    if isinstance(data, dict) and data.get('choices'):
        first = data['choices'][0]
        message = first.get('message', {})
        if isinstance(message, dict):
            content = message.get('content')
            if isinstance(content, str):
                return content.strip()
    if isinstance(data, dict) and 'text' in data and isinstance(data['text'], str):
        return data['text'].strip()
    return json.dumps(data, ensure_ascii=False, default=str)[:1000]


async def call_provider(name: str, key: str | None, base_url: str, model: str) -> None:
    if not key:
        print(f'=== {name} ===')
        print('SKIPPED: no key configured')
        print()
        return

    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': 'Say hello in exactly 3 words.'}],
        'max_tokens': 32,
        'temperature': 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post(
                f'{base_url.rstrip("/")}/chat/completions',
                headers={'Authorization': f'Bearer {key}'},
                json=payload,
            )
        print(f'=== {name} ===')
        print(f'status: {resp.status_code}')
        try:
            data = resp.json()
        except Exception:
            print('raw response:', resp.text[:800])
            print()
            return
        if isinstance(data, dict) and 'error' in data:
            print('error:', data.get('error'))
        else:
            print('model:', data.get('model'))
            print('reply:', pretty_text(data))
    except Exception as exc:
        print(f'=== {name} ===')
        print('exception:', repr(exc))
    print()


async def main() -> None:
    env = load_env(Path(__file__).resolve().parent / '.env')
    providers = [
        ('NVIDIA NIM', env.get('NVIDIA_API_KEY'), 'https://integrate.api.nvidia.com/v1', 'nvidia/nemotron-3-super-120b-a12b'),
        ('Groq', env.get('GROQ_API_KEY'), 'https://api.groq.com/openai/v1', 'openai/gpt-oss-120b'),
        ('InternAI', env.get('INTERN_API_KEY'), 'https://chat.intern-ai.org.cn/api/v1', 'intern-s2'),
    ]
    await asyncio.gather(*(call_provider(name, key, base_url, model) for name, key, base_url, model in providers))


if __name__ == '__main__':
    asyncio.run(main())
