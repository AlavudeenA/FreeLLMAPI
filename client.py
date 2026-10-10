from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import httpx

def print_attempts(attempts: Any) -> None:
    if not isinstance(attempts, list):
        return

    contacted = [attempt for attempt in attempts if isinstance(attempt, dict) and attempt.get("outcome") != "skipped"]
    if not contacted:
        return

    print("Provider endpoints contacted:")
    for attempt in contacted:
        status = attempt.get("status_code")
        status_text = f", HTTP {status}" if status is not None else ""
        print(
            f"  {attempt.get('provider', 'unknown')} | {attempt.get('model', 'unknown')} | "
            f"{attempt.get('endpoint', 'unknown endpoint')} | {attempt.get('outcome', 'unknown')}{status_text}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a prompt to the Free LLM API gateway.")
    parser.add_argument("prompt", nargs="*", help="Prompt text; defaults to a short greeting.")
    parser.add_argument(
        "--model",
        help="Use a specific model name, for example 'nvidia/nemotron-3-super-120b-a12b' or 'openai/gpt-oss-120b'.",
    )
    args = parser.parse_args()

    base_url = os.getenv("GATEWAY_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    gateway_api_key = os.getenv("GATEWAY_API_KEY")
    prompt_text = " ".join(args.prompt).strip() or "Say hello in one sentence."
    headers = {"Content-Type": "application/json"}
    if gateway_api_key:
        headers["Authorization"] = f"Bearer {gateway_api_key}"

    try:
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.2,
        }
        if args.model is not None:
            payload["model"] = args.model

        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120.0,
        )
    except httpx.RequestError as error:
        print(f"Could not connect to the gateway at {base_url}: {error}", file=sys.stderr)
        if base_url.startswith(("http://127.0.0.1", "http://localhost")):
            print(
                "Start the gateway in another terminal with: "
                "python -m uvicorn app.main:app --host 127.0.0.1 --port 8000",
                file=sys.stderr,
            )
        else:
            print("Check that the gateway URL is reachable and the server is running.", file=sys.stderr)
        return 1

    if not response.is_success:
        try:
            error_result = response.json()
        except ValueError:
            error_result = {}
        error_details = error_result.get("error") if isinstance(error_result, dict) else None
        if isinstance(error_details, dict):
            message = error_details.get("message", "Request failed")
        else:
            message = error_result.get("detail", response.text) if isinstance(error_result, dict) else response.text
        print(f"Gateway returned HTTP {response.status_code}: {message}", file=sys.stderr)
        print_attempts(error_result.get("attempts") if isinstance(error_result, dict) else None)
        return 1

    try:
        result: dict[str, Any] = response.json()
    except ValueError:
        print("Gateway returned an invalid JSON response.", file=sys.stderr)
        return 1

    print(f"Provider: {result.get('provider', 'unknown')}")
    print(f"Model: {result.get('model', 'unknown')}")
    print(f"Provider model: {result.get('provider_model', 'unknown')}")
    print(f"Provider endpoint: {result.get('provider_endpoint', 'unknown endpoint')}")
    print(f"Fallback used: {result.get('fallback_used', False)}")
    print_attempts(result.get("attempts"))

    choices = result.get("choices")
    message = choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        print(f"\n{content}")
    else:
        print("\nNo text content was returned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())