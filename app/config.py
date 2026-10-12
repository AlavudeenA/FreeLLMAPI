from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def secret_value(value: SecretStr | None) -> str | None:
    if value is None:
        return None
    secret = value.get_secret_value().strip()
    return secret or None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    api_key: SecretStr | None = None
    request_timeout_seconds: float = 60.0

    nvidia_api_key: SecretStr | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_nemotron_super_model: str = "nvidia/nemotron-3-super-120b-a12b"
    nvidia_nemotron_nano_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    nvidia_glm_flash_model: str = "z-ai/glm-5.3-flash"
    nvidia_embedding_model: str = "nvidia/nemotron-3-embed-1b"

    jina_api_key: SecretStr | None = None
    jina_base_url: str = "https://api.jina.ai/v1"
    jina_embedding_model: str = "jina-embeddings-v5-text-small"

    ollama_base_url: str = "http://127.0.0.1:11434"
    qwen_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    qwen_ollama_model: str = "qwen3-embedding:0.6b"
    bge_embedding_model: str = "BAAI/bge-m3"
    bge_ollama_model: str = "bge-m3"
    nomic_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    nomic_ollama_model: str = "nomic-embed-text"

    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openrouter/free"

    intern_api_key: SecretStr | None = None
    intern_base_url: str = "https://chat.intern-ai.org.cn/api/v1"
    intern_model: str = "intern-s2"

    cohere_api_key: SecretStr | None = None
    cohere_base_url: str = "https://api.cohere.com/v2"
    cohere_model: str = "command-a-plus-05-2026"

    cloudflare_api_token: SecretStr | None = None
    cloudflare_account_id: str | None = None
    cloudflare_base_url: str = "https://api.cloudflare.com/client/v4"
    cloudflare_model: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
