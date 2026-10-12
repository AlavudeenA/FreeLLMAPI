import httpx

from app.config import Settings, secret_value
from app.embeddings.gateway import EmbeddingGateway
from app.embeddings.providers.base import EmbeddingRoute
from app.embeddings.providers.jina import JinaEmbeddingProvider
from app.embeddings.providers.nvidia import NvidiaEmbeddingProvider
from app.embeddings.providers.ollama import OllamaEmbeddingProvider


def create_embedding_gateway(settings: Settings, client: httpx.AsyncClient) -> EmbeddingGateway:
    nvidia = NvidiaEmbeddingProvider(
        client,
        settings.nvidia_base_url,
        secret_value(settings.nvidia_api_key),
    )
    jina = JinaEmbeddingProvider(
        client,
        settings.jina_base_url,
        secret_value(settings.jina_api_key),
    )
    qwen = OllamaEmbeddingProvider(client, settings.ollama_base_url, settings.qwen_ollama_model)
    bge = OllamaEmbeddingProvider(client, settings.ollama_base_url, settings.bge_ollama_model)
    nomic = OllamaEmbeddingProvider(client, settings.ollama_base_url, settings.nomic_ollama_model)

    return EmbeddingGateway(
        (
            EmbeddingRoute(nvidia, settings.nvidia_embedding_model),
            EmbeddingRoute(jina, settings.jina_embedding_model),
            EmbeddingRoute(qwen, settings.qwen_embedding_model),
            EmbeddingRoute(bge, settings.bge_embedding_model),
            EmbeddingRoute(nomic, settings.nomic_embedding_model),
        )
    )
