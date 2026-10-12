import httpx

from app.config import Settings, secret_value
from app.reranking.gateway import RerankGateway
from app.reranking.provider import JinaRerankerProvider


def create_rerank_gateway(settings: Settings, client: httpx.AsyncClient) -> RerankGateway:
    provider = JinaRerankerProvider(
        client,
        settings.jina_base_url,
        secret_value(settings.jina_api_key),
    )
    return RerankGateway(provider, settings.jina_reranker_model)
