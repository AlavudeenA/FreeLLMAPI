from fastapi import Request

from app.embeddings.gateway import EmbeddingGateway
from app.gateway import LLMGateway
from app.reranking.gateway import RerankGateway


def get_gateway(request: Request) -> LLMGateway:
    return request.app.state.gateway


def get_embedding_gateway(request: Request) -> EmbeddingGateway:
    return request.app.state.embedding_gateway


def get_rerank_gateway(request: Request) -> RerankGateway:
    return request.app.state.rerank_gateway