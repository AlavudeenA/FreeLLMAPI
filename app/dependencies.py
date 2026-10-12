from fastapi import Request

from app.embeddings.gateway import EmbeddingGateway
from app.gateway import LLMGateway


def get_gateway(request: Request) -> LLMGateway:
    return request.app.state.gateway


def get_embedding_gateway(request: Request) -> EmbeddingGateway:
    return request.app.state.embedding_gateway