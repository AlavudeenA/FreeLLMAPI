from contextlib import asynccontextmanager
from secrets import compare_digest
from typing import Annotated
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from app.config import Settings
from app.dependencies import get_gateway
from app.gateway import LLMGateway, UpstreamExhaustedError
from app.providers.factory import create_gateway
from app.schemas import ChatCompletionRequest, ErrorDetails, ErrorResponse

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        app.state.gateway = create_gateway(settings, client)
        yield


app = FastAPI(title="Free LLM API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def create_chat_completion(
    body: ChatCompletionRequest,
    gateway: Annotated[LLMGateway, Depends(get_gateway)],
    authorization: str | None = Header(default=None),
):
    request_id = uuid4().hex
    if settings.api_key is not None:
        expected = f"Bearer {settings.api_key.get_secret_value()}"
        if authorization is None or not compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    try:
        return await gateway.complete(body, request_id)
    except UpstreamExhaustedError as error:
        response = ErrorResponse(
            request_id=request_id,
            error=ErrorDetails(code=error.code, message=str(error)),
            attempts=error.attempts,
        )
        return JSONResponse(status_code=error.status_code, content=response.model_dump(mode="json"))