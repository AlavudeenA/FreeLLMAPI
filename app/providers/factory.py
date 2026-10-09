import httpx

from app.config import Settings, secret_value
from app.gateway import LLMGateway
from app.providers.base import ProviderModel
from app.providers.cloudflare import CloudflareWorkersAIProvider
from app.providers.cohere import CohereProvider
from app.providers.groq import GroqProvider
from app.providers.intern import InternProvider
from app.providers.nvidia import NvidiaProvider


def create_gateway(settings: Settings, client: httpx.AsyncClient) -> LLMGateway:
    nvidia = NvidiaProvider(client, settings.nvidia_base_url, secret_value(settings.nvidia_api_key))
    groq = GroqProvider(client, settings.groq_base_url, secret_value(settings.groq_api_key))
    intern = InternProvider(client, settings.intern_base_url, secret_value(settings.intern_api_key))
    cohere = CohereProvider(client, settings.cohere_base_url, secret_value(settings.cohere_api_key))
    cloudflare = CloudflareWorkersAIProvider(
        client,
        settings.cloudflare_base_url,
        settings.cloudflare_account_id,
        secret_value(settings.cloudflare_api_token),
        settings.cloudflare_model,
    )

    routes = (
        ProviderModel(nvidia, settings.nvidia_nemotron_super_model),
        ProviderModel(nvidia, settings.nvidia_nemotron_nano_model),
        ProviderModel(groq, settings.groq_model),
        ProviderModel(nvidia, settings.nvidia_glm_flash_model),
        ProviderModel(intern, settings.intern_model),
        ProviderModel(cohere, settings.cohere_model),
        ProviderModel(cloudflare, settings.cloudflare_model),
    )
    return LLMGateway(routes)