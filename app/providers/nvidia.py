from app.providers.base import OpenAICompatibleProvider


class NvidiaProvider(OpenAICompatibleProvider):
    name = "NVIDIA NIM"