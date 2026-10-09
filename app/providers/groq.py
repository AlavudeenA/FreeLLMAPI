from app.providers.base import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    name = "Groq"
    provider_id = "groq"