from app.providers.base import OpenAICompatibleProvider


class InternProvider(OpenAICompatibleProvider):
    name = "InternAI"
    provider_id = "intern_ai"