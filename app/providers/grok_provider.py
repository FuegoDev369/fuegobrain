"""
app/providers/grok_provider.py
Adapter Grok (xAI) — implémente BaseLLMProvider via le SDK `openai`.

DEC-19 : l'API xAI est confirmée compatible avec le SDK OpenAI par la
documentation officielle (x.ai/api) — il suffit de pointer le client
`openai` standard vers base_url="https://api.x.ai/v1". Pas de SDK xAI
dédié ajouté pour éviter une 4ème dépendance SDK redondante.
"""

import openai

from app.providers.base_provider import (
    BaseLLMProvider,
    ProviderResponse,
    ProviderRateLimitError,
    ProviderAPIError,
)


class GrokProvider(BaseLLMProvider):
    """
    Adapter pour l'API Grok (xAI), via le client `openai` reciblé sur
    l'endpoint xAI compatible OpenAI (grok-4.1-fast et autres modèles Grok).

    Comme Mistral, Grok utilise le format de messages role="system" /
    role="user" plutôt qu'un paramètre `system` séparé.
    """

    def __init__(self, api_key: str):
        # Même client `openai.OpenAI`, simplement reciblé via base_url —
        # voir DEC-19 dans DECISIONS-PLANNIFICATEUR.md.
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

    def call(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int,
    ) -> ProviderResponse:
        """Effectue un appel synchrone à l'API Grok (xAI) et normalise la réponse."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return ProviderResponse(
                text=response.choices[0].message.content,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        except openai.RateLimitError as e:
            raise ProviderRateLimitError(str(e)) from e
        except openai.APIError as e:
            raise ProviderAPIError(str(e)) from e