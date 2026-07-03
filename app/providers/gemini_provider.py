"""
app/providers/gemini_provider.py
Adapter Gemini — implémente BaseLLMProvider pour le SDK `google-genai`.

Suit le même pattern que AnthropicProvider (TICKET-25). Gemini est le
provider PAR DÉFAUT du projet (voir DEC-17 dans DECISIONS-PLANNIFICATEUR.md)
— ce fichier est donc sur le chemin critique du démarrage out-of-the-box.
"""

from google import genai
from google.genai import types
from google.genai.errors import APIError as GeminiAPIError

from app.providers.base_provider import (
    BaseLLMProvider,
    ProviderResponse,
    ProviderRateLimitError,
    ProviderAPIError,
)


class GeminiProvider(BaseLLMProvider):
    """
    Adapter pour l'API Gemini (gemini-2.5-flash et autres modèles Gemini).

    Comme Anthropic, Gemini sépare le system prompt du message user via un
    paramètre dédié (`system_instruction`, dans `GenerateContentConfig`)
    plutôt qu'un message role="system" dans la liste des messages.
    """

    def __init__(self, api_key: str):
        # Le client est instancié une fois, réutilisé pour tous les appels
        # de cette instance — cohérent avec AnthropicProvider (TICKET-25).
        self.client = genai.Client(api_key=api_key)

    def call(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int,
    ) -> ProviderResponse:
        """Effectue un appel synchrone à l'API Gemini et normalise la réponse."""
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                ),
            )
            return ProviderResponse(
                text=response.text,
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count,
            )
        except GeminiAPIError as e:
            # Le SDK google-genai encode le code HTTP dans e.code.
            if getattr(e, "code", None) == 429:
                raise ProviderRateLimitError(str(e)) from e
            raise ProviderAPIError(str(e)) from e