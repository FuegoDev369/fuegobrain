"""
app/providers/anthropic_provider.py
Adapter Anthropic — implémente BaseLLMProvider pour le SDK `anthropic`.

Ce fichier EXTRAIT la logique d'appel précédemment inline dans
BaseAgent._call_anthropic_with_retry() (TICKET-05 du plan original) vers
un adapter dédié. C'est un refactor par extraction, pas une réécriture :
le comportement observable (format de réponse, exceptions levées) reste
identique à avant.
"""

import anthropic

from app.providers.base_provider import (
    BaseLLMProvider,
    ProviderResponse,
    ProviderRateLimitError,
    ProviderAPIError,
)


class AnthropicProvider(BaseLLMProvider):
    """
    Adapter pour l'API Anthropic (claude-sonnet-4-6 et autres modèles Claude).

    Le retry (3 tentatives, backoff linéaire) N'EST PAS géré ici — il reste
    centralisé dans BaseAgent._call_provider_with_retry() (TICKET-27/29),
    au-dessus de la couche provider. Cet adapter ne fait qu'UN appel et
    traduit les exceptions natives du SDK Anthropic en exceptions
    génériques ProviderRateLimitError / ProviderAPIError.
    """

    def __init__(self, api_key: str):
        # Le client est instancié une fois, réutilisé pour tous les appels
        # de cette instance — cohérent avec le pattern d'instanciation
        # unique des agents déjà en place dans l'Orchestrator.
        self.client = anthropic.Anthropic(api_key=api_key)

    def call(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int,
    ) -> ProviderResponse:
        """Effectue un appel synchrone à l'API Anthropic et normalise la réponse."""
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return ProviderResponse(
                text=response.content[0].text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        except anthropic.RateLimitError as e:
            raise ProviderRateLimitError(str(e)) from e
        except anthropic.APIError as e:
            raise ProviderAPIError(str(e)) from e