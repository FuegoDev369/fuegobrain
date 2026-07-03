"""
app/providers/__init__.py
Registry de providers LLM — point d'entrée unique pour instancier le bon
adapter (AnthropicProvider, MistralProvider, GeminiProvider, GrokProvider)
à partir d'un nom de provider configuré par agent (.env).
"""

from app.providers.base_provider import (
    BaseLLMProvider,
    ProviderResponse,
    ProviderRateLimitError,
    ProviderAPIError,
)
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.mistral_provider import MistralProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.grok_provider import GrokProvider


# Mapping nom de provider (str, tel qu'utilisé dans .env) → classe concrète.
# Les clés sont en minuscules — get_provider() normalise l'input via .lower()
# pour rester insensible à la casse (ex: "Gemini" et "GEMINI" fonctionnent).
PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {
    "anthropic": AnthropicProvider,
    "mistral": MistralProvider,
    "gemini": GeminiProvider,
    "grok": GrokProvider,
}


def get_provider(provider_name: str, api_key: str) -> BaseLLMProvider:
    """
    Factory — instancie le bon adapter provider à partir de son nom et de
    sa clé API.

    Args:
        provider_name: nom du provider, insensible à la casse
                        ("anthropic" | "mistral" | "gemini" | "grok")
        api_key: clé API à passer au constructeur du provider concret

    Returns:
        Instance du provider concret (AnthropicProvider, MistralProvider,
        GeminiProvider ou GrokProvider), déjà initialisée avec sa clé API.

    Raises:
        ValueError: si provider_name ne correspond à aucune clé du registry.
                    Échec volontairement immédiat et explicite — un nom de
                    provider mal orthographié dans .env doit être détecté
                    au démarrage de l'application, pas silencieusement à
                    la première requête.
    """
    provider_class = PROVIDER_REGISTRY.get(provider_name.lower())
    if provider_class is None:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Valid options: {', '.join(PROVIDER_REGISTRY.keys())}"
        )
    return provider_class(api_key=api_key)


__all__ = [
    "PROVIDER_REGISTRY",
    "get_provider",
    "BaseLLMProvider",
    "ProviderResponse",
    "ProviderRateLimitError",
    "ProviderAPIError",
]