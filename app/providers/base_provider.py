"""
app/providers/base_provider.py
Interface commune à tous les providers LLM (Anthropic, Mistral, Gemini, Grok).
Définit le format de réponse normalisé et le contrat que chaque adapter
concret doit implémenter, pour que BaseAgent reste provider-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderResponse:
    """
    Format de réponse NORMALISÉ, commun aux 4 providers.

    C'est la pièce centrale qui permet à BaseAgent de ne jamais avoir à
    connaître le format natif d'un SDK spécifique (response.content[0].text
    pour Anthropic, response.choices[0].message.content pour Mistral/Grok,
    response.text pour Gemini, etc.) — chaque adapter traduit son format
    natif vers cette forme unique avant de retourner.
    """

    text: str  # Contenu textuel de la réponse, quel que soit le provider
    input_tokens: int  # Tokens d'entrée consommés (normalisé depuis le SDK natif)
    output_tokens: int  # Tokens de sortie produits (normalisé depuis le SDK natif)


class ProviderRateLimitError(Exception):
    """
    Exception générique levée par n'importe quel provider en cas de 429 /
    rate limit. Permet à BaseAgent._call_provider_with_retry() de rester
    provider-agnostic dans sa logique de retry, sans connaître les
    exceptions natives de chaque SDK (anthropic.RateLimitError,
    mistralai.models.SDKError, etc.)
    """


class ProviderAPIError(Exception):
    """
    Exception générique pour toute autre erreur API (timeout, 500, clé
    invalide...). Chaque adapter concret catch l'exception native de son
    SDK et la relève sous cette forme générique.
    """


class BaseLLMProvider(ABC):
    """
    Contrat commun que chaque provider concret (AnthropicProvider,
    MistralProvider, GeminiProvider, GrokProvider) doit implémenter.
    """

    @abstractmethod
    def call(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int,
    ) -> ProviderResponse:
        """
        Effectue UN appel synchrone au LLM sous-jacent et retourne une
        ProviderResponse normalisée.

        Cette méthode reste SYNCHRONE (pas async) — c'est BaseAgent qui
        l'enveloppe dans asyncio.to_thread(), exactement comme pour l'appel
        Anthropic d'origine (voir DEC-03 du plan original). Chaque provider
        n'a donc pas à se soucier de l'async — un seul point d'enveloppe.

        Args:
            system_prompt: le system prompt de l'agent (RESEARCHER_SYSTEM_PROMPT, etc.)
            user_message: le message construit par build_user_message()
            model: l'identifiant exact du modèle pour CE provider
                   (ex: "gemini-2.5-flash", "mistral-small-latest", "grok-4.1-fast")
            max_tokens: la limite de tokens de sortie pour cet agent

        Returns:
            ProviderResponse avec le texte et les compteurs de tokens normalisés

        Raises:
            ProviderRateLimitError: équivalent générique de 429 (rate limit)
            ProviderAPIError: toute autre erreur API du provider
        """
        raise NotImplementedError
