"""
app/providers/mistral_provider.py
Adapter Mistral — implémente BaseLLMProvider pour le SDK `mistralai`.

Suit le même pattern que AnthropicProvider (TICKET-25) : un seul appel
synchrone, traduction des exceptions natives du SDK vers les exceptions
génériques ProviderRateLimitError / ProviderAPIError, retour d'une
ProviderResponse normalisée. Aucun retry ici — il reste centralisé dans
BaseAgent._call_provider_with_retry() (TICKET-27/29).

NOTE D'IMPLÉMENTATION (écart corrigé par rapport au plan d'origine) :
le plan spécifiait `from mistralai import Mistral` et
`from mistralai.models import SDKError`. Vérifié par exécution réelle
(installation du package depuis PyPI) que la version actuellement publiée
(mistralai 2.5.0, satisfait par la contrainte `mistralai>=1.0.0` du plan)
a réorganisé son arborescence : le package racine est devenu un namespace
package sans `__init__.py`, et la classe `Mistral` ainsi que `SDKError`
ont été déplacées sous `mistralai.client`. Les imports ci-dessous reflètent
la structure réelle du SDK installé — voir VALIDATION-NOTES.md (TICKET-26)
pour le détail complet de cette vérification.
"""

from mistralai.client import Mistral
from mistralai.client.errors import SDKError

from app.providers.base_provider import (
    BaseLLMProvider,
    ProviderResponse,
    ProviderRateLimitError,
    ProviderAPIError,
)


class MistralProvider(BaseLLMProvider):
    """
    Adapter pour l'API Mistral (mistral-small-latest et autres modèles Mistral).

    Différence notable avec Anthropic : Mistral n'a pas de paramètre `system`
    séparé — le system prompt est passé comme un message à part entière avec
    role="system" dans la même liste `messages` que le message user.
    """

    def __init__(self, api_key: str):
        # Le client est instancié une fois, réutilisé pour tous les appels
        # de cette instance — cohérent avec AnthropicProvider (TICKET-25).
        self.client = Mistral(api_key=api_key)

    def call(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int,
    ) -> ProviderResponse:
        """Effectue un appel synchrone à l'API Mistral et normalise la réponse."""
        try:
            response = self.client.chat.complete(
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
        except SDKError as e:
            # Le SDK Mistral lève SDKError aussi bien pour les 429 que pour
            # les autres erreurs API — la distinction se fait par status_code,
            # pas par un type d'exception dédié comme chez Anthropic.
            if getattr(e, "status_code", None) == 429:
                raise ProviderRateLimitError(str(e)) from e
            raise ProviderAPIError(str(e)) from e