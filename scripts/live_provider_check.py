"""
scripts/live_provider_check.py
Script de vérification manuelle — envoie UN appel minimal réel à chaque
provider configuré (Anthropic, Mistral, Gemini, Grok) et rapporte
succès/échec. Conçu pour être lancé via le workflow GitHub Actions
`providers-live-check.yml` (workflow_dispatch UNIQUEMENT — jamais sur
push/PR, voir DEC-21), ou manuellement en local.

Objectif strict : confirmer qu'une clé fonctionne et que le format de
réponse réel de chaque SDK/API correspond toujours à ce qu'attend
l'adapter correspondant (app/providers/*.py) — PAS évaluer la qualité
du modèle. Un seul appel, prompt minimal, max_tokens réduit.

Usage (⚠️ toujours via `python -m`, jamais `python scripts/live_provider_check.py`
directement — voir la note d'invocation dans providers-live-check.yml et
VALIDATION-NOTES.md pour le détail du mécanisme sys.path) :
  PROVIDERS_TO_TEST=all python -m scripts.live_provider_check
  PROVIDERS_TO_TEST=gemini,mistral python -m scripts.live_provider_check
"""

import os
import sys

from app.providers import PROVIDER_REGISTRY, get_provider

# Modèle minimal et bon marché par provider — un seul appel, prompt court,
# max_tokens réduit. Objectif : confirmer que la clé fonctionne et que le
# format de réponse correspond toujours à ce qu'attend l'adapter, PAS
# tester en profondeur la qualité du modèle (voir DÉCISION PLANNIFICATEUR
# du TICKET-32 : modèles "rapides/économiques" choisis intentionnellement,
# pas de Pro/Opus plus cher).
SMOKE_TEST_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "mistral": "mistral-small-latest",
    "gemini": "gemini-2.5-flash",
    "grok": "grok-4.1-fast",
}

KEY_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "grok": "GROK_API_KEY",
}


def check_provider(name: str) -> tuple[bool, str]:
    """
    Effectue UN appel réel minimal au provider `name` et retourne
    (ok, message). N'appelle jamais l'API deux fois pour un même provider
    — le message retourné ici est réutilisé tel quel partout ailleurs
    dans ce script (voir note dans main()).
    """
    api_key = os.environ.get(KEY_ENV_VARS[name])
    if not api_key:
        return False, f"SKIPPED — {KEY_ENV_VARS[name]} not set"

    try:
        provider = get_provider(name, api_key)
        response = provider.call(
            system_prompt="Reply with exactly one word: OK",
            user_message="Confirm you are working.",
            model=SMOKE_TEST_MODELS[name],
            max_tokens=10,
        )
        if response.text and len(response.text.strip()) > 0:
            return True, f"OK — got response: {response.text.strip()[:30]!r}"
        return False, "FAILED — empty response text"
    except Exception as e:  # noqa: BLE001 — smoke test volontairement générique
        return False, f"FAILED — {type(e).__name__}: {e}"


def main() -> None:
    requested = os.environ.get("PROVIDERS_TO_TEST", "all").strip().lower()
    targets = (
        list(PROVIDER_REGISTRY.keys())
        if requested in ("all", "")
        else [p.strip() for p in requested.split(",")]
    )

    print(f"Live provider check — testing: {', '.join(targets)}\n")

    # results stocke (ok: bool, message: str) — UN SEUL appel par provider,
    # jamais recalculé ensuite. Important : un précédent brouillon de ce
    # script ré-appelait check_provider() dans la logique de sortie pour
    # distinguer SKIPPED de FAILED, ce qui aurait doublé les appels API
    # réels (coût et consommation de quota inutiles). Corrigé : on garde
    # le message complet dès le premier appel et on le réutilise partout.
    results: dict[str, tuple[bool, str]] = {}
    for name in targets:
        if name not in PROVIDER_REGISTRY:
            print(f"  {name:12} ⚠️  unknown provider, skipping")
            continue
        ok, message = check_provider(name)
        results[name] = (ok, message)
        icon = "✅" if ok else ("⏭️ " if "SKIPPED" in message else "❌")
        print(f"  {name:12} {icon} {message}")

    print()
    hard_failures = [n for n, (ok, msg) in results.items() if not ok and "SKIPPED" not in msg]
    skipped = [n for n, (ok, msg) in results.items() if "SKIPPED" in msg]

    if skipped:
        print(f"Skipped (no Secret configured): {', '.join(skipped)}")
    if hard_failures:
        print(f"Real failures: {', '.join(hard_failures)}")
        sys.exit(1)  # Échec UNIQUEMENT si un provider configuré a réellement échoué
    else:
        print("All configured providers responded successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
