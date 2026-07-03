"""
app/config.py
Centralized configuration for FuegoBrain.
Reads all environment variables via pydantic-settings BaseSettings.
Exposes a cached singleton via get_settings().

Multi-provider extension (TICKET-28): each agent (researcher/reasoner/
synthesizer) resolves its own provider + model + API key at runtime via
get_agent_config(). All 4 provider API keys are optional — only the key
for the provider actually configured on an agent needs to be set (DEC-18).
"""

# stdlib
from functools import lru_cache
from typing import Optional, Union

# third-party
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings — loaded once from .env at startup.
    All fields are typed and validated by Pydantic v2.

    Note on API keys (DEC-18): none of the 4 provider API keys are mandatory
    at this layer anymore. With 4 possible providers, requiring all 4 keys
    at boot would break the "free out-of-the-box" goal (DEC-17 — Gemini is
    the default). Validation moves from static (Settings() instantiation)
    to dynamic (per-agent, via get_agent_config(), raised when the agent
    that needs a missing key is actually instantiated — see
    app/orchestrator/agents/base_agent.py, TICKET-29).
    """

    # ── API keys per provider ────────────────────────────────────────────
    # All optional — only the key for the provider(s) configured below
    # (researcher_provider / reasoner_provider / synthesizer_provider) must
    # actually be set for the app to run.
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key (required only if an agent uses 'anthropic')"
    )
    mistral_api_key: Optional[str] = Field(
        default=None, description="Mistral API key (required only if an agent uses 'mistral')"
    )
    gemini_api_key: Optional[str] = Field(
        default=None, description="Gemini API key (required only if an agent uses 'gemini')"
    )
    grok_api_key: Optional[str] = Field(
        default=None, description="Grok/xAI API key (required only if an agent uses 'grok')"
    )

    # ── Provider + model per agent ───────────────────────────────────────
    # Each agent has its own provider and model, independent of the others.
    # Default: Gemini Flash everywhere — free tier, no credit card required
    # (DEC-17 — chosen over Mistral for its more generous free-tier RPM,
    # better suited to a 3-sequential-call pipeline).
    researcher_provider: str = Field(
        default="gemini", description="Provider used by the Researcher agent"
    )
    researcher_model: str = Field(
        default="gemini-2.5-flash", description="Model used by the Researcher agent"
    )

    reasoner_provider: str = Field(
        default="gemini", description="Provider used by the Reasoner agent"
    )
    reasoner_model: str = Field(
        default="gemini-2.5-flash", description="Model used by the Reasoner agent"
    )

    synthesizer_provider: str = Field(
        default="gemini", description="Provider used by the Synthesizer agent"
    )
    synthesizer_model: str = Field(
        default="gemini-2.5-flash", description="Model used by the Synthesizer agent"
    )

    # ── Per-agent token limits ─────────────────────────────────────────────
    max_tokens_researcher: int = Field(
        default=800,
        description="Max output tokens for the Researcher agent",
    )
    max_tokens_reasoner: int = Field(
        default=1000,
        description="Max output tokens for the Reasoner agent",
    )
    max_tokens_synthesizer: int = Field(
        default=1500,
        description="Max output tokens for the Synthesizer agent",
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    # Stored as list[str]. pydantic-settings may receive a raw CSV string from
    # the environment — the validator below handles both cases transparently.
    demo_cors_origins: Union[list[str], str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins — comma-separated in .env",
    )

    # ── Timeouts & retry ──────────────────────────────────────────────────
    agent_timeout_seconds: int = Field(
        default=30,
        description="Per-agent provider call timeout in seconds",
    )
    rate_limit_retry_delay: float = Field(
        default=1.0,
        description="Base delay (seconds) for linear backoff on rate limit — retries at 1x, 2x, 3x",
    )

    # ── pydantic-settings model config ────────────────────────────────────
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        # Allow extra env vars in .env without raising errors
        "extra": "ignore",
    }

    # ── Validators ────────────────────────────────────────────────────────

    @field_validator("demo_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Union[str, list]) -> list[str]:
        """
        Parse DEMO_CORS_ORIGINS from either a CSV string or an already-parsed list.
        Handles both:
          - .env: DEMO_CORS_ORIGINS=http://localhost:3000,http://localhost:8000
          - Programmatic: demo_cors_origins=["http://localhost:3000"]
        """
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def cors_origins_list(self) -> list[str]:
        """
        Convenience property — always returns a clean list[str].
        Use this in main.py rather than accessing demo_cors_origins directly.
        """
        origins = self.demo_cors_origins
        if isinstance(origins, str):
            return [o.strip() for o in origins.split(",") if o.strip()]
        return origins

    # ── Multi-provider resolution ────────────────────────────────────────

    def get_agent_config(self, agent_name: str) -> tuple[str, str, str]:
        """
        Resolve (provider_name, model, api_key) for a given agent.

        Args:
            agent_name: "researcher" | "reasoner" | "synthesizer"

        Returns:
            Tuple (provider, model, api_key) ready to pass to
            app.providers.get_provider().

        Raises:
            ValueError: if the API key matching the provider configured for
                        this agent is absent (None). This is where the
                        "key presence" validation actually happens now
                        (DEC-18) — not at Settings() load time.
        """
        provider = getattr(self, f"{agent_name}_provider")
        model = getattr(self, f"{agent_name}_model")
        api_key = getattr(self, f"{provider}_api_key")
        if api_key is None:
            raise ValueError(
                f"Agent '{agent_name}' is configured to use provider '{provider}' "
                f"but {provider.upper()}_API_KEY is not set in .env"
            )
        return provider, model, api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.
    The @lru_cache ensures .env is read exactly once per process,
    regardless of how many times get_settings() is called.

    Usage:
        from app.config import get_settings
        settings = get_settings()
        provider, model, api_key = settings.get_agent_config("researcher")
    """
    return Settings()