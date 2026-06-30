"""
app/config.py
Centralized configuration for FuegoBrain.
Reads all environment variables via pydantic-settings BaseSettings.
Exposes a cached singleton via get_settings().
"""

# stdlib
from functools import lru_cache
from typing import Union

# third-party
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings — loaded once from .env at startup.
    All fields are typed and validated by Pydantic v2.
    A missing ANTHROPIC_API_KEY raises a clear error at boot, not at runtime.
    """

    # ── Anthropic ─────────────────────────────────────────────────────────
    # No default — mandatory. Pydantic raises ValidationError at startup if absent.
    anthropic_api_key: str = Field(..., description="Anthropic API key (required)")
    anthropic_model: str = Field(
        default="claude-sonnet-4-6",
        description="Anthropic model identifier — do not change in v1",
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
        description="Per-agent Anthropic call timeout in seconds",
    )
    rate_limit_retry_delay: float = Field(
        default=1.0,
        description="Base delay (seconds) for linear backoff on 429 — retries at 1x, 2x, 3x",
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.
    The @lru_cache ensures .env is read exactly once per process,
    regardless of how many times get_settings() is called.

    Usage:
        from app.config import get_settings
        settings = get_settings()
        print(settings.anthropic_model)
    """
    return Settings()