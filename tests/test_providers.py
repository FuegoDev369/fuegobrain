"""
tests/test_providers.py
Unit tests for the multi-provider adapter layer (app/providers/).

These tests are fully mocked — no real network calls are made to any of
the 4 LLM providers (Anthropic, Mistral, Gemini, Grok), and no real API
key is required. They exercise:
  - PROVIDER_REGISTRY contents and get_provider() factory behaviour
    (correct class resolution, case-insensitivity, invalid-name ValueError)
  - Each adapter's call() happy path: native SDK response → normalized
    ProviderResponse (text / input_tokens / output_tokens)
  - Each adapter's exception translation: native SDK exceptions →
    ProviderRateLimitError (429) and ProviderAPIError (anything else)

Native SDK exceptions are constructed with REAL exception classes backed
by real httpx.Request/httpx.Response objects (or, for google-genai, the
plain (code, response_json) constructor its own APIError expects) —
not bare MagicMock() instances standing in for exceptions. This mirrors
the approach already validated for MistralProvider during TICKET-26 (see
VALIDATION-NOTES.md): constructing exceptions the way the real SDK would
raise them is what actually confirms the `except anthropic.RateLimitError`
/ `except SDKError` / etc. clauses in each adapter really match, rather
than merely asserting that *some* exception propagates.

Import note — MistralProvider (app/providers/mistral_provider.py) imports
`Mistral`/`SDKError` from `mistralai.client` / `mistralai.client.errors`,
not from the package roots as the original plan specified. This was a
deliberate correction made and documented during TICKET-26 (real `pip
install mistralai` revealed the 2.x package restructured its layout) —
this test file imports from the same corrected paths, confirmed again by
real execution here (mistralai 2.5.1 installed in this sandbox).
"""

# stdlib
from unittest.mock import MagicMock, patch

# third-party
import anthropic
import httpx
import openai
import pytest
from google.genai.errors import APIError as GeminiAPIError
from mistralai.client.errors import SDKError as MistralSDKError

# local
from app.providers import PROVIDER_REGISTRY, get_provider
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base_provider import (
    ProviderAPIError,
    ProviderRateLimitError,
    ProviderResponse,
)
from app.providers.gemini_provider import GeminiProvider
from app.providers.grok_provider import GrokProvider
from app.providers.mistral_provider import MistralProvider


# ── Test helpers ────────────────────────────────────────────────────────────


def _httpx_response(status_code: int, url: str) -> httpx.Response:
    """
    Build a real httpx.Response (backed by a real httpx.Request), used to
    construct real anthropic/openai/mistralai exception instances the same
    way their SDKs do internally (MistralError.__init__ and friends read
    `.status_code` straight off this response object — a MagicMock stand-in
    would not reproduce that behaviour faithfully).
    """
    return httpx.Response(status_code=status_code, request=httpx.Request("POST", url))


def _mock_target(provider_name: str, provider: object) -> tuple[object, str]:
    """
    Return (owner_object, attribute_name) identifying the exact SDK method
    each adapter's call() invokes — used to patch precisely that call site
    with unittest.mock.patch.object() rather than replacing the whole
    provider.client (which would also hide bugs in how each adapter reaches
    into its client).
    """
    if provider_name == "anthropic":
        return provider.client.messages, "create"
    if provider_name == "mistral":
        return provider.client.chat, "complete"
    if provider_name == "gemini":
        return provider.client.models, "generate_content"
    if provider_name == "grok":
        return provider.client.chat.completions, "create"
    raise ValueError(f"no mock target defined for provider '{provider_name}'")


# Native rate-limit (429) exception factories — one per provider, each using
# the real exception class the provider's SDK actually raises.
def _anthropic_rate_limit_error() -> Exception:
    return anthropic.RateLimitError(
        "rate limit exceeded",
        response=_httpx_response(429, "https://api.anthropic.com/v1/messages"),
        body=None,
    )


def _mistral_rate_limit_error() -> Exception:
    return MistralSDKError(
        "rate limit exceeded",
        raw_response=_httpx_response(429, "https://api.mistral.ai/v1/chat/completions"),
    )


def _gemini_rate_limit_error() -> Exception:
    return GeminiAPIError(code=429, response_json={"error": {"status": "RESOURCE_EXHAUSTED"}})


def _grok_rate_limit_error() -> Exception:
    return openai.RateLimitError(
        "rate limit exceeded",
        response=_httpx_response(429, "https://api.x.ai/v1/chat/completions"),
        body=None,
    )


# Native non-rate-limit (500) exception factories — bonus coverage beyond
# the plan's strict scope (see test_each_provider_raises_generic_api_error
# below), extending to all 4 providers the pattern already validated for
# Mistral alone during TICKET-26.
def _anthropic_api_error() -> Exception:
    return anthropic.APIStatusError(
        "internal server error",
        response=_httpx_response(500, "https://api.anthropic.com/v1/messages"),
        body=None,
    )


def _mistral_api_error() -> Exception:
    return MistralSDKError(
        "internal server error",
        raw_response=_httpx_response(500, "https://api.mistral.ai/v1/chat/completions"),
    )


def _gemini_api_error() -> Exception:
    return GeminiAPIError(code=500, response_json={"error": {"status": "INTERNAL"}})


def _grok_api_error() -> Exception:
    return openai.APIStatusError(
        "internal server error",
        response=_httpx_response(500, "https://api.x.ai/v1/chat/completions"),
        body=None,
    )


# ── PROVIDER_REGISTRY / get_provider() ──────────────────────────────────────


def test_provider_registry_has_four_providers():
    assert len(PROVIDER_REGISTRY) == 4
    assert set(PROVIDER_REGISTRY.keys()) == {"anthropic", "mistral", "gemini", "grok"}


def test_get_provider_returns_correct_class():
    assert isinstance(get_provider("gemini", "key"), GeminiProvider)
    assert isinstance(get_provider("MISTRAL", "key"), MistralProvider)  # case-insensitive
    assert isinstance(get_provider("Anthropic", "key"), AnthropicProvider)
    assert isinstance(get_provider("grok", "key"), GrokProvider)


def test_get_provider_invalid_name_raises():
    with pytest.raises(ValueError, match="Unknown provider 'not-a-provider'"):
        get_provider("not-a-provider", "key")


# ── Happy-path call() — native SDK response → normalized ProviderResponse ──


def test_anthropic_provider_call_mocked():
    provider = AnthropicProvider(api_key="test-key")

    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="answer")]
    fake_response.usage.input_tokens = 10
    fake_response.usage.output_tokens = 20

    with patch.object(provider.client.messages, "create", return_value=fake_response) as mock_create:
        result = provider.call("system prompt", "user msg", "claude-sonnet-4-6", 100)

    mock_create.assert_called_once_with(
        model="claude-sonnet-4-6",
        max_tokens=100,
        system="system prompt",
        messages=[{"role": "user", "content": "user msg"}],
    )
    assert isinstance(result, ProviderResponse)
    assert result.text == "answer"
    assert result.input_tokens == 10
    assert result.output_tokens == 20


def test_mistral_provider_call_mocked():
    provider = MistralProvider(api_key="test-key")

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="answer"))]
    fake_response.usage.prompt_tokens = 10
    fake_response.usage.completion_tokens = 20

    with patch.object(provider.client.chat, "complete", return_value=fake_response) as mock_complete:
        result = provider.call("system prompt", "user msg", "mistral-small-latest", 100)

    mock_complete.assert_called_once_with(
        model="mistral-small-latest",
        max_tokens=100,
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user msg"},
        ],
    )
    assert isinstance(result, ProviderResponse)
    assert result.text == "answer"
    assert result.input_tokens == 10
    assert result.output_tokens == 20


def test_gemini_provider_call_mocked():
    provider = GeminiProvider(api_key="test-key")

    fake_response = MagicMock()
    fake_response.text = "answer"
    fake_response.usage_metadata.prompt_token_count = 10
    fake_response.usage_metadata.candidates_token_count = 20

    with patch.object(
        provider.client.models, "generate_content", return_value=fake_response
    ) as mock_generate:
        result = provider.call("system prompt", "user msg", "gemini-2.5-flash", 100)

    # Gemini's config is a types.GenerateContentConfig object, not a plain
    # dict — assert on its fields individually rather than on the raw
    # call args, since two GenerateContentConfig instances with identical
    # field values are not guaranteed to be __eq__-comparable.
    _, kwargs = mock_generate.call_args
    assert kwargs["model"] == "gemini-2.5-flash"
    assert kwargs["contents"] == "user msg"
    assert kwargs["config"].system_instruction == "system prompt"
    assert kwargs["config"].max_output_tokens == 100

    assert isinstance(result, ProviderResponse)
    assert result.text == "answer"
    assert result.input_tokens == 10
    assert result.output_tokens == 20


def test_grok_provider_call_mocked():
    provider = GrokProvider(api_key="test-key")

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="answer"))]
    fake_response.usage.prompt_tokens = 10
    fake_response.usage.completion_tokens = 20

    with patch.object(
        provider.client.chat.completions, "create", return_value=fake_response
    ) as mock_create:
        result = provider.call("system prompt", "user msg", "grok-4.1-fast", 100)

    mock_create.assert_called_once_with(
        model="grok-4.1-fast",
        max_tokens=100,
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user msg"},
        ],
    )
    assert isinstance(result, ProviderResponse)
    assert result.text == "answer"
    assert result.input_tokens == 10
    assert result.output_tokens == 20


# ── Exception translation — native SDK exception → generic Provider*Error ──


@pytest.mark.parametrize(
    "provider_name,provider_class,native_error_factory",
    [
        ("anthropic", AnthropicProvider, _anthropic_rate_limit_error),
        ("mistral", MistralProvider, _mistral_rate_limit_error),
        ("gemini", GeminiProvider, _gemini_rate_limit_error),
        ("grok", GrokProvider, _grok_rate_limit_error),
    ],
)
def test_each_provider_raises_generic_rate_limit_error(
    provider_name, provider_class, native_error_factory
):
    """
    Every adapter must translate its SDK-native 429 exception into the
    generic ProviderRateLimitError (TICKET-24) — this is what lets
    BaseAgent._call_provider_with_retry() (TICKET-29) retry on rate limits
    without knowing which of the 4 providers is actually configured.
    """
    provider = provider_class(api_key="test-key")
    target_obj, attr_name = _mock_target(provider_name, provider)

    with patch.object(target_obj, attr_name, side_effect=native_error_factory()):
        with pytest.raises(ProviderRateLimitError):
            provider.call("system", "user msg", "some-model", 10)


@pytest.mark.parametrize(
    "provider_name,provider_class,native_error_factory",
    [
        ("anthropic", AnthropicProvider, _anthropic_api_error),
        ("mistral", MistralProvider, _mistral_api_error),
        ("gemini", GeminiProvider, _gemini_api_error),
        ("grok", GrokProvider, _grok_api_error),
    ],
)
def test_each_provider_raises_generic_api_error_on_non_rate_limit(
    provider_name, provider_class, native_error_factory
):
    """
    Bonus regression coverage, added beyond TICKET-30's strict scope —
    mirrors the two-branch verification already done for Mistral alone
    during TICKET-26 (see VALIDATION-NOTES.md), extended here to all 4
    providers: a non-429 failure must translate to ProviderAPIError, never
    ProviderRateLimitError, so BaseAgent never wastes a retry on an error
    retrying cannot fix (auth failure, 500, etc.).
    """
    provider = provider_class(api_key="test-key")
    target_obj, attr_name = _mock_target(provider_name, provider)

    with patch.object(target_obj, attr_name, side_effect=native_error_factory()):
        with pytest.raises(ProviderAPIError):
            provider.call("system", "user msg", "some-model", 10)
