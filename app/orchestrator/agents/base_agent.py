"""
app/orchestrator/agents/base_agent.py
Abstract base class shared by all three FuegoBrain agents (Researcher, Reasoner, Synthesizer).

Multi-provider extension (TICKET-29): this class is now provider-agnostic.
It no longer instantiates the Anthropic SDK directly — instead it resolves,
via Settings.get_agent_config() (TICKET-28), which provider/model/api_key
this specific agent (AGENT_NAME) should use, then obtains the matching
BaseLLMProvider adapter via app.providers.get_provider() (TICKET-27).

Responsibilities:
  - Resolve this agent's (provider, model, api_key) once at construction
  - Execute the provider call inside asyncio.to_thread() to avoid blocking FastAPI
  - Retry on ProviderRateLimitError with linear backoff: 1×, 2×, 3× delay
  - Measure wall-clock duration and capture token usage (via ProviderResponse)
  - Build and return an AgentCallRecord for every completed call

Each concrete agent subclass must implement:
  - AGENT_NAME (str class attr)    — used in AgentCallRecord, pipeline_trace,
                                      AND to resolve this agent's provider config
  - SYSTEM_PROMPT (str class attr) — the agent's instruction prompt, stored in code
  - build_user_message()           — formats the user-role message from AgentContext
  - _get_max_tokens()              — returns the per-agent token ceiling from settings

The run() method is final — do not override it in subclasses.
"""

# stdlib
import asyncio
import time
from abc import ABC, abstractmethod

# local
from app.config import get_settings
from app.orchestrator.context import AgentCallRecord, AgentContext
from app.providers import (
    BaseLLMProvider,
    ProviderAPIError,
    ProviderRateLimitError,
    ProviderResponse,
    get_provider,
)


class BaseAgent(ABC):
    """
    Abstract base for ResearcherAgent, ReasonerAgent, and SynthesizerAgent.

    Lifecycle per pipeline run:
      1. Orchestrator calls agent.run(context)
      2. run() calls build_user_message(context)  ← defined by subclass
      3. run() calls _call_provider_with_retry(user_message)
      4. run() wraps result in AgentCallRecord and returns it

    The provider adapter is created once in __init__ and reused across calls —
    the Orchestrator singleton guarantees this is safe (single-threaded pipeline).
    Which concrete provider (Anthropic/Mistral/Gemini/Grok) backs self.provider
    depends entirely on this agent's *_PROVIDER setting in .env (DEC-17: Gemini
    is the default for all three agents).
    """

    # ── Abstract class attributes (must be defined in each subclass) ───────────

    @property
    @abstractmethod
    def AGENT_NAME(self) -> str:
        """Identifier used in AgentCallRecord.agent_name and pipeline_trace."""
        ...

    @property
    @abstractmethod
    def SYSTEM_PROMPT(self) -> str:
        """
        The agent's full system prompt.
        Stored in the subclass source code — intentionally visible and versionable.
        Never stored in .env or config.
        """
        ...

    # ── Initialisation ─────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self.settings = get_settings()

        # Resolve this agent's provider/model/api_key from Settings (TICKET-28).
        # self.AGENT_NAME is a @property returning a plain string constant
        # (e.g. "researcher") — it is defined at the class level via the
        # abstract property override in each subclass, so it is already
        # accessible here even though __init__ has not finished running yet
        # (no instance state is needed to resolve it).
        # Raises ValueError if the API key for the configured provider is
        # missing — this is the "dynamic validation" moment described in
        # DEC-18: no longer caught at Settings() load time, but here, at
        # this specific agent's instantiation.
        provider_name, self.model, api_key = self.settings.get_agent_config(
            self.AGENT_NAME
        )
        self.provider_name: str = provider_name  # NEW (TICKET-34) — stored so
                                                    # AgentCallRecord can carry it;
                                                    # previously resolved locally
                                                    # then discarded (the gap fixed
                                                    # by this ticket)

        # One provider adapter instance per agent instance — reused across
        # pipeline runs, exactly like the previous Anthropic-only client.
        self.provider: BaseLLMProvider = get_provider(provider_name, api_key)

        # max_tokens is resolved once at construction from settings.
        self.max_tokens: int = self._get_max_tokens()

    # ── Abstract methods (implemented by each concrete agent) ──────────────────

    @abstractmethod
    def build_user_message(self, context: AgentContext) -> str:
        """
        Construct the user-role message sent to the configured LLM provider.
        The system prompt is passed separately (as `system_prompt` to
        provider.call()) — it is never included here.

        Each agent formats its input differently:
          - ResearcherAgent : only the original query
          - ReasonerAgent   : query + researcher_output
          - SynthesizerAgent: query + researcher_output + reasoner_output + confidence
        """
        ...

    @abstractmethod
    def _get_max_tokens(self) -> int:
        """
        Return this agent's maximum output token count from settings.
        Called once in __init__ — avoids repeated settings lookups per call.
        """
        ...

    # ── Main entry point (do not override in subclasses) ──────────────────────

    async def run(self, context: AgentContext) -> AgentCallRecord:
        """
        Execute this agent against the provided context.
        This method is the single entry point called by the Orchestrator.

        Steps:
          1. Build the user message via build_user_message() (may raise AssertionError
             if the required context fields are not yet populated — that's a bug in
             the Orchestrator's execution order, not a user-facing error)
          2. Call the configured provider with linear-backoff retry on rate limits
          3. Wrap the result in an AgentCallRecord with timing and token data

        Returns:
          AgentCallRecord — consumed by the Orchestrator, eventually serialised into
          OrchestrateResponse.pipeline_trace by response_builder.py

        Raises:
          AssertionError            — precondition violation (Orchestrator bug)
          ProviderRateLimitError    — all retries exhausted (caught in main.py → HTTP 429)
          ProviderAPIError          — other provider errors (caught in main.py → HTTP 503)
        """
        user_message: str = self.build_user_message(context)

        start_time = time.time()
        response: ProviderResponse = await self._call_provider_with_retry(user_message)
        duration_ms: int = int((time.time() - start_time) * 1000)

        return AgentCallRecord(
            agent_name=self.AGENT_NAME,
            provider=self.provider_name,   # NEW (TICKET-34)
            model=self.model,               # NEW (TICKET-34)
            prompt_sent=user_message,
            response_text=response.text,
            duration_ms=duration_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    # ── Internal: provider call with linear-backoff retry ──────────────────────

    async def _call_provider_with_retry(self, user_message: str) -> ProviderResponse:
        """
        Renamed from _call_anthropic_with_retry (TICKET-29) — the retry logic
        itself is UNCHANGED (3 attempts, linear backoff 1×/2×/3× — DEC-08).
        Only the source of the exceptions changed: this method now catches
        the generic ProviderRateLimitError / ProviderAPIError raised by
        whichever adapter self.provider happens to be (Anthropic, Mistral,
        Gemini, or Grok), instead of the Anthropic SDK's native exceptions.

        Backoff strategy (DEC-08 — linear, not exponential):
          Attempt 1 → sleep 1× delay  → Attempt 2 → sleep 2× delay → Attempt 3 → raise

        self.provider.call() is synchronous by contract (BaseLLMProvider) — it
        is wrapped in asyncio.to_thread() to avoid blocking FastAPI's event
        loop, exactly as the Anthropic-only call was wrapped before (DEC-03).
        This wrapping now applies generically to any provider, not just
        Anthropic.
        """
        max_retries = 3
        delay = self.settings.rate_limit_retry_delay  # default 1.0s

        for attempt in range(max_retries):
            try:
                # asyncio.to_thread() runs the synchronous provider call in a
                # thread pool, freeing the event loop to handle other
                # requests concurrently.
                response: ProviderResponse = await asyncio.to_thread(
                    self.provider.call,
                    self.SYSTEM_PROMPT,
                    user_message,
                    self.model,
                    self.max_tokens,
                )
                return response

            except ProviderRateLimitError:
                if attempt == max_retries - 1:
                    # All retries exhausted — propagate to Orchestrator → main.py → HTTP 429
                    raise
                # Linear backoff: 1×, 2× — readable and predictable in demo conditions
                backoff_seconds = delay * (attempt + 1)
                await asyncio.sleep(backoff_seconds)

            except ProviderAPIError:
                # All other provider errors (timeout, auth, server errors, etc.)
                # — propagate immediately, no retry.
                raise

        # Unreachable — the loop always returns or raises before exhaustion,
        # but satisfies type checkers expecting a return on all paths.
        raise RuntimeError(f"[{self.AGENT_NAME}] _call_provider_with_retry: unexpected exit")
