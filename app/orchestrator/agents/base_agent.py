"""
app/orchestrator/agents/base_agent.py
Abstract base class shared by all three FuegoBrain agents (Researcher, Reasoner, Synthesizer).

Responsibilities:
  - Instantiate and hold the Anthropic SDK client (once per agent instance)
  - Execute the Anthropic API call inside asyncio.to_thread() to avoid blocking FastAPI
  - Retry on RateLimitError (429) with linear backoff: 1×, 2×, 3× delay
  - Measure wall-clock duration and capture token usage
  - Build and return an AgentCallRecord for every completed call

Each concrete agent subclass must implement:
  - AGENT_NAME (str class attr)    — used in AgentCallRecord and pipeline_trace
  - SYSTEM_PROMPT (str class attr) — the agent's instruction prompt, stored in code
  - build_user_message()           — formats the user-role message from AgentContext
  - _get_max_tokens()              — returns the per-agent token ceiling from settings

The run() method is final — do not override it in subclasses.
"""

# stdlib
import asyncio
import time
from abc import ABC, abstractmethod

# third-party
import anthropic

# local
from app.config import get_settings
from app.orchestrator.context import AgentCallRecord, AgentContext


class BaseAgent(ABC):
    """
    Abstract base for ResearcherAgent, ReasonerAgent, and SynthesizerAgent.

    Lifecycle per pipeline run:
      1. Orchestrator calls agent.run(context)
      2. run() calls build_user_message(context)  ← defined by subclass
      3. run() calls _call_anthropic_with_retry(user_message)
      4. run() wraps result in AgentCallRecord and returns it

    The Anthropic client is created once in __init__ and reused across calls —
    the Orchestrator singleton guarantees this is safe (single-threaded pipeline).
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
        # One Anthropic client per agent instance — reused across pipeline runs.
        self.client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        # max_tokens is resolved once at construction from settings.
        self.max_tokens: int = self._get_max_tokens()

    # ── Abstract methods (implemented by each concrete agent) ──────────────────

    @abstractmethod
    def build_user_message(self, context: AgentContext) -> str:
        """
        Construct the user-role message sent to the Anthropic API.
        The system prompt is passed separately via the `system` parameter —
        it is never included here.

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
          2. Call Anthropic with linear-backoff retry on rate limits
          3. Wrap the result in an AgentCallRecord with timing and token data

        Returns:
          AgentCallRecord — consumed by the Orchestrator, eventually serialised into
          OrchestrateResponse.pipeline_trace by response_builder.py

        Raises:
          AssertionError         — precondition violation (Orchestrator bug)
          anthropic.RateLimitError  — all retries exhausted (caught in main.py → HTTP 429)
          anthropic.APITimeoutError — hard timeout exceeded (caught in main.py → HTTP 503)
          anthropic.APIError        — other SDK errors (caught in main.py → HTTP 503)
        """
        user_message: str = self.build_user_message(context)

        start_time = time.time()
        response: anthropic.types.Message = await self._call_anthropic_with_retry(user_message)
        duration_ms: int = int((time.time() - start_time) * 1000)

        return AgentCallRecord(
            agent_name=self.AGENT_NAME,
            prompt_sent=user_message,
            response_text=response.content[0].text,
            duration_ms=duration_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    # ── Internal: Anthropic call with linear-backoff retry ────────────────────

    async def _call_anthropic_with_retry(
        self, user_message: str
    ) -> anthropic.types.Message:
        """
        Call client.messages.create() with up to 3 attempts on RateLimitError (429).

        Backoff strategy (DEC-08 — linear, not exponential):
          Attempt 1 → sleep 1× delay  → Attempt 2 → sleep 2× delay → Attempt 3 → raise

        The Anthropic SDK is synchronous — the call is wrapped in asyncio.to_thread()
        to avoid blocking FastAPI's event loop (DEC-03).

        Timeout and generic API errors are not retried — they propagate immediately.
        """
        max_retries = 3
        delay = self.settings.rate_limit_retry_delay  # default 1.0s

        for attempt in range(max_retries):
            try:
                # asyncio.to_thread() runs the synchronous SDK call in a thread pool,
                # freeing the event loop to handle other requests concurrently.
                response: anthropic.types.Message = await asyncio.to_thread(
                    self.client.messages.create,
                    model=self.settings.anthropic_model,
                    max_tokens=self.max_tokens,
                    system=self.SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": user_message}
                    ],
                )
                return response

            except anthropic.RateLimitError:
                if attempt == max_retries - 1:
                    # All retries exhausted — propagate to Orchestrator → main.py → HTTP 429
                    raise
                # Linear backoff: 1×, 2× — readable and predictable in demo conditions
                backoff_seconds = delay * (attempt + 1)
                await asyncio.sleep(backoff_seconds)

            except anthropic.APITimeoutError:
                # Timeout after AGENT_TIMEOUT_SECONDS — no retry, propagate immediately
                raise

            except anthropic.APIError:
                # All other API errors (auth, server errors, etc.) — propagate immediately
                raise

        # Unreachable — the loop always returns or raises before exhaustion,
        # but satisfies type checkers expecting a return on all paths.
        raise RuntimeError(f"[{self.AGENT_NAME}] _call_anthropic_with_retry: unexpected exit")
