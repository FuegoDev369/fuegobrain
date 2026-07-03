"""
app/orchestrator/orchestrator.py
The heart of FuegoBrain. Coordinates the sequential execution of the three agents,
accumulates context between steps, and assembles the final OrchestrateResponse.

This file is intentionally readable in < 10 minutes.
Every design decision is visible here — no magic, no frameworks.
Pipeline: Researcher → Reasoner → Synthesizer (strictly sequential, never parallel).
"""

# stdlib
import time

# local
from app.config import get_settings
from app.models import OrchestrateResponse
from app.orchestrator.agents import ResearcherAgent, ReasonerAgent, SynthesizerAgent
from app.orchestrator.context import AgentCallRecord, AgentContext, PipelineResult
from app.orchestrator.response_builder import build_response
from app.providers import ProviderAPIError, ProviderRateLimitError


class Orchestrator:
    """
    Coordinates the three-agent pipeline that powers POST /orchestrate.

    Instantiated once at application startup (via FastAPI lifespan) and reused
    across all requests. This means one provider adapter per agent per worker —
    no reconnection overhead per request. Which concrete provider each agent
    uses (Anthropic/Mistral/Gemini/Grok) is resolved per agent via Settings
    (DEC-17: Gemini is the default for all three).

    The pipeline is strictly sequential:
      1. ResearcherAgent  — collects facts and context
      2. ReasonerAgent    — analyses facts, surfaces tensions
      3. SynthesizerAgent — writes the final answer for the end user

    AgentContext is the shared data bus: it is created empty, then mutated
    after each agent completes. The Orchestrator guarantees preconditions
    (e.g. researcher_output is not None before Reasoner runs) via execution order.
    """

    def __init__(self) -> None:
        # Instantiate agents once — each resolves its own provider adapter
        # internally (Anthropic/Mistral/Gemini/Grok, per-agent config).
        # Reusing instances across requests avoids re-initialising the
        # underlying HTTP client on every call, which matters on
        # free-tier infrastructure.
        self.researcher = ResearcherAgent()
        self.reasoner = ReasonerAgent()
        self.synthesizer = SynthesizerAgent()
        self.settings = get_settings()

    async def run(self, query: str) -> OrchestrateResponse:
        """
        Execute the full pipeline on a raw query string.

        This is the single entry point called by main.py.
        Provider errors (ProviderRateLimitError, ProviderAPIError — generic,
        regardless of which LLM provider each agent is configured to use)
        are intentionally allowed to propagate; main.py catches them and
        maps them to HTTP status codes (429, 503).

        Args:
            query: The user's question, already validated by Pydantic
                   (10–2000 chars) in OrchestrateRequest.

        Returns:
            OrchestrateResponse: A fully-populated Pydantic model ready
                                 for JSON serialisation by FastAPI.
        """
        pipeline_start = time.time()

        # ── Shared context ────────────────────────────────────────────────────
        # AgentContext is the data bus between agents.
        # It is mutated in place after each step — never replaced.
        # This preserves a single source of truth and makes the flow explicit.
        context = AgentContext(original_query=query)

        # Accumulate one AgentCallRecord per agent (timing, tokens, prompts, response).
        # These become pipeline_trace in the HTTP response.
        records: list[AgentCallRecord] = []

        # ── STEP 1 : Researcher ───────────────────────────────────────────────
        # Input  : original query only
        # Output : structured fact list (FACTS / CONTEXT / UNKNOWNS)
        # The Researcher has no prior context — it sees only the raw question.
        researcher_record = await self._run_agent(self.researcher, context)
        context.researcher_output = researcher_record.response_text  # unlocks Reasoner
        records.append(researcher_record)

        # ── STEP 2 : Reasoner ─────────────────────────────────────────────────
        # Input  : original query + researcher_output
        # Output : analytical insights (ANALYSIS / KEY TENSIONS / CONFIDENCE)
        # Precondition: context.researcher_output is now set (guaranteed above).
        reasoner_record = await self._run_agent(self.reasoner, context)
        context.reasoner_output = reasoner_record.response_text  # unlocks Synthesizer
        records.append(reasoner_record)

        # ── STEP 3 : Synthesizer ──────────────────────────────────────────────
        # Input  : original query + researcher_output + reasoner_output
        # Output : final answer, written directly for the end user
        # Precondition: both researcher_output and reasoner_output are set.
        # No mutation after this step — synthesizer output IS the final answer.
        synthesizer_record = await self._run_agent(self.synthesizer, context)
        records.append(synthesizer_record)

        # ── Assemble result ───────────────────────────────────────────────────
        # Wall-clock total: measured here, not as the sum of agent durations,
        # so any inter-agent overhead is captured.
        total_duration_ms = int((time.time() - pipeline_start) * 1000)

        pipeline_result = PipelineResult(
            final_answer=synthesizer_record.response_text,
            agent_records=records,
            total_duration_ms=total_duration_ms,
            # model_used=... ← REMOVED (TICKET-34/35). The field no longer
            #                   exists on PipelineResult; PipelineMetadata.
            #                   models_used is computed directly from
            #                   agent_records in response_builder.py — no
            #                   need to pre-aggregate a single value here.
            original_query=query,
        )

        # Delegate dataclass → Pydantic conversion to ResponseBuilder.
        # This keeps orchestrator.py focused on coordination, not serialisation.
        return build_response(pipeline_result)

    async def _run_agent(
        self,
        agent: ResearcherAgent | ReasonerAgent | SynthesizerAgent,
        context: AgentContext,
    ) -> AgentCallRecord:
        """
        Execute a single agent and return its call record.

        Provider errors are re-raised as-is so main.py can map them to the
        correct HTTP status codes (429 for rate limits, 503 for other
        provider failures). This catches the GENERIC ProviderRateLimitError /
        ProviderAPIError (TICKET-24), not Anthropic-specific exceptions.
        Fixed in TICKET-35 — since TICKET-29, base_agent.py raises these
        generic types regardless of which provider (Anthropic/Mistral/
        Gemini/Grok) is actually configured; the previous
        `except anthropic.*` clauses here were dead code that could never
        match — the generic provider exception types are not subclasses
        of any Anthropic SDK exception.
        AssertionError (precondition violation) surfaces a pipeline bug —
        it should never occur in production if execution order is respected.

        Args:
            agent:   One of the three concrete agent instances.
            context: The shared, partially-populated AgentContext.

        Returns:
            AgentCallRecord: Timing, token counts, prompt sent, and raw response.

        Raises:
            ProviderRateLimitError: Retries already exhausted in BaseAgent.
            ProviderAPIError:       Any other provider API failure.
            AssertionError:         Precondition violated (pipeline bug).
        """
        try:
            return await agent.run(context)
        except ProviderRateLimitError:
            raise  # Retries already exhausted in BaseAgent — propagate to main.py
        except ProviderAPIError:
            raise  # Propagate to main.py → HTTP 503
