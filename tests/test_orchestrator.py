"""
tests/test_orchestrator.py
Unit tests for the Orchestrator pipeline coordination logic.

These tests are fully mocked — no real LLM API calls are made and no real
API key is required. They exercise:
  - Strict execution order (researcher → reasoner → synthesizer) and that
    AgentContext is correctly populated between steps
  - The shape of the returned OrchestrateResponse (pipeline_trace length,
    field renames, metadata aggregation, including provider/model per
    agent and the de-duplicated models_used summary — TICKET-33/34/35)
  - The integrity of the sample_queries.json fixture used by the demo and
    other tests

A dummy GEMINI_API_KEY is injected into the environment before the
Orchestrator (and the agents it instantiates) is constructed, since Gemini
is the default provider for all three agents (DEC-17) and
BaseAgent.__init__() resolves (provider, model, api_key) via
Settings.get_agent_config(), which raises ValueError if the configured
provider's key is missing (DEC-18).

TICKET-36 note: this file previously injected ANTHROPIC_API_KEY (stale,
pre-DEC-17/18 assumption) and built AgentCallRecord instances without the
`provider`/`model` fields added in TICKET-34 — both are corrected here.
"""

# stdlib
import json
import os
from unittest.mock import AsyncMock, patch

# third-party
import pytest

# Ensure a dummy API key is present before any module under test calls
# get_settings() — Gemini is the default provider (DEC-17) for all three
# agents, so GEMINI_API_KEY (not ANTHROPIC_API_KEY) is what
# get_agent_config() actually needs by default.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")

# local
from app.models import OrchestrateResponse
from app.orchestrator.context import AgentCallRecord
from app.orchestrator.orchestrator import Orchestrator


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_agent_record():
    """
    Factory fixture: builds a fake AgentCallRecord for a given agent name.
    Used to mock BaseAgent.run() / ResearcherAgent.run() / etc. without
    making any real provider API call.

    `provider`/`model` default to "gemini"/"gemini-2.5-flash" — the default
    config for all three agents (DEC-17) — since AgentCallRecord now
    requires these two fields (TICKET-34).
    """

    def _make(
        agent_name: str,
        provider: str = "gemini",
        model: str = "gemini-2.5-flash",
    ) -> AgentCallRecord:
        return AgentCallRecord(
            agent_name=agent_name,
            provider=provider,
            model=model,
            prompt_sent=f"prompt for {agent_name}",
            response_text=f"response from {agent_name}",
            duration_ms=100,
            input_tokens=50,
            output_tokens=30,
        )

    return _make


@pytest.fixture
def orchestrator(fake_agent_record):
    """
    A real Orchestrator instance (real ResearcherAgent/ReasonerAgent/
    SynthesizerAgent objects — only their .run() coroutine is mocked).
    Each agent's .run() returns its own fake AgentCallRecord, in pipeline
    order, so context propagation can be verified.
    """
    orch = Orchestrator()
    orch.researcher.run = AsyncMock(return_value=fake_agent_record("researcher"))
    orch.reasoner.run = AsyncMock(return_value=fake_agent_record("reasoner"))
    orch.synthesizer.run = AsyncMock(return_value=fake_agent_record("synthesizer"))
    return orch


# ── Pipeline order & context propagation ────────────────────────────────────

@pytest.mark.asyncio
async def test_orchestrator_pipeline_order(orchestrator):
    """
    Researcher must run before Reasoner, which must run before Synthesizer.
    Verified via unittest.mock's call ordering across the three mocked
    coroutines (a single shared manager would over-complicate this — instead
    we assert call_count incrementally is unnecessary; we assert order via
    a shared list populated by side_effect).
    """
    call_order: list[str] = []

    async def researcher_side_effect(context):
        call_order.append("researcher")
        return AgentCallRecord(
            agent_name="researcher",
            provider="gemini",
            model="gemini-2.5-flash",
            prompt_sent="p",
            response_text="r1",
            duration_ms=1,
            input_tokens=1,
            output_tokens=1,
        )

    async def reasoner_side_effect(context):
        # Precondition check: researcher_output must already be set.
        assert context.researcher_output == "r1"
        call_order.append("reasoner")
        return AgentCallRecord(
            agent_name="reasoner",
            provider="gemini",
            model="gemini-2.5-flash",
            prompt_sent="p",
            response_text="r2",
            duration_ms=1,
            input_tokens=1,
            output_tokens=1,
        )

    async def synthesizer_side_effect(context):
        # Precondition check: both prior outputs must already be set.
        assert context.researcher_output == "r1"
        assert context.reasoner_output == "r2"
        call_order.append("synthesizer")
        return AgentCallRecord(
            agent_name="synthesizer",
            provider="gemini",
            model="gemini-2.5-flash",
            prompt_sent="p",
            response_text="r3",
            duration_ms=1,
            input_tokens=1,
            output_tokens=1,
        )

    orchestrator.researcher.run = AsyncMock(side_effect=researcher_side_effect)
    orchestrator.reasoner.run = AsyncMock(side_effect=reasoner_side_effect)
    orchestrator.synthesizer.run = AsyncMock(side_effect=synthesizer_side_effect)

    await orchestrator.run("Test query")

    assert call_order == ["researcher", "reasoner", "synthesizer"]


# ── Response shape ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_orchestrator_returns_orchestrate_response(orchestrator):
    """orchestrator.run() must return a fully-populated OrchestrateResponse."""
    result = await orchestrator.run("Test query")
    assert isinstance(result, OrchestrateResponse)
    assert result.query == "Test query"
    assert len(result.pipeline_trace) == 3


@pytest.mark.asyncio
async def test_orchestrator_pipeline_trace_content(orchestrator):
    """pipeline_trace items must be in researcher → reasoner → synthesizer order."""
    result = await orchestrator.run("Test query")
    assert result.pipeline_trace[0].agent == "researcher"
    assert result.pipeline_trace[1].agent == "reasoner"
    assert result.pipeline_trace[2].agent == "synthesizer"


@pytest.mark.asyncio
async def test_orchestrator_pipeline_trace_provider_and_model(orchestrator):
    """
    Each pipeline_trace item must carry its own provider/model (TICKET-33/34/35)
    — the fields that let a mixed per-agent configuration remain fully
    traceable, not just the default single-provider case.
    """
    result = await orchestrator.run("Test query")
    for item in result.pipeline_trace:
        assert item.provider == "gemini"
        assert item.model == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_orchestrator_metadata(orchestrator):
    """Pipeline metadata must reflect a 3-agent run with the configured models."""
    result = await orchestrator.run("Test query")
    assert result.metadata.agent_count == 3
    # Was: assert result.metadata.model == "claude-sonnet-4-6"  (stale —
    # PipelineMetadata.model no longer exists, replaced by models_used
    # (list[str], de-duplicated, execution-order-preserving) — TICKET-33/35).
    assert result.metadata.models_used == ["gemini/gemini-2.5-flash"]
    # Mocked agent calls are near-instant, so total_duration_ms can legitimately
    # round to 0ms — assert non-negative rather than strictly positive to avoid
    # a flaky test while still catching a negative/uncomputed value.
    assert result.metadata.total_duration_ms >= 0


# ── Fixture integrity ─────────────────────────────────────────────────────────

def test_sample_queries_valid_json():
    """
    tests/fixtures/sample_queries.json must contain exactly the 3 demo
    examples, each with a query long enough to pass OrchestrateRequest
    validation (min_length=10).
    """
    with open("tests/fixtures/sample_queries.json", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data["examples"]) == 3
    for example in data["examples"]:
        assert "query" in example
        assert len(example["query"]) >= 10
