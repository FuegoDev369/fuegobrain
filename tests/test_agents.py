"""
tests/test_agents.py
Unit tests for the three FuegoBrain agents (Researcher, Reasoner, Synthesizer).

These tests are fully mocked — no real LLM API calls are made and no real
API key is required. They only exercise:
  - build_user_message() formatting and preconditions (AssertionError on
    missing context fields)
  - _extract_confidence() parsing logic (synthesizer.py)
  - presence/non-emptiness of the three module-level system prompts
  - that BaseAgent correctly stores the resolved provider name on the
    instance (TICKET-34 regression check)

A dummy GEMINI_API_KEY is injected into the environment before any agent is
instantiated, since Gemini is the default provider for all three agents
(DEC-17) and BaseAgent.__init__() resolves (provider, model, api_key) via
Settings.get_agent_config(), which raises ValueError if the configured
provider's key is missing (DEC-18). Constructing an agent does not trigger
a network call — only provider.call() (never invoked here) would.

TICKET-36 note: this file previously injected ANTHROPIC_API_KEY, a stale
assumption from before DEC-17/18 (when anthropic_api_key was the sole
mandatory key). Gemini is now the default provider for all three agents,
so GEMINI_API_KEY is what get_agent_config() actually needs here.
"""

# stdlib
import os

# third-party
import pytest

# Ensure a dummy API key is present before any module under test calls
# get_settings() — Gemini is the default provider (DEC-17) for all three
# agents, so GEMINI_API_KEY (not ANTHROPIC_API_KEY) is what
# get_agent_config() actually needs by default.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")

# local
from app.orchestrator.agents.reasoner import REASONER_SYSTEM_PROMPT, ReasonerAgent
from app.orchestrator.agents.researcher import (
    RESEARCHER_SYSTEM_PROMPT,
    ResearcherAgent,
)
from app.orchestrator.agents.synthesizer import (
    SYNTHESIZER_SYSTEM_PROMPT,
    SynthesizerAgent,
    _extract_confidence,
)
from app.orchestrator.context import AgentContext


# ── ResearcherAgent ──────────────────────────────────────────────────────────

def test_researcher_build_user_message():
    """The Researcher's user message must embed the original query verbatim."""
    ctx = AgentContext(original_query="Test query")
    agent = ResearcherAgent()
    msg = agent.build_user_message(ctx)
    assert "Test query" in msg


# ── ReasonerAgent ────────────────────────────────────────────────────────────

def test_reasoner_requires_researcher_output():
    """ReasonerAgent must refuse to build a message without researcher_output."""
    ctx = AgentContext(original_query="Test", researcher_output=None)
    agent = ReasonerAgent()
    with pytest.raises(AssertionError):
        agent.build_user_message(ctx)


def test_reasoner_build_user_message():
    """The Reasoner's user message must include both the query and facts sections."""
    ctx = AgentContext(original_query="Test", researcher_output="FACTS:\n- fact1")
    agent = ReasonerAgent()
    msg = agent.build_user_message(ctx)
    assert "[ORIGINAL QUERY]" in msg
    assert "[RESEARCHER OUTPUT]" in msg


# ── SynthesizerAgent ─────────────────────────────────────────────────────────

def test_synthesizer_requires_both_outputs():
    """SynthesizerAgent must refuse to build a message without reasoner_output."""
    ctx = AgentContext(
        original_query="Test", researcher_output="facts", reasoner_output=None
    )
    agent = SynthesizerAgent()
    with pytest.raises(AssertionError):
        agent.build_user_message(ctx)


def test_extract_confidence_high():
    """_extract_confidence() must parse the HIGH keyword from a well-formed line."""
    result = _extract_confidence("ANALYSIS:\n- point\nCONFIDENCE: HIGH — well sourced")
    assert result == "HIGH"


def test_extract_confidence_fallback():
    """_extract_confidence() must fall back to MEDIUM when no CONFIDENCE line exists."""
    result = _extract_confidence("No confidence line here")
    assert result == "MEDIUM"


# ── System prompts ───────────────────────────────────────────────────────────

def test_system_prompts_not_empty():
    """
    System prompts live in source code (never in .env) — sanity-check that
    each one is a substantial, non-trivial block of instructions.
    """
    assert len(RESEARCHER_SYSTEM_PROMPT) > 100
    assert len(REASONER_SYSTEM_PROMPT) > 100
    assert len(SYNTHESIZER_SYSTEM_PROMPT) > 100


# ── Multi-provider regression (TICKET-34/36) ─────────────────────────────────

def test_agent_stores_provider_name():
    """
    Regression test for TICKET-34 — BaseAgent must store self.provider_name,
    not just resolve it locally and discard it (the original gap that
    triggered the TICKET-33..37 correctif — see diagnostic point 3 in
    PLAN-IMPLEMENTATION.md).

    Without this test, a future edit to BaseAgent.__init__() could silently
    reintroduce the same oversight (resolving provider_name locally without
    storing it on the instance) and nothing would catch it.
    """
    agent = ResearcherAgent()
    assert agent.provider_name == "gemini"  # default provider, DEC-17
    assert agent.model == "gemini-2.5-flash"  # default model
