"""
tests/test_agents.py
Unit tests for the three FuegoBrain agents (Researcher, Reasoner, Synthesizer).

These tests are fully mocked — no real Anthropic API calls are made and no
real API key is required. They only exercise:
  - build_user_message() formatting and preconditions (AssertionError on
    missing context fields)
  - _extract_confidence() parsing logic (synthesizer.py)
  - presence/non-emptiness of the three module-level system prompts

A dummy ANTHROPIC_API_KEY is injected into the environment before any agent
is instantiated, since BaseAgent.__init__() reads it via get_settings() and
Settings.anthropic_api_key has no default (DEC-01). Constructing an agent
does not trigger a network call — only client.messages.create() (never
invoked here) would.
"""

# stdlib
import os

# third-party
import pytest

# Ensure a dummy API key is present before any module under test calls
# get_settings() — Settings.anthropic_api_key is mandatory (no default).
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")

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
