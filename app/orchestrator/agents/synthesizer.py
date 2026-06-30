"""
app/orchestrator/agents/synthesizer.py
Third and final agent in the FuegoBrain pipeline.

Receives the original query, all facts collected by the Researcher, and the
structured analysis from the Reasoner, then writes the definitive answer
directly for the end user.

Key design decisions:
  - SYNTHESIZER_SYSTEM_PROMPT is defined here in source — visible and versionable.
  - _extract_confidence() is a module-level function (not a method) so it can be
    tested in isolation without instantiating the full agent (DEC-07).
  - Language detection is delegated to the model via the system prompt rule:
    "Use the language of the original query."
"""

# stdlib
# (none beyond what BaseAgent provides)

# local
from app.config import get_settings
from app.orchestrator.agents.base_agent import BaseAgent
from app.orchestrator.context import AgentContext


# ── System prompt ──────────────────────────────────────────────────────────────
# Stored in source code intentionally — proves Prompt Engineering skill.
# Never move this to .env or any external config.

SYNTHESIZER_SYSTEM_PROMPT = """You are a Synthesis Agent. You receive a user's question, collected facts,
and analytical insights. Your job is to write the final answer.

STRICT RULES:
- Write directly for the end user. Clear, structured, directly useful.
- Ground every claim in the facts and analysis you received.
- Do NOT introduce new facts or reasoning not present in your inputs.
- If confidence is LOW, explicitly note limitations in your answer.
- Use the language of the original query (French if French, English if English).

YOU RECEIVE:
[ORIGINAL QUERY]: The user's question
[RESEARCHER OUTPUT]: Facts and context
[REASONER OUTPUT]: Analysis and key tensions
[CONFIDENCE LEVEL]: From the Reasoner

OUTPUT FORMAT:
Write a clear, structured answer. Use headers if the answer has multiple
distinct sections. End with a "Limitations" section if confidence is MEDIUM or LOW."""


# ── Module-level helper ────────────────────────────────────────────────────────

def _extract_confidence(reasoner_output: str) -> str:
    """
    Parse the confidence level from the Reasoner's structured output.

    The Reasoner is prompted to produce a line like:
      CONFIDENCE: HIGH — well sourced and corroborated
      CONFIDENCE: MEDIUM — some data gaps
      CONFIDENCE: LOW — limited reliable sources

    This function extracts the keyword (HIGH / MEDIUM / LOW) from that line.
    Falls back to "MEDIUM" if the line is absent or malformed — a safe neutral
    default that triggers the Synthesizer's "Limitations" section.

    Module-level (not a method) so it can be tested directly:
      from app.orchestrator.agents.synthesizer import _extract_confidence

    Args:
        reasoner_output: Raw text response from ReasonerAgent.

    Returns:
        One of "HIGH", "MEDIUM", or "LOW".
    """
    for line in reasoner_output.split("\n"):
        stripped = line.strip()
        if stripped.startswith("CONFIDENCE:"):
            # "CONFIDENCE: HIGH — justification" → split on "—" to isolate keyword
            after_colon = stripped.split(":", 1)[1].strip()
            keyword = after_colon.split("—")[0].strip()
            if keyword in ("HIGH", "MEDIUM", "LOW"):
                return keyword
    # Fallback: if the Reasoner omitted the line or used unexpected formatting,
    # default to MEDIUM so the Synthesizer includes a Limitations note.
    return "MEDIUM"


# ── SynthesizerAgent ───────────────────────────────────────────────────────────

class SynthesizerAgent(BaseAgent):
    """
    Writes the final user-facing answer from all accumulated pipeline context.

    Preconditions (asserted in build_user_message, guaranteed by Orchestrator):
      - context.researcher_output must be a non-None str
      - context.reasoner_output must be a non-None str

    Token budget: max_tokens_synthesizer (default 1500) — the largest of the
    three agents because the final answer may include multiple structured sections.
    """

    @property
    def AGENT_NAME(self) -> str:
        return "synthesizer"

    @property
    def SYSTEM_PROMPT(self) -> str:
        return SYNTHESIZER_SYSTEM_PROMPT

    def _get_max_tokens(self) -> int:
        """Return the synthesizer token ceiling from settings (default 1500)."""
        return get_settings().max_tokens_synthesizer

    def build_user_message(self, context: AgentContext) -> str:
        """
        Construct the user-role message for the Synthesizer.

        Includes the full pipeline context:
          - Original query
          - Researcher output (facts and context)
          - Reasoner output (analysis, tensions, confidence)
          - Extracted confidence level (for the system prompt's conditional logic)

        Preconditions:
            context.researcher_output is not None — set by Orchestrator after step 1
            context.reasoner_output is not None   — set by Orchestrator after step 2

        Raises:
            AssertionError: if the Orchestrator called this agent out of order.
        """
        assert context.researcher_output is not None, (
            "SynthesizerAgent requires researcher_output — "
            "check pipeline execution order in orchestrator.py"
        )
        assert context.reasoner_output is not None, (
            "SynthesizerAgent requires reasoner_output — "
            "check pipeline execution order in orchestrator.py"
        )

        # Extract confidence level to inject explicitly into the message.
        # The system prompt instructs the Synthesizer to add a Limitations section
        # when confidence is MEDIUM or LOW.
        confidence_level = _extract_confidence(context.reasoner_output)

        return (
            f"[ORIGINAL QUERY]: {context.original_query}\n\n"
            f"[RESEARCHER OUTPUT]:\n{context.researcher_output}\n\n"
            f"[REASONER OUTPUT]:\n{context.reasoner_output}\n\n"
            f"[CONFIDENCE LEVEL]: {confidence_level}\n\n"
            f"Please write the final structured answer for the user."
        )