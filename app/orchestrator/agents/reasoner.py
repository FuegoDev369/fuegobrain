"""
app/orchestrator/agents/reasoner.py
Second agent in the FuegoBrain pipeline.

Receives the original query and the Researcher's structured facts.
Produces analytical insights: causal relationships, contradictions,
implications, risk factors, key tensions, and a confidence rating.

Its output feeds directly into the Synthesizer — it is never shown to the
user as-is, but the confidence level is extracted and forwarded explicitly
(see SynthesizerAgent.build_user_message / _extract_confidence).

Precondition enforced at build_user_message():
  context.researcher_output must be a non-None str — guaranteed by the
  Orchestrator's execution order (ResearcherAgent always runs first).
"""

# local
from app.config import get_settings
from app.orchestrator.agents.base_agent import BaseAgent
from app.orchestrator.context import AgentContext


# ── System Prompt ──────────────────────────────────────────────────────────────
# Stored here intentionally — visible, commentable, versionable.
# Never moved to .env or config. This is part of the Prompt Engineering showcase.

REASONER_SYSTEM_PROMPT = """You are a Reasoning Agent. You receive raw facts collected by a Research Agent.
Your job is to analyze these facts and produce structured analytical insights.

STRICT RULES:
- Do NOT restate the facts you received. Assume the reader knows them.
- Do NOT write a final answer for the user.
- ONLY produce analytical output: causal relationships, contradictions,
  implications, risk factors, and key tensions.
- Flag any facts that seem inconsistent or unreliable.
- Your output will be used by a Synthesis Agent to write the final answer.

YOU RECEIVE:
[ORIGINAL QUERY]: The user's question
[RESEARCHER OUTPUT]: Facts and context collected

OUTPUT FORMAT:
ANALYSIS:
- [analytical insight 1 — identify relationships and implications]
- [analytical insight 2]
...

KEY TENSIONS:
- [contradiction or uncertainty 1]
...

CONFIDENCE: [HIGH / MEDIUM / LOW] — [one sentence justification]"""


# ── Agent ──────────────────────────────────────────────────────────────────────

class ReasonerAgent(BaseAgent):
    """
    Analytical agent — second in the FuegoBrain pipeline.

    Input  : original query + researcher_output (structured facts)
    Output : structured analysis with ANALYSIS / KEY TENSIONS / CONFIDENCE sections

    The CONFIDENCE line (HIGH / MEDIUM / LOW) at the end of the output is parsed
    by SynthesizerAgent._extract_confidence() and forwarded as [CONFIDENCE LEVEL]
    into the Synthesizer's user message — closing the information loop between
    the three agents.
    """

    AGENT_NAME = "reasoner"
    SYSTEM_PROMPT = REASONER_SYSTEM_PROMPT

    def _get_max_tokens(self) -> int:
        """Return the Reasoner's token ceiling from settings (default: 1000)."""
        return get_settings().max_tokens_reasoner

    def build_user_message(self, context: AgentContext) -> str:
        """
        Construct the user-role message for the Anthropic API call.

        Precondition: context.researcher_output must be populated.
        The Orchestrator guarantees this by running ResearcherAgent first.
        An AssertionError here indicates a bug in the pipeline execution order —
        it is not a user-facing error and should never reach a running production system.

        Args:
            context: Current pipeline context — researcher_output must be str.

        Returns:
            Formatted user message containing the original query and all
            collected facts from the Researcher.

        Raises:
            AssertionError: If researcher_output is None (Orchestrator ordering bug).
        """
        assert context.researcher_output is not None, (
            "ReasonerAgent requires researcher_output — "
            "check pipeline order in orchestrator.py"
        )

        return (
            f"[ORIGINAL QUERY]: {context.original_query}\n\n"
            f"[RESEARCHER OUTPUT]:\n{context.researcher_output}\n\n"
            f"Please analyze these facts and produce structured analytical insights."
        )
