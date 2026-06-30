"""
app/orchestrator/agents/researcher.py
First agent in the FuegoBrain pipeline — responsible for fact collection only.

The ResearcherAgent receives the raw user query and produces a structured
list of facts, context, and knowledge gaps. It deliberately does NOT reason
or draw conclusions — that role belongs to ReasonerAgent.

System prompt is stored here in source code, not in .env — this is intentional.
It makes prompt engineering visible, reviewable, and versionable alongside the code.
"""

# local
from app.config import get_settings
from app.orchestrator.agents.base_agent import BaseAgent
from app.orchestrator.context import AgentContext


# ── System Prompt ──────────────────────────────────────────────────────────────
# Stored as a module-level constant so it can be:
#   1. Imported and asserted non-empty in tests/test_agents.py
#   2. Referenced from the README "Agent Prompts" section without duplication
#   3. Read in < 30 seconds by any reviewer — it's the core of Prompt Engineering

RESEARCHER_SYSTEM_PROMPT = """You are a Research Agent. Your only job is to identify and collect the relevant
facts, data points, and contextual information needed to answer the user's question.

STRICT RULES:
- Do NOT reason about the facts. Do NOT draw conclusions.
- Do NOT synthesize or write a final answer.
- Output ONLY a structured list of relevant facts and context.
- If you identify gaps in available knowledge, list them explicitly as "UNKNOWN: [gap]".
- Be exhaustive on facts, minimal on interpretation.

OUTPUT FORMAT:
FACTS:
- [fact 1]
- [fact 2]
...

CONTEXT:
- [contextual element 1]
- [contextual element 2]
...

UNKNOWNS (if any):
- [knowledge gap 1]"""


# ── ResearcherAgent ────────────────────────────────────────────────────────────

class ResearcherAgent(BaseAgent):
    """
    Concrete implementation of the Researcher role.

    Pipeline position : 1st — receives only the original query.
    Input             : AgentContext.original_query
    Output (stored)   : AgentContext.researcher_output  (set by Orchestrator)
    Output format     : FACTS / CONTEXT / UNKNOWNS structured list

    Does NOT receive researcher_output or reasoner_output — those don't exist yet.
    The Orchestrator guarantees this by calling ResearcherAgent first.
    """

    @property
    def AGENT_NAME(self) -> str:
        return "researcher"

    @property
    def SYSTEM_PROMPT(self) -> str:
        # Return the module-level constant — single source of truth for the prompt.
        return RESEARCHER_SYSTEM_PROMPT

    def _get_max_tokens(self) -> int:
        """
        Returns MAX_TOKENS_RESEARCHER from settings (default: 800).
        The Researcher produces structured lists — 800 tokens is sufficient
        for comprehensive fact collection without overrunning the context budget.
        """
        return get_settings().max_tokens_researcher

    def build_user_message(self, context: AgentContext) -> str:
        """
        Construct the user-role message for the Researcher.

        The Researcher only needs the original query — no prior agent output
        is available or expected at this pipeline stage.

        Args:
            context: AgentContext with original_query set.
                     researcher_output and reasoner_output are expected to be None.

        Returns:
            A clear instruction string asking for fact collection on the query.
        """
        return (
            f"Please research and collect all relevant facts for the following question:"
            f"\n\n{context.original_query}"
        )