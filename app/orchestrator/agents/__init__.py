"""
app/orchestrator/agents/__init__.py
Public surface of the agents package.

Exposes the three concrete agent classes so that orchestrator.py (and tests)
can import them from a single location:

    from app.orchestrator.agents import ResearcherAgent, ReasonerAgent, SynthesizerAgent

Import order mirrors the pipeline execution order: Researcher → Reasoner → Synthesizer.
"""

from app.orchestrator.agents.researcher import ResearcherAgent
from app.orchestrator.agents.reasoner import ReasonerAgent
from app.orchestrator.agents.synthesizer import SynthesizerAgent

__all__ = ["ResearcherAgent", "ReasonerAgent", "SynthesizerAgent"]
