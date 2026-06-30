"""
app/orchestrator/context.py
Plain-Python dataclasses that carry state through the FuegoBrain pipeline.

Three structures:
  - AgentContext     : mutable context passed between agents (query + cumulative outputs)
  - AgentCallRecord  : immutable record of a single agent execution (for pipeline_trace)
  - PipelineResult   : final aggregate produced by the Orchestrator before serialisation

Design note: deliberately plain dataclasses, not Pydantic models.
Pydantic is reserved for the HTTP boundary (app/models.py).
The conversion from dataclasses → Pydantic happens in response_builder.py.
"""

# stdlib
import time
from dataclasses import dataclass, field
from typing import Optional


# ── AgentContext ───────────────────────────────────────────────────────────────

@dataclass
class AgentContext:
    """
    Cumulative context passed to each agent in sequence.
    Mutated by the Orchestrator after each agent completes — never replaced.

    Invariants enforced by the Orchestrator (not by this class):
      - original_query  : IMMUTABLE — never modified after creation
      - researcher_output : None until ResearcherAgent.run() completes
      - reasoner_output   : None until ReasonerAgent.run() completes

    ReasonerAgent.build_user_message()  asserts researcher_output is not None.
    SynthesizerAgent.build_user_message() asserts both outputs are not None.
    The Orchestrator guarantees these preconditions via execution order.
    """

    original_query: str                      # Set at pipeline start — never modified
    researcher_output: Optional[str] = None  # Populated after ResearcherAgent runs
    reasoner_output: Optional[str] = None    # Populated after ReasonerAgent runs

    # Note: synthesizer_output is not stored here.
    # It becomes PipelineResult.final_answer directly.


# ── AgentCallRecord ────────────────────────────────────────────────────────────

@dataclass
class AgentCallRecord:
    """
    Immutable record of a single agent execution.
    Created inside BaseAgent.run() and returned to the Orchestrator.
    Three records (researcher, reasoner, synthesizer) are collected into
    PipelineResult.agent_records, then mapped to AgentTraceItem Pydantic
    models in response_builder.py.

    Naming note — the HTTP boundary renames two fields (see INTERFACES-CRITIQUES.md):
      agent_name    → AgentTraceItem.agent
      response_text → AgentTraceItem.response
    Do NOT rename these attributes — response_builder.py relies on them exactly.
    """

    agent_name: str       # "researcher" | "reasoner" | "synthesizer"
    prompt_sent: str      # user-role message sent to Anthropic (system prompt excluded)
    response_text: str    # raw text from response.content[0].text
    duration_ms: int      # int((end_time - start_time) * 1000)
    input_tokens: int     # response.usage.input_tokens
    output_tokens: int    # response.usage.output_tokens
    started_at: float = field(default_factory=time.time)  # Unix timestamp (for ordering/debug)


# ── PipelineResult ─────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Final aggregate produced by the Orchestrator after all three agents complete.
    Passed to ResponseBuilder.build_response() which converts it to the
    OrchestrateResponse Pydantic model for JSON serialisation.

    total_duration_ms is measured in orchestrator.py (pipeline start → pipeline end),
    not as the sum of agent durations — captures any overhead between agent calls.
    """

    final_answer: str                        # synthesizer_record.response_text
    agent_records: list[AgentCallRecord]     # [researcher, reasoner, synthesizer] in order
    total_duration_ms: int                   # end-to-end wall-clock duration
    model_used: str                          # settings.anthropic_model ("claude-sonnet-4-6")
    original_query: str                      # echo of the original query for the response
