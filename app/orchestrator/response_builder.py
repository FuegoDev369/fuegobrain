"""
app/orchestrator/response_builder.py
Converts the internal PipelineResult dataclass into the OrchestrateResponse
Pydantic model that FastAPI serialises to JSON.

This module is the sole conversion layer between the pipeline's internal
dataclasses (context.py) and the HTTP API contracts (models.py).
It is a pure function — no I/O, no side effects, fully unit-testable.

Key renames that happen here (see INTERFACES-CRITIQUES.md):
  AgentCallRecord.agent_name    → AgentTraceItem.agent
  AgentCallRecord.response_text → AgentTraceItem.response
`provider` and `model` keep identical names — no rename (TICKET-33/34).

TICKET-35 fix: PipelineMetadata.models_used (list[str], de-duplicated,
execution-order-preserving) is now CALCULATED here from
result.agent_records — it is NOT mapped 1:1 from any PipelineResult field.
PipelineResult.model_used no longer exists (removed TICKET-34) — it was
structurally incapable of representing 3 potentially different
provider/model combinations with a single string.
"""

# local
from app.orchestrator.context import AgentCallRecord, PipelineResult
from app.models import AgentTraceItem, OrchestrateResponse, PipelineMetadata


def build_response(result: PipelineResult) -> OrchestrateResponse:
    """
    Convert a PipelineResult (internal dataclasses) into an OrchestrateResponse
    (Pydantic model) ready for JSON serialisation by FastAPI.

    Args:
        result: The completed pipeline result produced by orchestrator.py.

    Returns:
        OrchestrateResponse — the full HTTP response payload.

    Note:
        Token totals are summed here rather than pre-computed in the Orchestrator
        to keep orchestrator.py focused on pipeline coordination, not accounting.
    """
    # ── Step 1 : Build pipeline_trace ─────────────────────────────────────────
    # Map each AgentCallRecord → AgentTraceItem, applying the two field renames.
    pipeline_trace: list[AgentTraceItem] = [
        _record_to_trace_item(record)
        for record in result.agent_records
    ]

    # ── Step 2 : Aggregate token counts ───────────────────────────────────────
    total_input_tokens: int = sum(r.input_tokens for r in result.agent_records)
    total_output_tokens: int = sum(r.output_tokens for r in result.agent_records)

    # ── Step 3 : Build metadata ───────────────────────────────────────────────
    # models_used (TICKET-35 fix): de-duplicated, execution-order-preserving
    # list of "provider/model" combos actually used across the 3 agents.
    # dict.fromkeys() is the idiomatic way to de-duplicate while preserving
    # insertion order in Python 3.7+ — a set() would lose the order.
    # Stays at 1 element in the default config (all 3 agents share
    # gemini/gemini-2.5-flash, DEC-17); contains multiple entries for a
    # mixed per-agent configuration.
    models_used: list[str] = list(dict.fromkeys(
        f"{r.provider}/{r.model}" for r in result.agent_records
    ))

    metadata = PipelineMetadata(
        total_duration_ms=result.total_duration_ms,
        models_used=models_used,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        agent_count=len(result.agent_records),
    )

    # ── Step 4 : Assemble and return the full response ─────────────────────────
    return OrchestrateResponse(
        final_answer=result.final_answer,
        pipeline_trace=pipeline_trace,
        metadata=metadata,
        query=result.original_query,
    )


def _record_to_trace_item(record: AgentCallRecord) -> AgentTraceItem:
    """
    Map a single AgentCallRecord dataclass to an AgentTraceItem Pydantic model.

    Field renames (dataclass → Pydantic):
      record.agent_name    → item.agent     (HTTP-friendly name)
      record.response_text → item.response  (HTTP-friendly name)

    `provider` and `model` (TICKET-33/34) keep identical names — no rename.
    All other fields are identical in name and type.
    """
    return AgentTraceItem(
        agent=record.agent_name,         # renamed: agent_name → agent
        provider=record.provider,        # NEW (TICKET-33/34) — no rename
        model=record.model,               # NEW (TICKET-33/34) — no rename
        prompt_sent=record.prompt_sent,
        response=record.response_text,   # renamed: response_text → response
        duration_ms=record.duration_ms,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
    )
