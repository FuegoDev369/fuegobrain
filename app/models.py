"""
app/models.py
Pydantic v2 models defining the HTTP API contracts for FuegoBrain.
These models live exclusively at the HTTP boundary — internal pipeline
structures use plain dataclasses (see orchestrator/context.py).
"""

# stdlib
from typing import Optional

# third-party
from pydantic import BaseModel, ConfigDict, Field


# ── Request model ─────────────────────────────────────────────────────────────

class OrchestrateRequest(BaseModel):
    """
    Validates and types the body of POST /orchestrate.
    Pydantic enforces min/max length automatically — no manual check needed.
    """

    query: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="The complex question to be processed by the multi-agent pipeline",
    )

    # Swagger /docs example — shown in the "Try it out" panel
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Analyse l'impact de l'adoption crypto en Afrique de l'Ouest"
            }
        }
    )


# ── Pipeline trace models ─────────────────────────────────────────────────────

class AgentTraceItem(BaseModel):
    """
    Represents the full execution trace of one agent in the pipeline.
    Produced by ResponseBuilder from an internal AgentCallRecord dataclass.

    Note on naming:
      - `agent`    ← AgentCallRecord.agent_name    (renamed at HTTP boundary)
      - `response` ← AgentCallRecord.response_text  (renamed at HTTP boundary)
    """

    agent: str = Field(
        ...,
        description="Agent identifier: 'researcher' | 'reasoner' | 'synthesizer'",
    )
    prompt_sent: str = Field(
        ...,
        description="The user-role message sent to the Anthropic API (system prompt excluded)",
    )
    response: str = Field(
        ...,
        description="Raw text response from the agent",
    )
    duration_ms: int = Field(
        ...,
        description="Wall-clock duration of the Anthropic API call in milliseconds",
    )
    input_tokens: int = Field(
        ...,
        description="Tokens consumed as input (from usage.input_tokens)",
    )
    output_tokens: int = Field(
        ...,
        description="Tokens produced as output (from usage.output_tokens)",
    )


class PipelineMetadata(BaseModel):
    """
    Aggregated metadata for the complete pipeline execution.
    Computed in ResponseBuilder from the PipelineResult dataclass.
    """

    total_duration_ms: int = Field(
        ...,
        description="End-to-end wall-clock duration of the pipeline in milliseconds",
    )
    model: str = Field(
        ...,
        description="Anthropic model used for all agents (e.g. 'claude-sonnet-4-6')",
    )
    total_input_tokens: int = Field(
        ...,
        description="Sum of input tokens across all 3 agents",
    )
    total_output_tokens: int = Field(
        ...,
        description="Sum of output tokens across all 3 agents",
    )
    agent_count: int = Field(
        default=3,
        description="Number of agents in the pipeline (fixed at 3 in v1)",
    )


# ── Response model ────────────────────────────────────────────────────────────

class OrchestrateResponse(BaseModel):
    """
    Complete response structure for POST /orchestrate.
    Serialized directly to JSON by FastAPI.
    """

    final_answer: str = Field(
        ...,
        description="The Synthesizer's final answer — ready to display to the end user",
    )
    pipeline_trace: list[AgentTraceItem] = Field(
        ...,
        description="Ordered execution trace: [researcher, reasoner, synthesizer]",
    )
    metadata: PipelineMetadata = Field(
        ...,
        description="Aggregated pipeline metrics (duration, tokens, model)",
    )
    query: str = Field(
        ...,
        description="Echo of the original query for client-side reference",
    )


# ── Utility models ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response model for GET /health — used by UptimeRobot and Docker healthcheck."""

    status: str = Field(default="ok", description="Always 'ok' if the service is up")
    model: str = Field(..., description="Anthropic model currently configured")
    version: str = Field(default="1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """
    Structured error response for 429 / 503 / 500 HTTP errors.
    Used in the `responses` dict of FastAPI route decorators for Swagger docs.
    The actual HTTPException.detail field carries the message at runtime.
    """

    error: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(
        default=None,
        description="Technical detail or stack trace (optional)",
    )
