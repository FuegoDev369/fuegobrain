"""
app/main.py
FastAPI application entrypoint for FuegoBrain.
Wires together config, models, and the Orchestrator into the public HTTP API:
  GET  /health      — liveness probe (Render/UptimeRobot/Docker healthcheck)
  POST /orchestrate — runs the 3-agent pipeline on a user query
  /demo             — static files for the web demo (mounted if present)
"""

# stdlib
import os
from contextlib import asynccontextmanager

# third-party
import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# local
from app.config import get_settings
from app.models import (
    ErrorResponse,
    HealthResponse,
    OrchestrateRequest,
    OrchestrateResponse,
)
from app.orchestrator import Orchestrator

# ── Demo directory location ─────────────────────────────────────────────────
# Resolved relative to this file: app/main.py -> ../demo
DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "demo")


# ── Lifespan — singleton Orchestrator ───────────────────────────────────────
# FastAPI's modern lifespan pattern (replaces the deprecated @app.on_event).
# The Orchestrator (and the 3 agents + Anthropic clients it owns) is created
# exactly once per worker process and reused across every request.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: instantiate the orchestrator once.
    app.state.orchestrator = Orchestrator()
    yield
    # Shutdown: nothing to clean up in v1 (no open connections to close).


app = FastAPI(
    title="FuegoBrain — Multi-Agent Orchestration API",
    description="""
A transparent pipeline that decomposes complex questions into 3 specialized agents.
Built to be read, not just used.

## Pipeline
1. **Researcher** — Collects relevant facts and context
2. **Reasoner** — Analyzes facts, identifies tensions and implications
3. **Synthesizer** — Writes the final structured answer

## Architecture
Each agent call is fully traced: prompt sent, response received, duration, tokens.
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────
# Origins are configurable via DEMO_CORS_ORIGINS in .env — never hardcoded.
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Demo-Key"],
)


# ── GET /health ──────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Liveness probe used by Render, UptimeRobot, and the Docker healthcheck."""
    return HealthResponse(model=get_settings().anthropic_model)


# ── POST /orchestrate ────────────────────────────────────────────────────
@app.post(
    "/orchestrate",
    response_model=OrchestrateResponse,
    tags=["Pipeline"],
    summary="Run the multi-agent pipeline on a complex query",
    responses={
        200: {"description": "Pipeline completed successfully"},
        422: {"description": "Validation error — query too short/long"},
        429: {"model": ErrorResponse, "description": "Anthropic API rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "Anthropic API unavailable"},
    },
)
async def orchestrate(request: OrchestrateRequest) -> OrchestrateResponse:
    """
    Run the full Researcher → Reasoner → Synthesizer pipeline on `request.query`.
    Anthropic SDK errors propagate unwrapped from the Orchestrator and are
    mapped to HTTP status codes here — this is the single place that decides
    what each error type means for the client.
    """
    orchestrator: Orchestrator = app.state.orchestrator
    try:
        return await orchestrator.run(request.query)
    except anthropic.RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Anthropic API rate limit reached. Please wait a moment and retry.",
        )
    except anthropic.APIError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Anthropic API error: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline internal error: {str(e)}",
        )


# ── Static demo files ────────────────────────────────────────────────────
# Mounted AFTER the API routes so they are never shadowed.
# Guarded by os.path.exists so the API (and pytest) can run without demo/.
if os.path.exists(DEMO_DIR):
    app.mount("/demo", StaticFiles(directory=DEMO_DIR, html=True), name="demo")

    @app.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        """Redirect the root URL to the web demo's index page."""
        return FileResponse(os.path.join(DEMO_DIR, "index.html"))
