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
from app.providers import ProviderAPIError, ProviderRateLimitError

# ── Demo directory location ─────────────────────────────────────────────────
# Resolved relative to this file: app/main.py -> ../demo
DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "demo")


# ── Lifespan — singleton Orchestrator ───────────────────────────────────────
# FastAPI's modern lifespan pattern (replaces the deprecated @app.on_event).
# The Orchestrator (and the 3 agents + provider adapters they own) is created
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


# ── GET/HEAD /health ─────────────────────────────────────────────────────
# Explicit methods=["GET", "HEAD"] (not @app.get, which only registers GET
# on this FastAPI version — see docstring note below for why this matters).
@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Liveness probe used by Render, UptimeRobot, and the Docker healthcheck.

    ALWAYS returns 200 if the process is running — this is a liveness
    probe, not a readiness probe (DEC-22). It reports each agent's
    CONFIGURED provider/model from Settings (no LLM calls, zero cost), and
    surfaces a misconfiguration (missing API key for a configured provider)
    inline in the response body rather than raising — a monitoring tool
    like UptimeRobot should see the process as "up" even if one provider
    is misconfigured; that is a config problem to fix, not a reason to
    report the whole service as down.

    HEAD is explicitly supported alongside GET (not just left to framework
    defaults): UptimeRobot's free-tier HTTP(s) monitor sends HEAD requests
    by default and only allows switching to GET on a Pro plan — a GET-only
    route would return 405 to every free-tier keepalive ping, permanently
    marking the service "down" and defeating the anti-sleep mechanism this
    endpoint exists for (GUIDE-DEPLOIEMENT-RENDER.md, ÉTAPE 6). FastAPI
    does not auto-register HEAD for @app.get() routes here (unlike raw
    Starlette's Route, which does) — confirmed empirically, not assumed.
    """
    settings = get_settings()
    agents: dict[str, str] = {}
    for agent_name in ("researcher", "reasoner", "synthesizer"):
        try:
            provider, model, _ = settings.get_agent_config(agent_name)
            agents[agent_name] = f"{provider}/{model}"
        except ValueError as e:
            agents[agent_name] = f"MISCONFIGURED — {e}"
    return HealthResponse(agents=agents)


# ── POST /orchestrate ────────────────────────────────────────────────────
@app.post(
    "/orchestrate",
    response_model=OrchestrateResponse,
    tags=["Pipeline"],
    summary="Run the multi-agent pipeline on a complex query",
    responses={
        200: {"description": "Pipeline completed successfully"},
        422: {"description": "Validation error — query too short/long"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded on the configured LLM provider"},
        503: {"model": ErrorResponse, "description": "The configured LLM provider is unavailable"},
    },
)
async def orchestrate(request: OrchestrateRequest) -> OrchestrateResponse:
    """
    Run the full Researcher → Reasoner → Synthesizer pipeline on `request.query`.
    Provider errors (from whichever LLM provider each agent is configured to
    use — Anthropic, Mistral, Gemini, or Grok) propagate unwrapped from the
    Orchestrator and are mapped to HTTP status codes here — this is the
    single place that decides what each error type means for the client.
    """
    orchestrator: Orchestrator = app.state.orchestrator
    try:
        return await orchestrator.run(request.query)
    except ProviderRateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Rate limit reached on the configured LLM provider. Please wait a moment and retry.",
        )
    except ProviderAPIError as e:
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider error: {str(e)}",
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

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def root() -> FileResponse:
        """
        Redirect the root URL to the web demo's index page.

        HEAD is supported alongside GET for the same reason as /health
        (see health_check() docstring) — Render's own internal port-detection
        probe sends a HEAD request here during deploy; a 405 doesn't block
        the deploy (Render treats any HTTP response as "port is open"), but
        there is no reason to leave it inconsistent with /health once fixed.
        """
        return FileResponse(os.path.join(DEMO_DIR, "index.html"))
