# FuegoBrain ⚡
> A transparent multi-agent pipeline. Built to be read, not just used.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![Providers](https://img.shields.io/badge/providers-Anthropic%20%7C%20Mistral%20%7C%20Gemini%20%7C%20Grok-ff4500)
![Default](https://img.shields.io/badge/default-Gemini%20(free)-22c55e)
![Live Demo](https://img.shields.io/badge/demo-live-22c55e)

[![CI](https://github.com/fuegodev369/fuegobrain/actions/workflows/ci.yml/badge.svg)](https://github.com/fuegodev369/fuegobrain/actions/workflows/ci.yml)

> 🎬 *Demo GIF placeholder — record a 5-6s clip of the pipeline animation
> running on an example query and drop it here as `docs/demo.gif`.*

---

## The Problem

Most "AI agent" demos are a single LLM call with a system prompt — there's
no way to see *how* the answer was built, and no separation between
gathering facts, reasoning about them, and writing the answer. FuegoBrain
makes that pipeline explicit: three specialized agents, each with one job,
each fully traced — prompt sent, raw response, duration, and token usage —
so the reasoning process is as visible as the final answer.

---

## Architecture

```
                    ┌─────────────────┐
   User Query  ───▶ │  ResearcherAgent │  → FACTS / CONTEXT / UNKNOWNS
                    └────────┬─────────┘
                             │ researcher_output
                             ▼
                    ┌─────────────────┐
                    │   ReasonerAgent  │  → ANALYSIS / KEY TENSIONS / CONFIDENCE
                    └────────┬─────────┘
                             │ reasoner_output
                             ▼
                    ┌──────────────────┐
                    │ SynthesizerAgent │  → final_answer (for the user)
                    └────────┬─────────┘
                             │
                             ▼
                    OrchestrateResponse
              (final_answer + pipeline_trace + metadata)
```

- **Researcher** — collects relevant facts, context, and explicit knowledge
  gaps from the raw query. No interpretation, no conclusions.
- **Reasoner** — analyzes the Researcher's facts: causal relationships,
  contradictions, risk factors, key tensions, and a confidence rating.
- **Synthesizer** — writes the final, user-facing answer grounded strictly
  in what the Researcher and Reasoner produced.

The pipeline is strictly sequential and stateless between requests. State
flows through a single mutable `AgentContext` dataclass owned by the
`Orchestrator` — no LangChain, no hidden framework magic.

---

## Live Demo

🔗 **https://fuegobrain.onrender.com** *(placeholder — replace with the deployed Render URL)*

> 🖼️ *Screenshot placeholder — drop a screenshot of the running demo at
> `docs/screenshot.png` and reference it here once deployed.*

---

## Try the API

```bash
curl -X POST https://fuegobrain.onrender.com/orchestrate \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key differences between RAG and fine-tuning for LLMs?"}'
```

```bash
curl https://fuegobrain.onrender.com/health
```

Interactive Swagger docs are always available at `https://fuegobrain.onrender.com/docs`.

---

## Agent Prompts

All three system prompts live in source code (`app/orchestrator/agents/*.py`)
as module-level constants — never in `.env`, never hidden. This is
intentional: it makes the prompt engineering reviewable, versionable, and
testable in isolation.

### ResearcherAgent

```
You are a Research Agent. Your only job is to identify and collect the relevant
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
- [knowledge gap 1]
```

**Why these rules:** separating fact collection from reasoning prevents the
model from jumping to conclusions before all relevant context is on the
table — the same discipline a human researcher would apply before handing
off to an analyst. The explicit `UNKNOWN:` format forces honesty about gaps
instead of quietly papering over them.

### ReasonerAgent

```
You are a Reasoning Agent. You receive raw facts collected by a Research Agent.
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

CONFIDENCE: [HIGH / MEDIUM / LOW] — [one sentence justification]
```

**Why these rules:** banning restatement keeps the output dense and
forces the model to add value rather than summarize. The mandatory
`CONFIDENCE` line gives the Synthesizer (and the end user) an explicit,
parseable signal about how much to trust the answer — extracted
programmatically by `_extract_confidence()` rather than re-inferred by
another LLM call.

### SynthesizerAgent

```
You are a Synthesis Agent. You receive a user's question, collected facts,
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
distinct sections. End with a "Limitations" section if confidence is MEDIUM or LOW.
```

**Why these rules:** the "no new facts" constraint is the project's core
anti-hallucination guard — the Synthesizer is explicitly forbidden from
inventing anything that didn't pass through the Researcher or Reasoner
first. Surfacing low confidence as a visible "Limitations" section, rather
than hiding it, is a deliberate trust-building choice.

---

## Run Locally

```bash
git clone https://github.com/fuegodev369/fuegobrain.git
cd fuegobrain
pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY (default provider, free tier)
uvicorn app.main:app --reload
```

Then open `http://localhost:8000` for the demo, or `http://localhost:8000/docs`
for the Swagger UI.

### Validating the full stack: it depends on what you're running on

This project ships three validation layers — running the app directly,
Docker, and CI — and not every environment can run all three. Pick the
path that matches your machine:

**Most desktop/laptop environments (Linux, macOS, Windows with Docker
Desktop):** all three layers work locally. Run the app as shown above,
then optionally confirm the container builds too:

```bash
docker build -t fuegobrain:local .
docker run --rm -e GEMINI_API_KEY=$GEMINI_API_KEY -p 8000:8000 fuegobrain:local
# or, for the dev-friendly hot-reload version:
docker-compose up
```

**Mobile/constrained environments (Termux on Android, some sandboxed CI
runners, chromebooks without a Linux container layer):** `uvicorn` runs
fine — FastAPI has no special requirements — but `docker build` will
usually fail with `Cannot connect to the Docker daemon`, because these
environments don't expose the kernel features (cgroups, namespaces) Docker
needs without root. **This isn't a project bug — it's a platform
limitation.** This repo was actually developed under exactly this
constraint; see `VALIDATION-NOTES.md` for the specifics. If you're in
this situation, skip straight to the GitHub Actions step below instead of
troubleshooting a local Docker install.

**Don't have Docker anywhere, but still want the build confirmed:** push
to GitHub. The CI workflow (`.github/workflows/ci.yml`) builds the image
on a clean Ubuntu runner with a working Docker daemon, on every push and
PR — no local Docker required. See `GUIDE-GITHUB-ACTIONS.md` for the full
walkthrough (setup, reading results, troubleshooting).

### Three things that sound similar but check different layers

| Layer | What it actually validates | Needs Docker locally? | Needs a real API key? |
|---|---|---|---|
| `uvicorn app.main:app --reload` | The app itself runs and serves requests | No | Yes |
| `docker build` / `docker-compose up` | The container image builds and starts correctly | Yes | Yes (to test `/orchestrate`; `/health` alone doesn't need it) |
| GitHub Actions (`ci.yml`) | Tests pass (mocked, no real API calls) + image builds on a clean runner | No (runs on GitHub's infrastructure) | No (uses a placeholder key — tests are mocked) |
| Render deployment | The image builds **and** runs in production: real network, real env vars, real public URL | No (Render builds it server-side) | Yes (real key, configured in Render's dashboard) |

The short version: **CI proves the build is sound. Render proves the
deployment actually works.** Neither one substitutes for the other —
see `GUIDE-GITHUB-ACTIONS.md` and `GUIDE-DEPLOIEMENT-RENDER.md` for the
complete procedures, including the free-tier sleep workaround and CORS
configuration for the demo frontend.

---

## Multi-Provider Support

FuegoBrain isn't locked to a single LLM vendor. Each agent — Researcher,
Reasoner, Synthesizer — can independently use Anthropic, Mistral, Gemini,
or Grok, configured entirely through environment variables. No code changes
required to switch.

**Default configuration runs entirely on Gemini's free tier** (Flash model,
no credit card required) — the project works out of the box at zero cost.

| Provider | Free tier? | Notes |
|---|---|---|
| **Gemini** (default) | ✅ Yes — confirmed, no CC required | Flash/Flash-Lite only; Pro moved to paid in April 2026 |
| **Mistral** | ✅ Yes — "Experiment" tier, no CC required | ~1B tokens/month, rate-limited (~2 RPM) |
| **Anthropic** | ❌ No | Paid from the first request |
| **Grok (xAI)** | ❌ No — verified against official docs | Some third-party articles claim free credits; not confirmed by x.ai's own pricing page as of June 2026 |

### Switching providers

Set per-agent provider and model in `.env`:

```bash
RESEARCHER_PROVIDER=mistral
RESEARCHER_MODEL=mistral-small-latest

SYNTHESIZER_PROVIDER=anthropic
SYNTHESIZER_MODEL=claude-sonnet-4-6
```

Only the API key for the provider(s) actually in use needs to be set — an
unused provider's key can stay empty in `.env`.

### How it works

A small adapter layer (`app/providers/`) normalizes each provider's SDK
response into a common `ProviderResponse` shape (`text`, `input_tokens`,
`output_tokens`). `BaseAgent` calls whichever provider is configured for
that agent without knowing which one it is — same retry logic, same
pipeline, regardless of vendor. See `app/providers/base_provider.py` for
the full interface.

---

## Stack & Choices

| Technology | Role | Why this choice |
|---|---|---|
| Python 3.11 | Runtime | Stable, modern typing features (`X \| None`), wide library support |
| FastAPI 0.115+ | HTTP layer | Async-native, automatic OpenAPI docs, Pydantic integration out of the box |
| Pydantic v2 | HTTP boundary validation | Strict input validation (`query` length 10–2000) and typed, self-documenting response schemas |
| stdlib `dataclasses` | Internal pipeline state | Pydantic is reserved for the HTTP boundary only — internal state stays framework-free and trivially readable |
| Multi-provider adapters (`app/providers/`) | LLM calls | Anthropic, Mistral, Gemini, and Grok SDKs behind a common `BaseLLMProvider` interface; each provider's synchronous SDK call is wrapped in `asyncio.to_thread()` to avoid blocking FastAPI's event loop — see "Multi-Provider Support" above |
| `pydantic-settings` | Configuration | Validates env vars at boot (fail fast on missing key) instead of crashing mid-request |
| HTML + CSS + JS vanilla | Demo frontend | Zero external dependencies — runs from a single static folder, no build step |
| Docker | Containerization | Reproducible builds; `python:3.11-slim` base for a small image |
| Render.com | Backend hosting | Free-tier friendly, builds directly from the `Dockerfile` |
| Vercel / GitHub Pages | Frontend hosting | Static demo can be hosted independently of the API |

**Notable architectural decision:** orchestration is hand-written Python —
no LangChain or similar framework. The entire pipeline (`orchestrator.py`)
is designed to be readable end-to-end by a mid-level Python developer in
under 10 minutes, with the agent execution order explicit in the code
rather than buried in configuration or framework abstractions.

---

## Project Structure

```
fuegobrain/
├── app/
│   ├── main.py                      # FastAPI app, routes, CORS, lifespan
│   ├── models.py                    # Pydantic v2 models — HTTP boundary only
│   ├── config.py                    # Settings via pydantic-settings + lru_cache (per-agent provider config)
│   ├── providers/
│   │   ├── base_provider.py         # BaseLLMProvider interface + ProviderResponse
│   │   ├── anthropic_provider.py    # Anthropic adapter
│   │   ├── mistral_provider.py      # Mistral adapter
│   │   ├── gemini_provider.py       # Gemini adapter (default)
│   │   ├── grok_provider.py         # Grok/xAI adapter (OpenAI-compatible SDK)
│   │   └── __init__.py              # PROVIDER_REGISTRY + get_provider() factory
│   └── orchestrator/
│       ├── orchestrator.py          # Sequential pipeline coordinator (read this first)
│       ├── context.py               # AgentContext / AgentCallRecord / PipelineResult (dataclasses)
│       ├── response_builder.py      # dataclass → Pydantic conversion + token aggregation
│       └── agents/
│           ├── base_agent.py        # Shared retry/timing/provider-call logic (provider-agnostic)
│           ├── researcher.py        # Agent 1 — fact collection
│           ├── reasoner.py          # Agent 2 — analysis
│           └── synthesizer.py       # Agent 3 — final answer
├── demo/
│   ├── index.html                   # Demo UI shell
│   ├── style.css                    # "Fuego" dark theme
│   └── app.js                       # Pipeline call + cosmetic stage animation
├── tests/
│   ├── test_agents.py               # Per-agent unit tests (mocked)
│   ├── test_orchestrator.py         # Pipeline order + response shape tests (mocked)
│   ├── test_providers.py            # Provider adapter unit tests (mocked)
│   └── fixtures/sample_queries.json # Example queries used by demo + tests
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Roadmap

- **v2 — Redis context management:** persist `AgentContext` across requests
  for follow-up questions without re-running the full pipeline.
- **v2 — Multi-LLM routing:** allow swapping the model per agent (e.g.
  a cheaper model for Researcher, a stronger one for Synthesizer).
- **v2 — Parallel agents:** explore where strict sequential ordering can be
  relaxed (e.g. running independent sub-research tasks concurrently)
  without compromising the Reasoner's preconditions.
- **v1.5 — Real-time pipeline status:** replace the current cosmetic
  frontend animation with genuine SSE/`/status/{request_id}` polling
  (deferred in v1 — see `DECISIONS-PLANNIFICATEUR.md`, DEC-05).
