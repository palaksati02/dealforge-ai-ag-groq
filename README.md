# DealForge AI - Multi-Agent Procurement Automation

A natural-language purchase request goes through a pipeline of agents: requirement analysis, vendor discovery,
pricing (tiers, 18% GST, freight), compliance, weighted deal scoring, budget validation, negotiation, an
**independent risk audit by a Google ADK agent over A2A**, then human approval gating and Purchase Order issuance.

**Frameworks:** CrewAI + Google ADK only. **Protocols:** MCP (FastMCP) and A2A (official `a2a-sdk`).
**Design rule:** LLMs do language and reasoning; money and policy are deterministic MCP tools.

```
Browser UI -> FastAPI (:8000)
              |  CrewAI agents (LLM: Requirement, Negotiation, Coordination)
              |       |-- MCP tools (FastMCP: in-process or stdio child process) -> DB (MySQL/PostgreSQL/SQLite)
              |       '-- tool request_adk_risk_assessment --A2A (Agent Card + JSON-RPC)--> Google ADK agent (:8001)
              '-- guardrails . callbacks . tracing (local + Langfuse)
```

## Requirements status (honest)

| # | Requirement | Status | Proof |
|---|---|---|---|
| 1 | Context sharing / state | Met | `workflow_states` persisted every step; `test_req1_*` |
| 2 | MCP tools (>=2) | Met | 10 tools, 7+ used per run, results consumed + cross-checked, stdio transport; `test_req2_*` |
| 3 | Pydantic structured output | Met | schemas for every tool/LLM/A2A payload; Markdown PO; `test_req3_*` |
| 4 | Design patterns | Met | sequential, tool use, **reflection loop**, HITL, fail-safe, replanning; `test_req4_*` |
| 5 | Callback monitoring | Met | `MonitoringCallbackHandler` + CrewAI callbacks; `test_req5_*` |
| 6 | A2A CrewAI <-> ADK | Met | official Agent Card + JSON-RPC; `test_req6_*` |
| 7 | CrewAI and ADK only | Met | `crew.kickoff` + ADK `Runner` really execute; `test_req7_*` |
| 8 | Guardrails | Met (pattern-based) | input, tool-output, agent-output, A2A payload, ADK callbacks; `test_req8_*` |
| 9 | Observability | Met in code; **verify your Langfuse keys** | real tokens/cost; Langfuse v5 streaming verified on the wire against a mock server |
| 10 | Evaluation / cost / time | Met (deterministic measured; LLM figures are estimates until `--llm`) | `docs/PERFORMANCE.md` |
| 11 | Cloud deployment | **Config ready, not deployed** | `render.yaml`, `docker-compose.yml`; suite passes on real PostgreSQL |

**Live URL:** _add your deployed URL here after deploying (see docs/DEPLOYMENT.md)_

### Known limits (say these before you are asked)
- Only 3 of the 8 CrewAI agents call an LLM (Requirement, Negotiation, Coordination). The rest are deterministic
  by design because they handle money/policy.
- LLM agents run with **least-privilege tools**: Requirement and Negotiation hold none, Coordination holds only the
  A2A tool. Saving negotiations, approvals and POs stay in the orchestrator (a live run showed a real model
  inventing IDs when handed a DB tool). Each LLM run is also bounded by `LLM_TIMEOUT_SECONDS` (default 60).
- Live Gemini has been observed working for the Requirement agent (with reflection) and for CrewAI -> A2A -> ADK
  (ADK agent on Gemini). The Negotiation LLM draft has not been observed live yet. Everything else in LLM mode
  is tested with a scripted LLM through the real CrewAI/ADK machinery.
  Without a key the ADK agent uses a deterministic offline model (the Runner + tool loop still run).
- Cost covers CrewAI agents (token counts are approximate for Gemini in CrewAI) and the ADK agent (exact, reported
  back over `/usage/{request_id}`). Check the `model_version` in logs: Google may serve a different model than the
  one you requested, so set `LLM_PRICE_*` for the model that actually answered.
- Guardrails are regex/pattern based, not an ML classifier.
- Traces are in-memory; durable audit rows live in the database.
- The submitted PDF report and slide deck pre-date these changes; regenerate/edit them before submitting.

## Quick start
```bash
pip install -r requirements.txt
cp .env.example .env            # defaults work offline (SQLite fallback, deterministic agents)
python run.py                   # UI http://localhost:8000 , docs /docs , health /api/health
python scripts/run_demo.py      # terminal walkthrough incl. the real A2A hop
pytest tests                    # 65 tests
python scripts/evaluate_performance.py --runs 3 --estimate   # -> docs/PERFORMANCE.md
```
LLM mode: `DEMO_MODE=False`, `GROQ_API_KEY=gsk_...` (or `GEMINI_API_KEY`), then `python scripts/evaluate_performance.py --llm`.
Real MCP wire protocol: `MCP_TRANSPORT=stdio`. Whole suite on another DB: `DEALFORGE_TEST_DATABASE_URL=postgresql+psycopg2://...`.

## Documents
- `docs/VIVA_CHECKLIST.md` - deploy tonight, demo order, answers to the examiner's questions
- `docs/VIVA_MAP.md` - where each requirement lives, how to demo it, which test proves it, tricky questions
- `docs/DEPLOYMENT.md` - Render and Docker steps
- `docs/PERFORMANCE.md` - measured latency, outcomes, LLM cost estimate
