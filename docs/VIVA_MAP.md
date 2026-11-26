# Viva map: "show me where you did X"

For every requirement: **where** it is, how to **show it live**, which **test proves it**, and the
honest **limits** (examiners respect candour; they punish claims the code does not support).

Line numbers are as of this version; use the function names if they drift.

---

## 1. Context sharing & state updates
- **Where:** every step in `ProcurementWorkflowEngine._execute_workflow_inner` (`app/agents/procurement_crew.py`) calls
  `repo.save_workflow_state(...)` (`app/db/repositories/procurement_repo.py:341`, table `workflow_states`).
  The winner chosen by Deal Evaluation feeds Budget -> Negotiation -> A2A payload (`RiskAssessmentPayload`).
  Human pause/resume restores state in `resume_workflow`.
- **Show:** run `python scripts/run_demo.py`, then `GET /api/workflows/{id}` / the timeline in the UI.
- **Proof:** `test_req1_state_is_persisted_and_chained`, `test_req1_overrides_survive_the_workflow`.
- **Limit:** shared state is the DB + Python objects; it is not CrewAI's internal task-context memory
  (our focused crews are single-task).

## 2. Tool integration via MCP
- **Where:** FastMCP server `app/mcp/server.py` (10 `@mcp.tool()`s). Client `app/mcp/client.py`:
  `MCPClient.call_tool` (:230), `_StdioMCPTransport` (:98) spawns the server as a **child process** over stdio.
  `TOOL_OUTPUT_SCHEMAS` (:43) = strict output contract per tool.
- **The engine uses the returned values** (not discarded): vendors, quotes, compliance, budget, negotiation save,
  approval creation and PO creation all come from MCP results; local services are only an independent cross-check
  that raises `MCP_LOCAL_PRICE_MISMATCH` / `MCP_LOCAL_BUDGET_MISMATCH`.
- **Show:** `MCP_TRANSPORT=stdio` then run a workflow; `GET /api/observability/mcp/invocations` lists every call.
- **Proof:** `test_req2_mcp_output_is_consumed_and_cross_checked` (tampering with a tool changes the outcome),
  `test_req2_real_mcp_wire_protocol_over_stdio`, `test_req2_workflow_uses_at_least_two_mcp_tools`.
- **Limit:** the tools are database lookups and calculators (allowed by the brief), not a web-search API.
  Default transport is in-process (fast, used by tests); stdio is one env var away.

## 3. Structured output with Pydantic
- **Where:** `app/schemas/schemas.py` (requirement extract, quote breakdown, budget/compliance results, tool
  contracts, `NegotiationDraft`, `ReflectionVerdict`), `app/a2a/schemas.py` (strict `Literal` + `0..100` bounds),
  Markdown PO in `app/services/purchase_order_service.py`. CrewAI tasks use `output_pydantic=`.
- **Proof:** `test_req3_*`, `test_req8_typed_tool_output_validation`.

## 4. Agentic design patterns
| Pattern | Where |
|---|---|
| Sequential pipeline | `Process.sequential` in `app/agents/crew.py`; ordered steps in the engine |
| Tool use | MCP tools + the A2A tool (`app/agents/crew_tools.py`) |
| **Reflection / self-correction** | `LLMAgentRunner.run_requirement_agent` + `find_reflection_issues` (`app/agents/llm_agents.py:102`): critic checks consistency with the deterministic parser and that numbers are grounded in the text; on failure it retries once with feedback, then falls back |
| Human-in-the-loop | `PAUSED` state, `ApprovalService`, `resume_workflow`, `/api/approvals` |
| Fail-safe | A2A client returns HIGH-risk/human-review on any failure |
| Dynamic replanning | `app/services/replanning_service.py`, `app/api/routes/replanning.py` |
- **Proof:** `test_req4_reflection_self_correction_loop`, `..._falls_back_when_llm_keeps_failing`,
  `test_req4_human_in_the_loop_pause_and_resume`, `test_req4_compliance_gate_refuses_when_no_vendor_qualifies`.

## 5. Task monitoring via callbacks
- **Where:** `MonitoringCallbackHandler` (`app/core/callbacks.py:47`): `on_step_start`, `on_intermediate_output`,
  `on_step_complete`, `on_error` (:136). Wired into every engine step, MCP calls, and CrewAI's `task_callback`
  / `step_callback` (`LLMAgentRunner._kickoff`). Errors in `execute_workflow` are caught, logged via `on_error`,
  and the trace is closed as `FAILED`. Durable audit rows: `agent_event_logs`.
- **Show:** `GET /api/observability/events?workflow_id=...`.
- **Proof:** `test_req5_callbacks_capture_flow_and_errors`.

## 6. Agent-to-Agent (A2A), CrewAI <-> ADK
- **Where (client / CrewAI side):** `app/a2a/client.py` - `A2AClient._exchange` (:127): discovers the **Agent Card**
  (`A2ACardResolver`), builds an `a2a-sdk` client, sends JSON-RPC `message/send`, strictly validates the reply.
  The CrewAI Coordination agent owns the tool `request_adk_risk_assessment` (`app/agents/crew_tools.py:22`).
- **Where (server / ADK side):** `app/adk_service/server.py` - `build_a2a_app` uses ADK's `to_a2a` (:50);
  `app/adk_service/agent.py` - `LlmAgent` + `FunctionTool` + callbacks, executed by ADK's `Runner`.
- **Show:** open `http://127.0.0.1:8001/.well-known/agent-card.json`; run a workflow and read the `[A2A]` log lines
  (discover -> request -> response); `GET /api/observability/a2a/exchanges`.
- **Proof:** `test_req6_agent_card_is_published`, `test_req6_crewai_to_adk_over_official_a2a`,
  `test_req6_crewai_agent_invokes_a2a_tool_which_reaches_adk` (CrewAI ReAct loop -> tool -> A2A -> ADK),
  `test_req6_failsafe_when_agent_unreachable`.
- **Limit:** without a Gemini key the ADK agent uses `OfflineRiskModel` (a deterministic `BaseLlm` that emits the
  tool call), so the Runner/tool loop is real but no LLM reasoning happens. Say this plainly.

## 7. CrewAI **and** ADK, nothing else
- **CrewAI really executes:** `crew.kickoff` at `app/agents/llm_agents.py:196`, for 3 agents (Requirement, Negotiation,
  Coordination). All 8 agents are defined in `build_agents` (`app/agents/crew.py:31`) with their MCP tools.
- **ADK really executes:** `Runner.run_async` in `ADKProcurementRiskAgent.assess` (`agent.py:320`) and inside the A2A server.
- **Least privilege:** `_kickoff(allowed_tools=...)` in `llm_agents.py` - Requirement/Negotiation hold NO tools, Coordination only the A2A tool (`test_llm_agents_get_least_privilege_tools`). A live run showed a model hallucinating a request_id when handed a DB tool, so persistence and PO creation stay in the orchestrator. Runs are bounded by `LLM_TIMEOUT_SECONDS` with deterministic fallback (`test_slow_llm_times_out_and_falls_back`).
- **Why only 3 LLM agents?** Pricing, GST, budget, compliance and PO issuance are money/policy: deterministic
  MCP tools on purpose ("LLMs for language, code for arithmetic"). That is a design decision, not a gap.
- **Proof:** `test_req7_crewai_kickoff_runs_all_three_llm_agents`, `test_req7_adk_runner_executes_agent_tool_loop`,
  `test_req7_only_crewai_and_adk_frameworks_in_requirements`.
- **Limit:** LLM mode is verified with a scripted LLM driving the real CrewAI/ADK machinery. It has **not** been run
  against live Gemini from the build environment - run `python scripts/evaluate_performance.py --llm` with your key.

## 8. Guardrails & validation callbacks
- **Where:** `app/core/guardrails.py`: `validate_user_prompt` (:117, injection + toxicity), `validate_tool_output`
  (:187, null/error/**strict Pydantic** per tool), `scan_payload` (:105, injection inside agent-to-agent data),
  `validate_agent_output` (:239). Task-level guardrails on CrewAI tasks (price must be stated exactly; JSON must
  validate). ADK callbacks `guardrail_before_model` / `validate_before_tool` / `validate_after_tool` (`agent.py:227`).
  Real counters: `SecurityGuardrails.get_stats()` feeds the dashboard.
- **Proof:** `test_req8_*` (typed output, A2A injection blocked, ADK callback blocks, limits, LLM invents a price).
- **Limit:** pattern-based (regex), not an ML classifier; it is documented, tested, and excludes benign phrases
  like "attack surface".

## 9. Observability / provenance
- **Where:** `app/core/observability.py`: one trace per workflow, spans per step with **real** durations and token
  usage (from `crew.usage_metrics`); cost = tokens x `settings.LLM_PRICE_*` (`llm_cost_usd`). Langfuse v5
  observation API streams live when keys exist (`_finish_langfuse`, :136). Provenance endpoints: `/traces`,
  `/events`, `/a2a/exchanges`, `/mcp/invocations`. `get_cross_matrix` (:272) is computed from traces; topology uses live checks.
- **Show:** with keys set, the trace tree appears in Langfuse; `GET /api/observability/langfuse-status` verifies credentials.
- **ADK-side tokens:** the ADK agent's real usage is captured by `record_usage_after_model` (`agent.py`), exposed at `GET /usage/{request_id}` and merged into the A2A span (`test_adk_agent_token_usage_flows_through_a2a_into_trace_and_cost`).
- **Proof:** `test_req9_*`, `test_spans_and_token_usage_reach_langfuse_ingestion` (wire-level, mock server).
- **Limit:** verified against a mock ingestion endpoint, not a live Langfuse server. Use `/langfuse-status` to confirm yours.
  Traces are in memory (lost on restart); durable audit rows are in the DB.

## 10. Evaluation, cost and running time
- **Where:** `scripts/evaluate_performance.py` -> `docs/PERFORMANCE.md` + `docs/performance_evaluation.json`.
  8 scenarios x N runs, warm-up excluded, mean/min/max/p95, per-step breakdown, outcome checks (including the
  injection block, compliance refusal and A2A-down fail-safe). `--estimate` adds an LLM cost estimate from the real
  CrewAI prompts; `--llm` measures real tokens/latency with a key.
- **Be precise:** in deterministic mode LLM cost is genuinely 0; the INR/USD LLM figures are labelled **estimates**
  until you run `--llm`. Prices are config (`LLM_PRICE_*`) - confirm against Google's current price list.

## 11. Cloud deployment
- **Where:** `render.yaml` (web service + managed PostgreSQL), `Dockerfile`, `docker-compose.yml` (app, **separate**
  ADK container, MySQL, optional Langfuse). `docs/DEPLOYMENT.md` has the steps.
- **Verified here:** the whole suite passes on a real PostgreSQL; both YAML files parse.
- **Not verified here:** an actual Render/Docker deployment (no cloud account/Docker in the build environment).
  Requirement 11 is only met once **you** deploy and put the live URL in the README.

---

## Likely tricky questions
- *"Is this the real A2A protocol?"* Yes: Agent Card discovery + JSON-RPC `message/send` using the official `a2a-sdk`
  on both sides (server via ADK's `to_a2a`).
- *"What if the ADK service is down?"* Fail-safe: HIGH risk, human review forced, never an auto-approval (`test_req6_failsafe...`).
- *"How do you stop an LLM inventing prices?"* It never computes them: numbers come from MCP; the negotiation task
  guardrail rejects any draft that does not state the exact computed price; reflection grounds extracted numbers in the text.
- *"Why are tokens zero?"* Deterministic mode makes no LLM calls. Set `DEMO_MODE=False` + `GEMINI_API_KEY` for real usage.
