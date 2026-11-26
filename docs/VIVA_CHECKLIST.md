# Viva checklist: deploy tonight, demo tomorrow

Examiner's minimum criteria: **(1) project deployed, (2) Langfuse deployed (not local), (3) cross matrix on the UI
(cost, agent runs, run counts), (4) guardrails / MCP / A2A / containers explained.** Marks are cut without 1-3.

## A. Deploy tonight (Render + Langfuse Cloud) ~30-40 min
1. **Rotate secrets first** (Gemini key, Langfuse keys, DB password) - they appeared in chat/screenshots.
2. **Push to GitHub** (inner `DealForge-AI` folder):
   ```
   git init
   git add .
   git ls-files | findstr /i "\.env"        # MUST list only .env.example. If .env appears: git rm --cached .env
   git commit -m "DealForge AI"
   git branch -M main
   git remote add origin https://github.com/<your-user>/<repo>.git
   git push -u origin main
   ```
3. **Render:** New -> Blueprint -> pick the repo. It reads `render.yaml` and creates the web service + PostgreSQL.
   Set the secrets it asks for: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST=https://cloud.langfuse.com`,
   `LANGFUSE_PROJECT_URL` (your project page), and optionally `GEMINI_API_KEY`.
   Build takes several minutes (CrewAI + ADK are large). Measured memory is ~290 MB (free tier limit 512 MB).
4. **Verify the deployed service** (replace URL):
   - `https://<app>.onrender.com/api/health` -> `"status": "healthy"`, ADK reachable, 10 MCP tools
   - `https://<app>.onrender.com/api/observability/langfuse-status` -> `"credentials_verified": true`
   - Open the UI -> **Observability** -> click **Run sample workflows** -> cost/runs/MCP/A2A/guardrail panels fill in
   - **Langfuse Cloud** (cloud.langfuse.com) -> Tracing: the traces from the *deployed* app appear. Screenshot them.
5. Put the live URL in `README.md`.

**Langfuse "deployed":** you use Langfuse Cloud (hosted by Langfuse), not a local container. The deployed backend streams
traces to it - that is the proof. Do not demo against `localhost`.

**Free-tier reality:** Render free services sleep when idle (first request ~30-60 s) - open the URL 5 minutes before the
viva. Render's database is persistent: the cross matrix survives restarts (traces are stored in the DB).

## B. Real LLM cost on the UI (optional, mind the quota)
Default deployment runs deterministic agents -> measured cost shows **0** (honest: no LLM calls). The UI also shows a
clearly labelled **projected LLM cost per workflow** (from real prompt sizes). To show *measured* LLM cost/tokens:
set `DEMO_MODE=false` + `GEMINI_API_KEY` on Render, click **Run sample workflows** once (LLM mode runs 1 workflow ~ 6
Gemini requests; free tier ~20/day). Do this **before** the viva, not during. Set `LLM_PRICE_INPUT_PER_M` /
`LLM_PRICE_OUTPUT_PER_M` to the real price of the model that served the requests (log field `model_version`).

## C. Demo order (5-7 minutes)
1. Live URL -> Dashboard -> submit a request (e.g. "Purchase 50 laptops ... budget Rs 38,00,000") -> run workflow.
2. Approvals -> approve -> Purchase Order issued (human-in-the-loop).
3. Observability -> **System Facts** panel: runs, cost, MCP, A2A, guardrails, deployment, Langfuse.
4. Type a prompt injection ("Ignore all previous instructions...") -> blocked; the guardrail counter increases.
5. Langfuse Cloud -> trace tree of that run (agent/tool/A2A spans, tokens/cost if LLM mode).
6. `https://<adk-host>/.well-known/agent-card.json` (or the Agent Card URL shown in the A2A card).

## D. Answers to the examiner's questions
- **How many containers / services?** Render: **1 web service** (FastAPI + CrewAI agents + FastMCP + the Google ADK A2A
  server running on loopback) + **1 managed PostgreSQL**; Langfuse is **Langfuse Cloud (SaaS)**. Docker Compose:
  **3 containers** (app, ADK service, MySQL) + 2 optional (self-hosted Langfuse + its Postgres).
- **How many MCP servers / tools?** **1 server** (`DealForge-Procurement-Server`, FastMCP) exposing **10 tools**:
  search_vendors, get_vendor_details, get_vendor_quotes, get_product_inventory, get_compliance_status,
  calculate_quote_total, validate_budget, save_negotiation_proposal, create_approval_request, create_purchase_order.
  Transport: in-process by default, real stdio child process with `MCP_TRANSPORT=stdio`.
- **What is A2A?** Agent-to-Agent protocol: agents discover each other through an **Agent Card**
  (`/.well-known/agent-card.json`) and exchange messages via **JSON-RPC**. Here a CrewAI agent calls the independent
  Google ADK risk agent; if it is unreachable the deal is forced to human review (fail-safe).
- **Guardrails?** 5 layers: input (injection/toxicity/limits), tool-output type validation, agent-output validation
  (price must match), A2A payload scan + schema validation, ADK before/after callbacks. Counters are shown live in the UI.
- **Cost?** Measured = tokens x configured price; 0 in deterministic mode. Projection is labelled an estimate.
- **Why are some agents deterministic?** Money/policy must be exact: LLMs write language (3 agents), code does arithmetic.
