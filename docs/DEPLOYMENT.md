# Deployment

## A. Render (managed cloud) - `render.yaml`
1. Push the repo to GitHub (`.env` is git-ignored; never commit it).
2. Render dashboard -> **New -> Blueprint** -> select the repo. It creates the web service and a PostgreSQL database
   and injects `DATABASE_URL` automatically.
3. Fill the `sync: false` secrets: `GEMINI_API_KEY` (optional), `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`.
4. Open `https://<service>.onrender.com/api/health` - expect `"status": "healthy"`, `adk_a2a_service.reachable: true`,
   `mcp.tools_discovered: 10`. Put that URL in the README.
Notes: the free tier sleeps when idle (first request is slow; `ADK_TIMEOUT_SECONDS=30`). `APP_ENV=production` makes the
app **fail loudly** if the database is unreachable instead of silently using SQLite. The ADK agent runs in the same
service on loopback (real HTTP). The suite was run against a real PostgreSQL; the Render deployment itself was not.

## B. Docker Compose (separate ADK container, MySQL) - `docker-compose.yml`
```bash
cp .env.example .env        # set DB_PASSWORD, optional GEMINI_API_KEY ...
docker compose up --build -d
# optional self-hosted Langfuse
# add LANGFUSE_DB_PASSWORD, LANGFUSE_NEXTAUTH_SECRET, LANGFUSE_SALT to .env
docker compose --profile observability up -d
# open http://localhost:3000, create a project, copy its keys into .env, then: docker compose up -d app
```
Services: `app` (:8000), `adk_service` (:8001, serves `/.well-known/agent-card.json`), `mysql`.

## C. Enabling the LLM-backed agents
`DEMO_MODE=False` and `GEMINI_API_KEY=...` (or `USE_LLM_AGENTS=true`). Use a current model in `GEMINI_MODEL`
(`gemini-1.5-pro` and `gemini-2.5-flash` are retired for new users; list what your key can use and test it with a one-line call). Set `LLM_PRICE_*` for the model that actually serves requests (log field `model_version`) and `LLM_TIMEOUT_SECONDS` to bound slow calls. Then `python scripts/evaluate_performance.py --llm` measures real latency/tokens/cost.

## D. Verify
`pytest tests` (65 tests) | `python scripts/evaluate_performance.py --runs 3 --estimate` | `/api/health`
