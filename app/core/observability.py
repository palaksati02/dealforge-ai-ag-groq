"""
Observability: tracing, provenance and cost accounting.

* Every workflow is one TRACE; every agent step / tool call / A2A exchange is a SPAN with real wall-clock
  duration, inputs/outputs and (for LLM-backed steps) REAL token usage reported by CrewAI/ADK.
* Cost = measured tokens x configured per-token price (settings.LLM_PRICE_*). Deterministic steps cost 0 and
  say so; nothing is seeded, estimated from constants, or faked.
* If Langfuse keys are configured the same trace is streamed live to Langfuse (v3+ observation API):
  root "chain" observation -> child agent/tool/guardrail observations -> generation observations with usage.
  Without keys Langfuse export is OFF and reported as such (never "synced").
* The in-memory store powers the dashboard/API; durable audit rows are written by the repository
  (`agent_event_logs`, `workflow_states`).
"""
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    LANGFUSE_AVAILABLE = False

_LF_TYPE = {"AGENT": "agent", "TOOL": "tool", "A2A": "agent", "GUARDRAIL": "guardrail", "APPROVAL": "span"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def llm_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    cost = (
        (settings.LLM_PRICE_INPUT_PER_M * prompt_tokens) + (settings.LLM_PRICE_OUTPUT_PER_M * completion_tokens)
    ) / 1_000_000
    return float(round(cost, 8))


class TraceSpan:
    """One observed step (agent turn, MCP tool call, A2A exchange, guardrail check)."""

    def __init__(self, span_id: str, trace_id: str, name: str, span_type: str,
                 parent_span_id: Optional[str] = None, inputs: Optional[Dict[str, Any]] = None,
                 framework: Optional[str] = None, integration: Optional[str] = None):
        self.span_id = span_id
        self.trace_id = trace_id
        self.name = name
        self.span_type = span_type          # AGENT | TOOL | A2A | GUARDRAIL | APPROVAL
        self.parent_span_id = parent_span_id
        self.start_time = time.perf_counter()
        self.started_at = _now_iso()
        self.end_time: Optional[float] = None
        self.duration_ms: float = 0.0
        self.inputs = inputs or {}
        self.outputs: Optional[Dict[str, Any]] = None
        self.status = "RUNNING"
        self.error: Optional[str] = None
        self.mode = "deterministic"         # "llm" when real model tokens were consumed
        self.framework = framework or ("Google ADK" if "ADK" in name else "CrewAI")
        self.integration = integration or {"TOOL": "MCP tool", "A2A": "A2A protocol"}.get(span_type, "Agent step")
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.estimated_cost_usd = 0.0
        self._lf = None                      # live Langfuse observation (if exporting)

    def finish(self, outputs: Optional[Dict[str, Any]] = None, status: str = "SUCCESS",
               error: Optional[str] = None, prompt_tokens: int = 0, completion_tokens: int = 0,
               mode: Optional[str] = None) -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        self.outputs = outputs
        self.status = status
        self.error = error
        self.token_usage = {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens) + int(completion_tokens),
        }
        self.estimated_cost_usd = llm_cost_usd(prompt_tokens, completion_tokens) if (prompt_tokens or completion_tokens) else 0.0
        self.mode = mode or ("llm" if self.token_usage["total_tokens"] > 0 else "deterministic")
        tracer._finish_langfuse(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id, "trace_id": self.trace_id, "parent_span_id": self.parent_span_id,
            "name": self.name, "span_type": self.span_type, "framework": self.framework,
            "integration": self.integration, "mode": self.mode, "status": self.status,
            "started_at": self.started_at, "duration_ms": round(self.duration_ms, 3),
            "token_usage": self.token_usage,
            "estimated_cost_usd": self.estimated_cost_usd,
            "estimated_cost_inr": round(self.estimated_cost_usd * float(settings.USD_INR_RATE), 6),
            "inputs": self.inputs, "outputs": self.outputs, "error": self.error,
        }


class ObservabilityTracer:
    def __init__(self):
        self._lock = threading.RLock()
        self.traces: Dict[str, List[TraceSpan]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._roots: Dict[str, Any] = {}
        self._history: Dict[str, Dict[str, Any]] = {}   # traces loaded from the DB (previous runs / restarts)
        self.langfuse_client = None
        self._init_langfuse()

    # ------------------------------------------------------------------ Langfuse
    def _init_langfuse(self) -> None:
        if not LANGFUSE_AVAILABLE or not settings.langfuse_configured:
            logger.info("[Langfuse] export OFF (no LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY configured). "
                        "Local tracing stays fully active.")
            return
        try:
            self.langfuse_client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY, secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
            logger.info(f"[Langfuse] client created for {settings.LANGFUSE_HOST} (use /api/observability/langfuse-status to verify credentials).")
        except Exception as e:
            self.langfuse_client = None
            logger.warning(f"[Langfuse] could not create client: {e}. Export disabled.")

    def langfuse_status(self, verify: bool = False) -> Dict[str, Any]:
        info = {"host": settings.LANGFUSE_HOST, "keys_configured": settings.langfuse_configured,
                "sdk_installed": LANGFUSE_AVAILABLE, "exporting": self.langfuse_client is not None,
                "credentials_verified": None}
        if verify and self.langfuse_client is not None:
            try:
                info["credentials_verified"] = bool(self.langfuse_client.auth_check())
            except Exception as e:
                info["credentials_verified"] = False
                info["error"] = str(e)[:200]
        return info

    def _finish_langfuse(self, span: TraceSpan) -> None:
        obs = span._lf
        if obs is None:
            return
        try:
            obs.update(output=span.outputs, level="ERROR" if span.status == "ERROR" else "DEFAULT",
                       status_message=span.error or span.status,
                       metadata={"duration_ms": span.duration_ms, "mode": span.mode, "framework": span.framework})
            if span.token_usage["total_tokens"] > 0:
                gen = obs.start_observation(
                    name=f"{span.name} (LLM usage)", as_type="generation", model=settings.llm_model_name,
                    usage_details={"input": span.token_usage["prompt_tokens"], "output": span.token_usage["completion_tokens"]},
                    cost_details={"total": span.estimated_cost_usd},
                )
                gen.end()
            obs.end()
        except Exception as e:
            logger.debug(f"[Langfuse] span export error: {e}")

    # ------------------------------------------------------------------ trace API
    def start_trace(self, trace_id: str, workflow_id: str, request_id: str, prompt: str) -> str:
        with self._lock:
            self.traces[trace_id] = []
            self.metadata[trace_id] = {"workflow_id": workflow_id, "request_id": request_id,
                                       "start_time": _now_iso(), "prompt": prompt, "status": "RUNNING",
                                       "langfuse": "disabled"}
        if self.langfuse_client is not None:
            try:
                lf_id = self.langfuse_client.create_trace_id(seed=trace_id)
                root = self.langfuse_client.start_observation(
                    trace_context={"trace_id": lf_id}, name="DealForge_Procurement_Workflow", as_type="chain",
                    input={"prompt": prompt}, metadata={"workflow_id": workflow_id, "request_id": request_id},
                )
                self._roots[trace_id] = root
                self.metadata[trace_id]["langfuse"] = "streaming"
                self.metadata[trace_id]["langfuse_trace_id"] = lf_id
            except Exception as e:
                logger.warning(f"[Langfuse] could not open trace: {e}")
        logger.info(f"[Tracer] started trace {trace_id} for request {request_id}")
        return trace_id

    def annotate(self, trace_id: str, **kwargs: Any) -> None:
        with self._lock:
            self.metadata.setdefault(trace_id, {}).update(kwargs)

    def start_span(self, trace_id: str, name: str, span_type: str, parent_span_id: Optional[str] = None,
                   inputs: Optional[Dict[str, Any]] = None, framework: Optional[str] = None,
                   integration: Optional[str] = None) -> TraceSpan:
        span = TraceSpan(str(uuid.uuid4())[:8], trace_id, name, span_type, parent_span_id, inputs, framework, integration)
        with self._lock:
            self.traces.setdefault(trace_id, []).append(span)
        root = self._roots.get(trace_id)
        if root is not None:
            try:
                span._lf = root.start_observation(name=name, as_type=_LF_TYPE.get(span_type, "span"), input=inputs)
            except Exception as e:
                logger.debug(f"[Langfuse] span open error: {e}")
        return span

    def end_trace(self, trace_id: str, status: str = "COMPLETED") -> Dict[str, Any]:
        with self._lock:
            meta = self.metadata.setdefault(trace_id, {})
            meta["status"] = status
            meta["end_time"] = _now_iso()
        root = self._roots.pop(trace_id, None)
        if root is not None and self.langfuse_client is not None:
            try:
                summary = self.get_trace_summary(trace_id)
                root.update(output={"status": status, "total_tokens": summary["tokens"]["total_tokens"],
                                    "cost_usd": summary["financial_cost"]["estimated_cost_usd"]})
                root.end()
                self.langfuse_client.flush()
                meta["langfuse"] = "submitted"   # handed to the SDK; delivery is confirmed in the Langfuse UI
            except Exception as e:
                meta["langfuse"] = f"error: {str(e)[:80]}"
                logger.warning(f"[Langfuse] trace submission error: {e}")
        self.persist(trace_id)
        return self.get_trace_summary(trace_id)

    def has_trace(self, trace_id: str) -> bool:
        return trace_id in self.traces or trace_id in self._history

    def persist(self, trace_id: str) -> None:
        """Upsert the trace into the database (never raises: observability must not break a workflow)."""
        try:
            from app.db.session import SessionLocal
            from app.db.models.entities import TraceRecord
            summary = json.loads(json.dumps(self.get_trace_summary(trace_id), default=str))
            meta = json.loads(json.dumps(self.metadata.get(trace_id, {}), default=str))
            db = SessionLocal()
            try:
                rec = db.get(TraceRecord, trace_id)
                if rec is None:
                    rec = TraceRecord(trace_id=trace_id)
                    db.add(rec)
                rec.workflow_id, rec.request_id = summary.get("workflow_id"), summary.get("request_id")
                rec.status, rec.summary, rec.meta = summary.get("status", "RUNNING"), summary, meta
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"[Tracer] could not persist trace {trace_id}: {e}")

    def load_from_db(self, limit: int = 1000) -> int:
        """Load previously persisted traces so the dashboard shows history after a restart."""
        try:
            from app.db.session import SessionLocal
            from app.db.models.entities import TraceRecord
            db = SessionLocal()
            try:
                rows = db.query(TraceRecord).order_by(TraceRecord.created_at.desc()).limit(limit).all()
                with self._lock:
                    for rec in reversed(rows):                       # oldest first
                        if rec.trace_id not in self.traces:
                            self._history[rec.trace_id] = rec.summary or {}
                            self.metadata.setdefault(rec.trace_id, rec.meta or {})
                logger.info(f"[Tracer] loaded {len(rows)} persisted trace(s) from the database.")
                return len(rows)
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"[Tracer] could not load persisted traces: {e}")
            return 0

    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        if trace_id not in self.traces and trace_id in self._history:
            return self._history[trace_id]
        with self._lock:
            spans = list(self.traces.get(trace_id, []))
            meta = dict(self.metadata.get(trace_id, {}))
        prompt_t = sum(s.token_usage["prompt_tokens"] for s in spans)
        compl_t = sum(s.token_usage["completion_tokens"] for s in spans)
        cost = sum(s.estimated_cost_usd for s in spans)
        return {
            "trace_id": trace_id, "workflow_id": meta.get("workflow_id"), "request_id": meta.get("request_id"),
            "status": meta.get("status", "RUNNING"), "langfuse": meta.get("langfuse", "disabled"),
            "total_spans": len(spans),
            "total_duration_ms": round(sum(s.duration_ms for s in spans), 2),
            "total_duration_seconds": round(sum(s.duration_ms for s in spans) / 1000.0, 3),
            "tokens": {"prompt_tokens": prompt_t, "completion_tokens": compl_t, "total_tokens": prompt_t + compl_t},
            "financial_cost": {"estimated_cost_usd": round(cost, 8),
                               "estimated_cost_inr": round(cost * float(settings.USD_INR_RATE), 6)},
            "spans": [s.to_dict() for s in spans],
        }

    def list_all_traces(self) -> List[Dict[str, Any]]:
        with self._lock:
            live = list(self.traces.keys())
            hist = [t for t in self._history.keys() if t not in self.traces]
        return [self.get_trace_summary(t) for t in hist + live]

    def reset(self) -> None:
        with self._lock:
            self.traces.clear(); self.metadata.clear(); self._roots.clear(); self._history.clear()

    def export_to_langfuse(self, trace_id: str) -> Dict[str, Any]:
        """Trace rendered in the Langfuse ingestion schema (inspection/offline import)."""
        summary = self.get_trace_summary(trace_id)
        meta = self.metadata.get(trace_id, {})
        observations = []
        for s in summary.get("spans", []):
            tok = s.get("token_usage", {})
            observations.append({
                "id": s.get("span_id"), "traceId": trace_id,
                "type": "GENERATION" if tok.get("total_tokens", 0) > 0 else "SPAN",
                "name": s.get("name"), "startTime": s.get("started_at"), "input": s.get("inputs"), "output": s.get("outputs"),
                "statusMessage": s.get("status"), "level": "ERROR" if s.get("status") == "ERROR" else "DEFAULT",
                "usage": {"input": tok.get("prompt_tokens", 0), "output": tok.get("completion_tokens", 0),
                          "total": tok.get("total_tokens", 0), "unit": "TOKENS"},
                "calculatedTotalCost": s.get("estimated_cost_usd", 0.0),
                "metadata": {"span_type": s.get("span_type"), "framework": s.get("framework"), "mode": s.get("mode"),
                             "duration_ms": s.get("duration_ms")},
            })
        return {
            "id": trace_id, "name": f"DealForge_Procurement_Workflow_{str(summary.get('workflow_id') or 'trace')[:8]}",
            "timestamp": meta.get("start_time"), "input": {"prompt": meta.get("prompt")},
            "output": {"status": summary.get("status")},
            "metadata": {"workflow_id": summary.get("workflow_id"), "request_id": summary.get("request_id"),
                         "frameworks": ["CrewAI", "Google ADK"], "protocols": ["MCP", "A2A"],
                         "total_cost_usd": summary.get("financial_cost", {}).get("estimated_cost_usd", 0.0)},
            "tags": ["dealforge", "crewai", "google-adk", "mcp", "a2a"],
            "observations": observations,
        }

    # ------------------------------------------------------------------ dashboard (all computed)
    def get_cross_matrix(self) -> Dict[str, Any]:
        from app.core.guardrails import SecurityGuardrails
        traces = self.list_all_traces()
        rate = float(settings.USD_INR_RATE)

        by_agent: Dict[str, Dict[str, Any]] = {}
        for t in traces:
            for sp in t["spans"]:
                a = by_agent.setdefault(sp["name"], {
                    "agent_name": sp["name"], "role": sp["span_type"], "framework": sp["framework"],
                    "integration": sp["integration"], "executions": 0, "_lat": 0.0, "prompt_tokens": 0,
                    "completion_tokens": 0, "cost_usd": 0.0, "errors": 0, "llm_runs": 0})
                a["executions"] += 1
                a["_lat"] += sp["duration_ms"]
                a["prompt_tokens"] += sp["token_usage"]["prompt_tokens"]
                a["completion_tokens"] += sp["token_usage"]["completion_tokens"]
                a["cost_usd"] += sp["estimated_cost_usd"]
                a["errors"] += 1 if sp["status"] == "ERROR" else 0
                a["llm_runs"] += 1 if sp["mode"] == "llm" else 0
        gstats = SecurityGuardrails.get_stats()
        agent_matrix = []
        for a in by_agent.values():
            tot = a["prompt_tokens"] + a["completion_tokens"]
            agent_matrix.append({
                "agent_name": a["agent_name"], "role": f"{a['role']} ({'LLM' if a['llm_runs'] else 'deterministic'})",
                "framework": a["framework"], "integration": a["integration"], "executions": a["executions"],
                "avg_latency_ms": round(a["_lat"] / a["executions"], 3), "prompt_tokens": a["prompt_tokens"],
                "completion_tokens": a["completion_tokens"], "total_tokens": tot,
                "cost_usd": round(a["cost_usd"], 8), "cost_inr": round(a["cost_usd"] * rate, 6),
                "guardrail_status": f"{a['errors']} error(s) / {a['executions']} run(s)",
            })

        scenarios = []
        for t in traces:
            m = self.metadata.get(t["trace_id"], {})
            scenarios.append({
                "scenario_id": t["trace_id"][:8], "name": (m.get("prompt") or "")[:70],
                "category": m.get("category", "-"), "quantity": m.get("quantity", 0),
                "budget_inr": float(m.get("budget", 0) or 0),
                "agents_invoked": sum(1 for s in t["spans"] if s["span_type"] in ("AGENT", "A2A")),
                "duration_ms": t["total_duration_ms"], "total_tokens": t["tokens"]["total_tokens"],
                "cost_inr": t["financial_cost"]["estimated_cost_inr"], "result": t["status"],
            })

        total_cost = sum(t["financial_cost"]["estimated_cost_usd"] for t in traces)
        total_tokens = sum(t["tokens"]["total_tokens"] for t in traces)
        total_ms = sum(t["total_duration_ms"] for t in traces)
        n = len(traces)
        all_spans = [sp for t in traces for sp in t["spans"]]
        llm_runs = sum(1 for sp in all_spans if sp["mode"] == "llm")
        a2a_spans = [sp for sp in all_spans if sp["span_type"] == "A2A"]
        mcp_calls = sum(int((self.metadata.get(t["trace_id"], {}) or {}).get("mcp_calls", 0) or 0) for t in traces)
        status_counts: Dict[str, int] = {}
        for t in traces:
            status_counts[t["status"]] = status_counts.get(t["status"], 0) + 1
        return {
            "summary": {
                "total_workflows_executed": n,
                "workflow_status_counts": status_counts,
                "total_agent_runs": sum(a["executions"] for a in agent_matrix),
                "llm_agent_runs": llm_runs,
                "deterministic_agent_runs": len(all_spans) - llm_runs,
                "total_tokens_consumed": total_tokens,
                "total_cost_usd": round(total_cost, 8), "total_cost_inr": round(total_cost * rate, 6),
                "avg_latency_ms": round(total_ms / n, 2) if n else 0.0,
                "avg_latency_seconds": round(total_ms / n / 1000.0, 3) if n else 0.0,
                "guardrail_status": f"{gstats['violations']} violation(s) blocked in {gstats['checks']} check(s)",
                "guardrail_stats": gstats,
                "llm_mode": settings.llm_agents_enabled,
                "pricing": {"model": settings.llm_model_name, "provider": settings.llm_provider, "usd_per_m_input": float(settings.LLM_PRICE_INPUT_PER_M),
                            "usd_per_m_output": float(settings.LLM_PRICE_OUTPUT_PER_M)},
                "active_frameworks": ["CrewAI", "Google ADK"],
            },
            "agent_cross_matrix": agent_matrix,
            "scenario_cross_matrix": scenarios,
            "deployment_topology": self._topology(),
            "langfuse": {**self.langfuse_status(), "project_url": os.environ.get("LANGFUSE_PROJECT_URL") or settings.LANGFUSE_HOST,
                         "traces_streamed": sum(1 for t in traces if (self.metadata.get(t["trace_id"], {}) or {}).get("langfuse") in ("submitted", "streaming"))},
            "mcp": self._mcp_block(mcp_calls),
            "a2a": {
                "protocol": "A2A (official a2a-sdk): Agent Card discovery + JSON-RPC message/send",
                "client_agent": "CrewAI Approval/Coordination agent (tool: request_adk_risk_assessment)",
                "server_agent": "Google ADK GoogleADK_ProcurementRiskAgent (served via ADK to_a2a)",
                "agents_connected": 2, "exchanges_total": len(a2a_spans),
                "exchanges_success": sum(1 for sp in a2a_spans if sp["status"] == "SUCCESS"),
                "exchanges_failsafe": sum(1 for sp in a2a_spans if sp["status"] != "SUCCESS"),
                "agent_card_url": f"{settings.adk_service_url}/.well-known/agent-card.json",
            },
            "guardrails": {
                **gstats,
                "layers": [
                    "Input: prompt-injection + toxicity filter, domain limits (qty 1-10,000, budget bounds)",
                    "Tool output: strict Pydantic type validation of every MCP result",
                    "Agent output: LLM JSON validated + price/number guardrails (CrewAI task guardrails)",
                    "A2A: payload injection scan out, strict schema validation in; fail-safe to human review",
                    "ADK callbacks: before_model (injection), before/after_tool (type checks)",
                ],
            },
            "deployment": self._deployment_info(),
            "projection": self._projection(),
        }

    def _mcp_block(self, calls_recorded: int) -> Dict[str, Any]:
        try:
            from app.mcp.client import mcp_client
            tools = mcp_client.discover_tools()
            transport = mcp_client.transport
        except Exception:
            tools, transport = [], settings.MCP_TRANSPORT
        return {"servers": 1, "server_name": "DealForge-Procurement-Server (FastMCP)", "transport": transport,
                "tools_count": len(tools), "tools": tools, "tool_calls_recorded": calls_recorded}

    @staticmethod
    def _deployment_info() -> Dict[str, Any]:
        if os.environ.get("RENDER"):
            platform = "Render (cloud)"
        elif os.path.exists("/.dockerenv"):
            platform = "Docker container"
        else:
            platform = "Local process"
        return {"platform": platform, "public_url": os.environ.get("RENDER_EXTERNAL_URL"),
                "environment": settings.APP_ENV, "database": __import__("app.db.session", fromlist=["engine"]).engine.dialect.name}

    @staticmethod
    def _projection() -> Optional[Dict[str, Any]]:
        """LLM cost PROJECTION per workflow from the evaluation script's estimate (real CrewAI prompt sizes).
        Clearly separate from measured cost; shown so the UI is informative even in deterministic demo mode."""
        try:
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            with open(os.path.join(root, "docs", "performance_evaluation.json"), encoding="utf-8") as f:
                est = json.load(f).get("llm_estimate", {}).get("per_scenario", [])
            if not est:
                return None
            n = len(est)
            usd = sum(e["est_cost_usd"] for e in est) / n
            tok = sum(e["est_prompt_tokens"] + e["est_completion_tokens"] for e in est) / n
            return {"label": "ESTIMATE (not measured): LLM-mode cost per workflow, from real prompt sizes",
                    "avg_cost_usd": round(usd, 6), "avg_cost_inr": round(usd * float(settings.USD_INR_RATE), 4),
                    "avg_tokens": round(tok), "based_on_scenarios": n,
                    "price_note": "uses configured LLM_PRICE_* values"}
        except Exception:
            return None

    def _topology(self) -> Dict[str, Any]:
        """Live checks only. Nothing is labelled 'deployed' unless it actually answered."""
        import httpx
        from app.db.session import check_db_connection
        adk_ok = False
        try:
            adk_ok = httpx.get(f"{settings.adk_service_url}/.well-known/agent-card.json", timeout=1.5).status_code == 200
        except Exception:
            pass
        try:
            from app.mcp.client import mcp_client
            tools = mcp_client.discover_tools()
        except Exception:
            tools = []
        return {
            "components": [
                {"name": "FastAPI app (CrewAI engine + UI)", "status": "UP (this process)"},
                {"name": "Google ADK A2A service", "endpoint": settings.adk_service_url,
                 "status": "REACHABLE" if adk_ok else "UNREACHABLE"},
                {"name": "Database", "status": "CONNECTED" if check_db_connection() else "DISCONNECTED"},
                {"name": "Langfuse", "status": "EXPORTING" if self.langfuse_client else "DISABLED (no keys)"},
            ],
            "mcp_transport": settings.MCP_TRANSPORT, "mcp_tools_count": len(tools), "mcp_tools": tools,
            "environment": settings.APP_ENV,
        }


tracer = ObservabilityTracer()
