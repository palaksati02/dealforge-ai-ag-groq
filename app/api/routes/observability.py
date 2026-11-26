from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.observability import tracer
from app.core.callbacks import workflow_monitor
from app.core.guardrails import SecurityGuardrails, GuardrailViolationException

router = APIRouter(prefix="/observability", tags=["Observability & Tracing"])


class GuardrailCheckRequest(BaseModel):
    prompt: str
    quantity: Optional[int] = None
    budget: Optional[float] = None


@router.get("/traces", response_model=List[Dict[str, Any]])
def list_traces():
    """Retrieve full observability execution traces with latency, token usage, and provenance."""
    return tracer.list_all_traces()


@router.get("/traces/{trace_id}", response_model=Dict[str, Any])
def get_trace_details(trace_id: str):
    """Retrieve detailed execution trace for a specific multi-agent workflow."""
    if not tracer.has_trace(trace_id):
        raise HTTPException(status_code=404, detail="Trace ID not found")
    return tracer.get_trace_summary(trace_id)


@router.get("/traces/{trace_id}/langfuse", response_model=Dict[str, Any])
def export_trace_to_langfuse(trace_id: str):
    """Export execution trace in standard Langfuse OSS / Cloud ingestion schema."""
    if not tracer.has_trace(trace_id):
        raise HTTPException(status_code=404, detail="Trace ID not found")
    return tracer.export_to_langfuse(trace_id)


@router.get("/events", response_model=List[Dict[str, Any]])
def list_callback_events(workflow_id: Optional[str] = None):
    """Retrieve monitoring events recorded via callback functions."""
    return workflow_monitor.get_events(workflow_id)


@router.get("/cross-matrix")
def get_cross_matrix():
    """
    Comprehensive multi-agent performance cross-matrix, token metrics, financial costs,
    and cloud deployment topology for the UI dashboard.
    """
    return tracer.get_cross_matrix()


@router.get("/langfuse-status")
@router.post("/sync-langfuse")
def langfuse_status_endpoint():
    """Honest Langfuse status. Verifies credentials against the Langfuse server when keys are configured.
    Traces are streamed live while workflows run; they cannot be 'synced' retroactively."""
    status = tracer.langfuse_status(verify=True)
    per_trace = {t_id: m.get("langfuse", "disabled") for t_id, m in tracer.metadata.items()}
    return {"langfuse": status, "traces": per_trace,
            "note": "Export is OFF unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set."}


@router.get("/a2a/exchanges")
def a2a_exchanges():
    """Provenance log of agent-to-agent exchanges (agent card, endpoint, status, latency)."""
    from app.a2a.client import A2AClient
    return A2AClient.exchanges


@router.get("/mcp/invocations")
def mcp_invocations():
    """Audit trail of MCP tool calls (tool, transport, arguments, latency, error)."""
    from app.mcp.client import mcp_client
    return {"transport": mcp_client.transport, "invocations": mcp_client.get_invocation_history()}


@router.get("/metrics")
def get_observability_metrics():
    """Aggregated performance metrics, token usage, cost accounting, and SLA statistics."""
    traces = tracer.list_all_traces()
    total_runs = len(traces)
    total_duration_ms = sum(t["total_duration_ms"] for t in traces)
    avg_latency_ms = round(total_duration_ms / total_runs, 2) if total_runs > 0 else 0.0
    total_tokens = sum(t["tokens"]["total_tokens"] for t in traces)
    total_cost_usd = sum(t["financial_cost"]["estimated_cost_usd"] for t in traces)
    total_cost_inr = sum(t["financial_cost"]["estimated_cost_inr"] for t in traces)

    return {
        "total_executions": total_runs,
        "average_latency_ms": avg_latency_ms,
        "average_latency_seconds": round(avg_latency_ms / 1000.0, 3),
        "token_usage": {
            "total_tokens_consumed": total_tokens,
            "avg_tokens_per_workflow": round(total_tokens / total_runs, 1) if total_runs > 0 else 0,
        },
        "financial_cost": {
            "total_cost_usd": round(total_cost_usd, 6),
            "total_cost_inr": round(total_cost_inr, 4),
            "avg_cost_per_run_usd": round(total_cost_usd / total_runs, 6) if total_runs > 0 else 0.0,
            "avg_cost_per_run_inr": round(total_cost_inr / total_runs, 4) if total_runs > 0 else 0.0,
        },
        "architecture_layers": {
            "orchestration": "CrewAI agents (LLM: requirement, negotiation, coordination) + deterministic MCP tools",
            "auditor": "Google ADK Risk Agent",
            "inter_agent_protocol": "A2A (official a2a-sdk, JSON-RPC, Agent Card discovery)",
            "tool_protocol": "Model Context Protocol (FastMCP)",
            "database": "SQLAlchemy 2.x (MySQL / PostgreSQL / SQLite)",
            "observability": "Langfuse v3+ tracing (when keys set) + local trace store + callbacks",
            "guardrails": "Regex & Semantic Injection Filter + Pydantic Type Validator",
        }
    }


@router.post("/guardrail-check")
def test_guardrails(payload: GuardrailCheckRequest):
    """Test input prompt against security guardrails and domain boundaries."""
    try:
        SecurityGuardrails.validate_user_prompt(payload.prompt)
        if payload.quantity is not None or payload.budget is not None:
            from decimal import Decimal
            b_dec = Decimal(str(payload.budget)) if payload.budget is not None else None
            SecurityGuardrails.validate_domain_constraints(quantity=payload.quantity, budget=b_dec)
        return {
            "is_safe": True,
            "status": "PASSED",
            "message": "Input passed all prompt injection, toxicity, and domain boundary guardrails."
        }
    except GuardrailViolationException as gve:
        return {
            "is_safe": False,
            "status": "BLOCKED",
            "rule_violated": gve.rule_name,
            "severity": gve.severity,
            "message": gve.message,
            "sanitized_input": gve.sanitized_input
        }


SAMPLE_PROMPTS = [
    "Purchase 50 laptops with at least 16 GB RAM and 512 GB SSD, a maximum budget of \u20b938,00,000, delivery within 14 days, and a minimum two-year warranty.",
    "Purchase 15 4K displays, budget of \u20b97,50,000, delivery within 21 days, 24 months warranty.",
    "Purchase 20 managed 48-port PoE+ gigabit switches, budget of \u20b918,00,000, delivery within 14 days, 36 months warranty.",
]


@router.post("/run-samples")
def run_sample_workflows(n: int = 3):
    """Run up to 3 real sample workflows end-to-end so the cross-matrix fills with genuine traces.
    Uses the configured mode (LLM agents if enabled, otherwise deterministic) - it is not fake data."""
    from app.agents.procurement_crew import ProcurementWorkflowEngine
    from app.db.repositories.procurement_repo import ProcurementRepository
    from app.db.session import SessionLocal
    from app.core.config import settings as _cfg
    n = max(1, min(int(n), len(SAMPLE_PROMPTS)))
    note = None
    if _cfg.llm_agents_enabled and n > 1:
        # each LLM workflow costs ~6 requests; free-tier keys have daily request caps -> protect the quota
        n, note = 1, "LLM mode is ON: limited to 1 sample workflow (about 6 LLM requests) to protect the daily quota."
    results = []
    db = SessionLocal()
    try:
        for text in SAMPLE_PROMPTS[:n]:
            engine = ProcurementWorkflowEngine(db)
            ext = engine.run_requirement_agent(text)
            req = ProcurementRepository(db).create_purchase_request(
                requester="Sample runner", raw_request_text=text, product_requirements=ext.model_dump(mode="json"),
                quantity=ext.quantity, budget_limit=ext.budget_limit, delivery_deadline_days=ext.delivery_deadline_days,
                warranty_requirement_months=ext.warranty_requirement_months)
            try:
                res = engine.execute_workflow(req.request_id)
                results.append({"request": text[:60], "status": res["status"], "workflow_id": res["workflow_id"]})
            except Exception as e:  # a failed sample must not hide the others
                results.append({"request": text[:60], "status": "ERROR", "error": str(e)[:160]})
    finally:
        db.close()
    from app.core.config import settings as _s
    return {"ran": len(results), "llm_agents_enabled": _s.llm_agents_enabled, "note": note, "results": results}
