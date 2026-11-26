"""
CrewAI tools that are not plain MCP wrappers.

`request_adk_risk_assessment` is how the CrewAI Approval/Coordination agent talks to the
independent Google ADK agent: CrewAI agent -> this tool -> A2A client -> (HTTP, official A2A
JSON-RPC) -> ADK agent. The validated typed result is stashed by request_id so the orchestrator
never has to trust the LLM's re-typing of the numbers.
"""
import threading
from typing import Dict, Optional

from crewai.tools import tool

from app.a2a.client import A2AClient
from app.a2a.schemas import RiskAssessmentPayload, RiskAssessmentResponsePayload

_results: Dict[str, RiskAssessmentResponsePayload] = {}
_lock = threading.Lock()


@tool("request_adk_risk_assessment")
def request_adk_risk_assessment(payload_json: str) -> str:
    """
    Send the proposed procurement deal (a JSON document with request_id, vendor_id, vendor_name,
    total_cost, budget_limit, is_budget_valid, budget_overrun, is_compliant, failed_compliance_reasons,
    vendor_rating, delivery_days, warranty_months, evaluation_score, approval_policy_threshold)
    to the independent Google ADK Risk Agent using the A2A protocol and return its risk assessment JSON.
    Pass the JSON string exactly as provided; do not modify any number.
    """
    payload = RiskAssessmentPayload.model_validate_json(payload_json)
    result = A2AClient().request_risk_assessment(payload)
    with _lock:
        _results[payload.request_id] = result
    return result.model_dump_json()


def pop_a2a_result(request_id: str) -> Optional[RiskAssessmentResponsePayload]:
    with _lock:
        return _results.pop(request_id, None)
