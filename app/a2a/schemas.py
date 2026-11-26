import datetime
import uuid
from decimal import Decimal
from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field


class A2AMessageEnvelope(BaseModel):
    """Internal audit/correlation record of one agent-to-agent exchange.
    The wire protocol itself is the official A2A JSON-RPC (see app/a2a/client.py);
    this model is what we persist/log for provenance."""
    protocol_version: str = Field("A2A/1.0", description="Agent-to-Agent protocol specification version")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Traceable correlation ID")
    sender_agent: str = Field(..., description="ID or role of sending agent, e.g. CrewAI:ApprovalAgent")
    target_agent: str = Field(..., description="ID or role of target agent, e.g. GoogleADK:ProcurementRiskAgent")
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    message_type: str = Field(..., description="E.g. RISK_ASSESSMENT_REQUEST, RISK_ASSESSMENT_RESPONSE")
    payload: Dict[str, Any] = Field(default_factory=dict)
    status: str = Field("SUCCESS", description="SUCCESS, ERROR, TIMEOUT")
    error_message: Optional[str] = None


class RiskAssessmentPayload(BaseModel):
    request_id: str
    vendor_id: str
    vendor_name: str
    total_cost: float
    budget_limit: float
    is_budget_valid: bool
    budget_overrun: float = 0.0
    is_compliant: bool
    failed_compliance_reasons: List[str] = []
    vendor_rating: float
    delivery_days: int
    warranty_months: int
    evaluation_score: float
    approval_policy_threshold: float = 1000000.0


class RiskAssessmentResponsePayload(BaseModel):
    """Strictly validated result returned by the remote ADK agent over A2A.
    Out-of-range scores / unknown categories are rejected rather than trusted."""
    risk_category: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_factors: List[str]
    explanation: str
    recommended_action: Literal["APPROVE", "REVIEW", "REJECT", "ESCALATE"]
    human_review_required: bool
    assessed_by: str = "Google ADK Risk Assessment Agent (A2A Protocol)"
