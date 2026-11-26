import re
import threading
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Type
from pydantic import BaseModel, ValidationError
from app.core.logging import logger


class GuardrailViolationException(Exception):
    """Raised when an input or tool execution violates security or domain guardrails."""
    def __init__(self, rule_name: str, message: str, severity: str = "HIGH", sanitized_input: Optional[str] = None):
        super().__init__(message)
        self.rule_name = rule_name
        self.message = message
        self.severity = severity
        self.sanitized_input = sanitized_input


class SecurityGuardrails:
    """
    Enterprise-grade security guardrails for DealForge AI:
    1. Prompt Injection & Jailbreak Defense
    2. Toxic & Abusive Language Filtering
    3. Tool Return & Parameter Type Validation Callbacks
    4. Procurement Domain Boundary Enforcement
    """

    # Prompt injection and jailbreak patterns
    INJECTION_PATTERNS = [
        r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b",
        r"(?i)\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b",
        r"(?i)\bsystem\s+override\b",
        r"(?i)\byou\s+are\s+now\s+(?:an?\s+)?(?:unfiltered|jailbroken|developer\s+mode|dan)\b",
        r"(?i)\bpretend\s+you\s+have\s+no\s+(?:rules|guidelines|restrictions)\b",
        r"(?i)\bprint\s+(?:your\s+)?(?:system\s+prompt|instructions|api\s*key)\b",
        r"(?i)\bdrop\s+table\b",
        r"(?i)\bunion\s+select\b",
        r"(?i)<script\b[^>]*>",
        r"(?i)javascript:\s*",
        r"(?i)\bexec\s*\(\s*['\"]",
        r"(?i)\beval\s*\(\s*['\"]",
        r"(?i)base64_decode",
    ]

    # Toxic, abusive, or harmful keyword patterns
    # NOTE: legitimate security-hardware requests ("attack surface reduction firewall",
    # "exploit mitigation") must not be blocked, so benign technical collocations are
    # excluded with negative lookaheads, and the bare words "hack"/"exploit" are not
    # treated as toxic on their own (injection patterns above already cover real exploits).
    TOXIC_PATTERNS = [
        r"(?i)\b(?:kill|murder|bomb|threaten|harass)\b",
        r"(?i)\battack\b(?!\s+(?:surface|vector|detection|prevention|protection|mitigation|simulation))",
        r"(?i)\bhack\s+(?:into|your|the\s+system)\b",
        r"(?i)\b(?:idiot|stupid|bastard|scam|fraudulent\s+scheme)\b",
    ]

    # Domain constraints (documented in README: quantity 1 - 10,000)
    MIN_QUANTITY = 1
    MAX_QUANTITY = 10_000
    MIN_BUDGET = Decimal("100.00")
    MAX_BUDGET = Decimal("1000000000.00")  # ₹100 Crore ceiling

    # Real, thread-safe counters feeding the observability dashboard (no hardcoded "0 violations").
    _stats_lock = threading.Lock()
    _stats: Dict[str, Any] = {"checks": 0, "violations": 0, "by_rule": {}}

    @classmethod
    def _record(cls, rule: Optional[str] = None) -> None:
        with cls._stats_lock:
            cls._stats["checks"] += 1
            if rule:
                cls._stats["violations"] += 1
                cls._stats["by_rule"][rule] = cls._stats["by_rule"].get(rule, 0) + 1

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        with cls._stats_lock:
            return {
                "checks": cls._stats["checks"],
                "violations": cls._stats["violations"],
                "by_rule": dict(cls._stats["by_rule"]),
            }

    @classmethod
    def reset_stats(cls) -> None:
        with cls._stats_lock:
            cls._stats = {"checks": 0, "violations": 0, "by_rule": {}}

    @classmethod
    def scan_text(cls, text: str, source: str = "input") -> None:
        """Raise if a free-text string carries an injection/jailbreak pattern (used on
        agent-to-agent payload strings and LLM-facing text, not just the first user prompt)."""
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, text):
                cls._record("PROMPT_INJECTION_DEFENSE")
                logger.warning(f"Security Alert: injection pattern in {source}: {pattern}")
                raise GuardrailViolationException(
                    rule_name="PROMPT_INJECTION_DEFENSE",
                    message=f"Malicious instruction pattern detected in {source}.",
                    severity="CRITICAL",
                    sanitized_input=re.sub(pattern, "[REDACTED_ATTACK_VECTOR]", text),
                )

    @classmethod
    def scan_payload(cls, obj: Any, source: str = "payload") -> None:
        """Recursively scan every string inside a dict/list payload for injection patterns."""
        if isinstance(obj, str):
            cls.scan_text(obj, source)
        elif isinstance(obj, dict):
            for v in obj.values():
                cls.scan_payload(v, source)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                cls.scan_payload(v, source)

    @classmethod
    def validate_user_prompt(cls, text: str) -> Tuple[bool, Optional[str]]:
        """
        Scans incoming natural-language purchase request for prompt injection,
        jailbreak attempts, and toxic or malicious patterns.
        """
        if not text or not text.strip():
            cls._record("EMPTY_PROMPT_RULE")
            raise GuardrailViolationException(
                rule_name="EMPTY_PROMPT_RULE",
                message="User procurement prompt cannot be empty or whitespace only.",
                severity="MEDIUM"
            )

        # 1. Prompt Injection Checks
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, text):
                logger.warning(f"Security Alert: Prompt injection pattern matched: {pattern}")
                cls._record("PROMPT_INJECTION_DEFENSE")
                raise GuardrailViolationException(
                    rule_name="PROMPT_INJECTION_DEFENSE",
                    message="Malicious prompt injection or instruction override attempt detected.",
                    severity="CRITICAL",
                    sanitized_input=re.sub(pattern, "[REDACTED_ATTACK_VECTOR]", text)
                )

        # 2. Toxic Language Checks
        for pattern in cls.TOXIC_PATTERNS:
            if re.search(pattern, text):
                logger.warning(f"Security Alert: Toxic/harmful pattern matched: {pattern}")
                cls._record("TOXIC_LANGUAGE_DEFENSE")
                raise GuardrailViolationException(
                    rule_name="TOXIC_LANGUAGE_DEFENSE",
                    message="Abusive, toxic, or hazardous language detected in procurement request.",
                    severity="HIGH"
                )

        cls._record()
        return True, None

    @classmethod
    def validate_domain_constraints(
        cls,
        quantity: Optional[int] = None,
        budget: Optional[Decimal] = None,
        category: Optional[str] = None
    ) -> bool:
        """
        Enforces procurement business logic boundaries and prevents arithmetic exploits.
        """
        if quantity is not None:
            if quantity < cls.MIN_QUANTITY or quantity > cls.MAX_QUANTITY:
                cls._record("QUANTITY_BOUNDARY_GUARDRAIL")
                raise GuardrailViolationException(
                    rule_name="QUANTITY_BOUNDARY_GUARDRAIL",
                    message=f"Procurement quantity {quantity} is outside safe allowable bounds ({cls.MIN_QUANTITY} to {cls.MAX_QUANTITY}).",
                    severity="HIGH"
                )

        if budget is not None:
            if budget < cls.MIN_BUDGET or budget > cls.MAX_BUDGET:
                cls._record("BUDGET_BOUNDARY_GUARDRAIL")
                raise GuardrailViolationException(
                    rule_name="BUDGET_BOUNDARY_GUARDRAIL",
                    message=f"Budget ₹{budget:,.2f} is outside enterprise allowable boundaries (₹{cls.MIN_BUDGET} to ₹{cls.MAX_BUDGET}).",
                    severity="HIGH"
                )

        return True

    @classmethod
    def validate_tool_output(
        cls,
        tool_name: str,
        output: Any,
        expected_type: Optional[Type[BaseModel]] = None,
        is_list: bool = False,
    ) -> Any:
        """
        Callback validation of tool return values: null check, error-envelope check and
        STRICT Pydantic type validation (element-wise for list results). Corrupted or
        type-mismatched data never reaches agent state.
        """
        if output is None:
            cls._record("TOOL_OUTPUT_NULL_CHECK")
            raise GuardrailViolationException(
                rule_name="TOOL_OUTPUT_NULL_CHECK",
                message=f"Tool '{tool_name}' returned NULL, violating deterministic execution guarantees.",
                severity="HIGH"
            )

        if isinstance(output, dict) and "error" in output and expected_type is not None:
            cls._record("TOOL_OUTPUT_ERROR_ENVELOPE")
            raise GuardrailViolationException(
                rule_name="TOOL_OUTPUT_ERROR_ENVELOPE",
                message=f"Tool '{tool_name}' reported an error: {output['error']}",
                severity="HIGH"
            )

        if expected_type is not None and issubclass(expected_type, BaseModel):
            try:
                if is_list:
                    if not isinstance(output, list):
                        raise ValueError(f"expected a list, got {type(output).__name__}")
                    for item in output:
                        expected_type.model_validate(item)
                elif isinstance(output, dict):
                    expected_type.model_validate(output)
                elif not isinstance(output, expected_type):
                    raise ValueError(f"expected dict/{expected_type.__name__}, got {type(output).__name__}")
            except (ValidationError, ValueError) as ve:
                logger.error(f"Tool output type validation error in {tool_name}: {ve}")
                cls._record("TOOL_OUTPUT_SCHEMA_VALIDATION")
                raise GuardrailViolationException(
                    rule_name="TOOL_OUTPUT_SCHEMA_VALIDATION",
                    message=f"Tool '{tool_name}' output failed strict Pydantic type validation: {ve}",
                    severity="HIGH"
                )

        cls._record()
        return output

    @classmethod
    def validate_agent_output(cls, agent_name: str, output: BaseModel) -> BaseModel:
        """Guardrail applied to structured LLM-agent output before it is accepted into state."""
        qty = getattr(output, "quantity", None)
        budget = getattr(output, "budget_limit", None)
        cls.validate_domain_constraints(
            quantity=qty,
            budget=Decimal(str(budget)) if budget is not None else None,
        )
        cls._record()
        return output
