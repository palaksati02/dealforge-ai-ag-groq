"""
MCP client for the DealForge FastMCP server.

Two transports, selected by `settings.MCP_TRANSPORT`:
  * "inprocess" (default; used by the unit tests): calls the FastMCP server object directly.
  * "stdio"                : spawns `python -m app.mcp.server` as a SEPARATE PROCESS and talks the real
                             MCP wire protocol (JSON-RPC over stdio) through `mcp.ClientSession`.

Every call is: argument-logged -> executed -> output validated against a strict Pydantic contract
(SecurityGuardrails.validate_tool_output) -> recorded in an audit trail -> emitted to the callbacks.
The orchestrator uses the RETURNED values as the source of truth (they are not discarded).
"""
import asyncio
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from app.core.callbacks import workflow_monitor
from app.core.config import settings
from app.core.guardrails import GuardrailViolationException, SecurityGuardrails
from app.core.logging import logger
from app.mcp.server import mcp
from app.schemas.schemas import (
    ApprovalCreateResult,
    BudgetValidationResult,
    ComplianceValidationResult,
    NegotiationSaveResult,
    PurchaseOrderCreateResult,
    QuoteCalculationBreakdown,
    VendorResponse,
    VendorSearchResult,
)
from crewai.tools import tool

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# tool name -> (Pydantic contract, result_is_list)
TOOL_OUTPUT_SCHEMAS: Dict[str, Tuple[Type[BaseModel], bool]] = {
    "search_vendors": (VendorSearchResult, True),
    "get_vendor_details": (VendorResponse, False),
    "get_compliance_status": (ComplianceValidationResult, False),
    "calculate_quote_total": (QuoteCalculationBreakdown, False),
    "validate_budget": (BudgetValidationResult, False),
    "save_negotiation_proposal": (NegotiationSaveResult, False),
    "create_approval_request": (ApprovalCreateResult, False),
    "create_purchase_order": (PurchaseOrderCreateResult, False),
}


class MCPInvocationRecord:
    """Record of an atomic MCP protocol tool call for auditability and provenance."""
    def __init__(self, tool_name: str, arguments: Dict[str, Any], result: Any, duration_ms: float,
                 error: Optional[str] = None, transport: str = "inprocess"):
        self.timestamp = time.time()
        self.tool_name = tool_name
        self.arguments = arguments
        self.result = result
        self.duration_ms = duration_ms
        self.error = error
        self.transport = transport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "transport": self.transport,
            "arguments": self.arguments,
            "result_summary": str(self.result)[:200] if self.result is not None else None,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
        }


def _unwrap_structured(structured: Any) -> Any:
    """FastMCP wraps non-object returns as {"result": ...}; unwrap them."""
    if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
        return structured["result"]
    return structured


def _parse_content_blocks(blocks: Any) -> Any:
    """Fallback: parse JSON out of text content blocks."""
    for block in blocks or []:
        text = getattr(block, "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except Exception:
                return text
    return None


class _StdioMCPTransport:
    """Runs the FastMCP server in a child process and talks real MCP over stdio.
    A dedicated event-loop thread owns the long-lived ClientSession."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session = None
        self._ready = threading.Event()
        self._stop: Optional[asyncio.Event] = None
        self._startup_error: Optional[BaseException] = None
        self._lock = threading.Lock()

    def _ensure_started(self, timeout: float = 30.0) -> None:
        with self._lock:
            if self._session is not None:
                return
            self._ready.clear()
            self._startup_error = None
            self._thread = threading.Thread(target=self._thread_main, name="mcp-stdio", daemon=True)
            self._thread.start()
            if not self._ready.wait(timeout):
                raise RuntimeError("MCP stdio server did not start in time")
            if self._startup_error:
                raise RuntimeError(f"MCP stdio server failed to start: {self._startup_error}")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        except BaseException as exc:  # noqa: BLE001
            self._startup_error = exc
            self._ready.set()
        finally:
            self._session = None

    async def _serve(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp.server"],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        )
        self._stop = asyncio.Event()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._ready.set()
                await self._stop.wait()

    def _submit(self, coro, timeout: float):
        self._ensure_started()
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def list_tools(self, timeout: float = 15.0) -> List[str]:
        async def _go():
            res = await self._session.list_tools()
            return [t.name for t in res.tools]
        return self._submit(_go(), timeout)

    def call(self, name: str, arguments: Dict[str, Any], timeout: float) -> Any:
        async def _go():
            res = await self._session.call_tool(name, arguments)
            if res.isError:
                msg = _parse_content_blocks(res.content)
                raise RuntimeError(f"MCP server error: {msg}")
            if res.structuredContent is not None:
                return _unwrap_structured(res.structuredContent)
            return _parse_content_blocks(res.content)
        return self._submit(_go(), timeout)

    def close(self) -> None:
        if self._loop and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            self._thread.join(timeout=10)
        self._session = None
        self._loop = None


class MCPClient:
    """
    Model Context Protocol (MCP) client: discovers the FastMCP server's tools and executes them
    with strict validation, latency tracking and audit logging.
    """

    MAX_RECORDS = 500

    def __init__(self, server=None, transport: Optional[str] = None):
        self.server = server or mcp
        self._transport_name = transport  # None -> read settings lazily
        self._stdio: Optional[_StdioMCPTransport] = None
        self._tools_cache: Optional[List[str]] = None
        self.invocations: List[MCPInvocationRecord] = []
        self.total_calls = 0           # monotonic (invocations is capped)

    @property
    def transport(self) -> str:
        return (self._transport_name or settings.MCP_TRANSPORT or "inprocess").lower()

    def _stdio_transport(self) -> _StdioMCPTransport:
        if self._stdio is None:
            self._stdio = _StdioMCPTransport()
        return self._stdio

    def close(self) -> None:
        if self._stdio is not None:
            self._stdio.close()
            self._stdio = None

    # -- discovery --------------------------------------------------------------------------
    def discover_tools(self) -> List[str]:
        """Discover the tools actually exposed by the FastMCP server (no hardcoded fallback list)."""
        try:
            if self.transport == "stdio":
                names = self._stdio_transport().list_tools()
            else:
                names = [t.name for t in self._run_async(self.server.list_tools())]
            self._tools_cache = names
            logger.info(f"[MCP Client/{self.transport}] Discovered {len(names)} tools: {', '.join(names)}")
            return names
        except Exception as e:
            logger.error(f"[MCP Client] Tool discovery failed: {e}")
            self._tools_cache = []
            return []

    # -- execution --------------------------------------------------------------------------
    def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        workflow_id: Optional[str] = None,
        request_id: Optional[str] = None,
        timeout_seconds: float = 20.0,
    ) -> Any:
        """Execute a tool via the MCP protocol; return the VALIDATED result."""
        start_time = time.perf_counter()
        transport = self.transport
        logger.info(f"[MCP/{transport}] >> calling tool '{tool_name}' with args {str(arguments)[:200]}")

        try:
            if transport == "stdio":
                parsed = self._stdio_transport().call(tool_name, arguments, timeout_seconds)
            else:
                raw = self._run_async(asyncio.wait_for(self.server.call_tool(tool_name, arguments), timeout=timeout_seconds))
                parsed = self._parse_inprocess(raw)

            duration_ms = (time.perf_counter() - start_time) * 1000.0

            # Guardrail: strict type validation of the tool's return value
            schema = TOOL_OUTPUT_SCHEMAS.get(tool_name)
            if schema:
                SecurityGuardrails.validate_tool_output(tool_name, parsed, expected_type=schema[0], is_list=schema[1])
            else:
                SecurityGuardrails.validate_tool_output(tool_name, parsed)

            self._add_record(MCPInvocationRecord(tool_name, arguments, parsed, duration_ms, transport=transport))
            logger.info(f"[MCP/{transport}] << '{tool_name}' ok in {duration_ms:.2f}ms")

            if workflow_id and request_id:
                workflow_monitor.on_intermediate_output(
                    workflow_id=workflow_id, request_id=request_id, agent_name="MCP Tool Layer",
                    step_name=f"MCP_TOOL:{tool_name}",
                    output_data={"result_summary": str(parsed)[:100]},
                    summary=f"MCP tool '{tool_name}' returned valid data ({duration_ms:.1f}ms)",
                )
            return parsed

        except GuardrailViolationException as gve:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self._add_record(MCPInvocationRecord(tool_name, arguments, None, duration_ms, error=gve.message, transport=transport))
            logger.error(f"[MCP/{transport}] '{tool_name}' output rejected by guardrail: {gve.message}")
            raise
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self._add_record(MCPInvocationRecord(tool_name, arguments, None, duration_ms, error=str(e), transport=transport))
            logger.error(f"[MCP/{transport}] '{tool_name}' failed ({duration_ms:.2f}ms): {e}")
            raise RuntimeError(f"MCP Tool '{tool_name}' failed: {e}")

    @staticmethod
    def _parse_inprocess(raw: Any) -> Any:
        """FastMCP.call_tool returns (content_blocks, structured) or a bare sequence/CallToolResult."""
        if isinstance(raw, tuple) and len(raw) >= 2 and isinstance(raw[1], (dict, list)):
            return _unwrap_structured(raw[1])
        if hasattr(raw, "structuredContent") and getattr(raw, "structuredContent", None) is not None:
            return _unwrap_structured(raw.structuredContent)
        blocks = raw[0] if (isinstance(raw, tuple) and len(raw) == 1) else raw
        return _parse_content_blocks(blocks if isinstance(blocks, (list, tuple)) else [blocks])

    def _add_record(self, rec: MCPInvocationRecord) -> None:
        self.total_calls += 1
        self.invocations.append(rec)
        del self.invocations[:-self.MAX_RECORDS]

    def _run_async(self, coro):
        """Run an async FastMCP coroutine safely from sync code, inside or outside a running loop."""
        box: Dict[str, Any] = {}

        def runner():
            try:
                box["value"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001
                box["error"] = exc

        t = threading.Thread(target=runner, name="mcp-inproc", daemon=True)
        t.start()
        t.join()
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def get_invocation_history(self) -> List[Dict[str, Any]]:
        return [inv.to_dict() for inv in self.invocations]


# Singleton MCP Client instance
mcp_client = MCPClient()


# ---------------------------------------------------------------------------
# CrewAI Tool Wrappers (Invoking Tools through Genuine MCP Protocol)
# ---------------------------------------------------------------------------

@tool("search_vendors_mcp")
def search_vendors_mcp(product_category: str) -> str:
    """
    Search database for active vendors and matching catalog products in a category using FastMCP.
    """
    res = mcp_client.call_tool("search_vendors", {"product_category": product_category})
    return json.dumps(res, default=str)


@tool("get_vendor_details_mcp")
def get_vendor_details_mcp(vendor_id: str) -> str:
    """
    Retrieve contact, address, and rating details for a specific vendor from MySQL via FastMCP.
    """
    res = mcp_client.call_tool("get_vendor_details", {"vendor_id": vendor_id})
    return json.dumps(res, default=str)


@tool("get_product_inventory_mcp")
def get_product_inventory_mcp(product_id: str) -> str:
    """
    Retrieve live stock levels and pricing across vendors for a product ID via FastMCP.
    """
    res = mcp_client.call_tool("get_product_inventory", {"product_id": product_id})
    return json.dumps(res, default=str)


@tool("calculate_quote_total_mcp")
def calculate_quote_total_mcp(unit_price: float, quantity: int) -> str:
    """
    Deterministically compute base subtotal, volume tier discounts, 18% GST, shipping, and grand total via FastMCP.
    """
    res = mcp_client.call_tool("calculate_quote_total", {"unit_price": unit_price, "quantity": quantity})
    return json.dumps(res, default=str)


@tool("validate_budget_mcp")
def validate_budget_mcp(total_cost: float, budget_limit: float) -> str:
    """
    Deterministically audit total procurement cost against budget limit via FastMCP.
    """
    res = mcp_client.call_tool("validate_budget", {"total_cost": total_cost, "budget_limit": budget_limit})
    return json.dumps(res, default=str)


@tool("create_purchase_order_mcp")
def create_purchase_order_mcp(request_id: str, vendor_id: str, quote_id: str, approved_total: float, approved_by: str) -> str:
    """
    Deterministically issue and commit a formal Purchase Order to MySQL via FastMCP.
    """
    res = mcp_client.call_tool("create_purchase_order", {
        "request_id": request_id,
        "vendor_id": vendor_id,
        "quote_id": quote_id,
        "approved_total": approved_total,
        "approved_by": approved_by
    })
    return json.dumps(res, default=str)


@tool("get_compliance_status_mcp")
def get_compliance_status_mcp(vendor_id: str, request_id: str) -> str:
    """
    Evaluate procurement compliance rules (active status, warranty, delivery SLA, stock) for a vendor via FastMCP.
    """
    res = mcp_client.call_tool("get_compliance_status", {"vendor_id": vendor_id, "request_id": request_id})
    return json.dumps(res, default=str)


@tool("save_negotiation_proposal_mcp")
def save_negotiation_proposal_mcp(request_id: str, quote_id: str, original_amount: float, target_amount: float,
                                  counter_offer_amount: float, justification: str, draft_message: str) -> str:
    """
    Persist a counter-offer proposal and its justification to the database via FastMCP.
    """
    res = mcp_client.call_tool("save_negotiation_proposal", {
        "request_id": request_id, "quote_id": quote_id, "original_amount": original_amount,
        "target_amount": target_amount, "counter_offer_amount": counter_offer_amount,
        "justification": justification, "draft_message": draft_message,
    })
    return json.dumps(res, default=str)
