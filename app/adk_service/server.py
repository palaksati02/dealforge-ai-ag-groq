"""
Google ADK Risk Agent service, exposed over the OFFICIAL A2A protocol.

`google.adk.a2a.utils.agent_to_a2a.to_a2a` wraps the ADK agent in an a2a-sdk server which provides:
  * GET  /.well-known/agent-card.json   -> the Agent Card (identity, skills, endpoint URL)
  * POST /                              -> A2A JSON-RPC (`message/send`, `message/stream`, `tasks/get`...)

Run standalone:   uvicorn app.adk_service.server:adk_app --host 0.0.0.0 --port 8001
Or embedded:      start_embedded_adk_server()  (used by `python run.py` and the test-suite so the
                  two agents still talk over real HTTP on a single machine).
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import socket
import threading
import time
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx
import uvicorn
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.adk_service.agent import adk_risk_agent, ADK_USAGE
from app.core.config import settings
from app.core.logging import logger

AGENT_CARD_PATH = "/.well-known/agent-card.json"


def _advertised_location() -> Tuple[str, str, int]:
    """(protocol, host, port) written into the Agent Card."""
    if settings.ADK_PUBLIC_URL:
        u = urlparse(settings.ADK_PUBLIC_URL)
        proto = u.scheme or "http"
        return proto, u.hostname or "localhost", u.port or (443 if proto == "https" else 80)
    return "http", settings.ADK_SERVICE_HOST, settings.ADK_SERVICE_PORT


def build_a2a_app(protocol: Optional[str] = None, host: Optional[str] = None, port: Optional[int] = None):
    """Create the A2A Starlette application for the ADK risk agent."""
    d_proto, d_host, d_port = _advertised_location()
    app = to_a2a(
        adk_risk_agent.adk_agent,
        host=host or d_host,
        port=port or d_port,
        protocol=protocol or d_proto,
        runner=adk_risk_agent.runner,
    )

    async def health(_request):
        return JSONResponse({
            "status": "healthy",
            "service": "Google ADK Risk Assessment Agent",
            "protocol": "A2A (JSON-RPC)",
            "agent_card": AGENT_CARD_PATH,
            "agent_name": adk_risk_agent.adk_agent.name,
        })

    async def usage(request):
        return JSONResponse(ADK_USAGE.get(request.path_params["request_id"], {}))

    app.router.routes.append(Route("/health", health, methods=["GET"]))
    app.router.routes.append(Route("/usage/{request_id}", usage, methods=["GET"]))
    return app


# Module-level app for `uvicorn app.adk_service.server:adk_app`
adk_app = build_a2a_app()


# ---------------------------------------------------------------------------
# Embedded server helper
# ---------------------------------------------------------------------------
_embedded: dict = {}


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def is_a2a_server_up(base_url: str, timeout: float = 1.0) -> bool:
    try:
        return httpx.get(f"{base_url.rstrip('/')}{AGENT_CARD_PATH}", timeout=timeout).status_code == 200
    except Exception:
        return False


def start_embedded_adk_server(host: str = "127.0.0.1", port: Optional[int] = None, wait_seconds: float = 15.0) -> str:
    """Start the A2A ADK service in a background thread (idempotent). Returns its base URL."""
    port = port or settings.ADK_SERVICE_PORT
    base_url = f"http://{host}:{port}"
    if base_url in _embedded or is_a2a_server_up(base_url):
        return base_url

    app = build_a2a_app("http", host, port)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, name=f"adk-a2a-{port}", daemon=True)
    thread.start()
    _embedded[base_url] = (server, thread)

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if is_a2a_server_up(base_url):
            logger.info(f"[A2A] Embedded Google ADK service ready at {base_url} (Agent Card: {base_url}{AGENT_CARD_PATH})")
            return base_url
        time.sleep(0.2)
    raise RuntimeError(f"Embedded ADK A2A server did not become ready at {base_url}")


def stop_embedded_adk_servers() -> None:
    for base_url, (server, thread) in list(_embedded.items()):
        server.should_exit = True
        thread.join(timeout=5)
        _embedded.pop(base_url, None)


if __name__ == "__main__":
    uvicorn.run(
        "app.adk_service.server:adk_app",
        host=os.environ.get("ADK_BIND_HOST", "0.0.0.0"),
        port=settings.ADK_SERVICE_PORT,
        reload=False,
    )
