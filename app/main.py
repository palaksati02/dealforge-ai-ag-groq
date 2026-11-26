import sys
import os

# Prevent app/ directory from shadowing the installed 'mcp' library package
script_dir = os.path.dirname(os.path.abspath(__file__))
while script_dir in sys.path:
    sys.path.remove(script_dir)

# Ensure project root is at the front of sys.path
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from app.core.config import settings
from app.core.logging import logger
from app.db.base import Base
from app.db.session import engine, check_db_connection
from app.db.seed import seed_database
from app.adk_service.server import start_embedded_adk_server, stop_embedded_adk_servers, is_a2a_server_up, AGENT_CARD_PATH

# Import API Routers
from app.api.routes.requests import router as requests_router
from app.api.routes.workflows import router as workflows_router
from app.api.routes.vendors import router as vendors_router
from app.api.routes.quotes import router as quotes_router
from app.api.routes.evaluations import router as evaluations_router
from app.api.routes.negotiations import router as negotiations_router
from app.api.routes.approvals import router as approvals_router
from app.api.routes.purchase_orders import router as purchase_orders_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.replanning import router as replanning_router
from app.api.routes.observability import router as observability_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing DealForge AI Enterprise Platform...")

    # Verify and initialize database tables
    try:
        is_connected = check_db_connection()
        if is_connected:
            logger.info("Connected to primary database. Creating tables if not present...")
            Base.metadata.create_all(bind=engine)
            seed_database()
            logger.info("Database initialization and seed verification complete.")
            from app.core.observability import tracer
            tracer.load_from_db()   # restore dashboard history (cross-matrix) after a restart/cold start
        else:
            logger.warning("Database connection failed on startup. Verify MySQL credentials in .env.")
    except Exception as e:
        logger.error(f"Database startup error: {e}")

    # Start the Google ADK A2A service in-process on loopback if no external one is configured.
    # (Docker/Render deployments point ADK_SERVICE_HOST at a separate service and skip this.)
    if settings.ADK_AUTOSTART and settings.ADK_SERVICE_HOST in ("127.0.0.1", "localhost") and not settings.ADK_SERVICE_URL_OVERRIDE:
        try:
            start_embedded_adk_server(host="127.0.0.1", port=settings.ADK_SERVICE_PORT)
        except Exception as e:
            logger.error(f"Embedded ADK A2A service failed to start: {e}. Deals will be routed to human review (fail-safe).")

    yield

    logger.info("Shutting down DealForge AI...")
    stop_embedded_adk_servers()
    try:
        from app.mcp.client import mcp_client
        mcp_client.close()
    except Exception:
        pass
    try:
        from app.core.observability import tracer
        if tracer.langfuse_client:
            tracer.langfuse_client.flush()
    except Exception:
        pass


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "DealForge AI: An Integrated Multi-Agent Enterprise Procurement Automation System. "
        "Powered by CrewAI, Google ADK, Model Context Protocol (MCP), A2A Protocol, and MySQL."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend dashboard and inter-agent integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Endpoints
@app.get("/api/health", tags=["Health"])
def health_check():
    """Real dependency checks: database, A2A/ADK agent card, MCP tool discovery, LLM mode."""
    from app.mcp.client import mcp_client
    db_ok = check_db_connection()
    adk_ok = is_a2a_server_up(settings.adk_service_url, timeout=2.0)
    tools = mcp_client.discover_tools()
    healthy = db_ok and adk_ok and len(tools) > 0
    return {
        "status": "healthy" if healthy else "degraded",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "llm_agents_enabled": settings.llm_agents_enabled,
        "database_connected": db_ok,
        "adk_a2a_service": {"url": settings.adk_service_url, "agent_card": f"{settings.adk_service_url}{AGENT_CARD_PATH}",
                            "reachable": adk_ok},
        "mcp": {"transport": mcp_client.transport, "tools_discovered": len(tools)},
        "langfuse_exporting": __import__("app.core.observability", fromlist=["tracer"]).tracer.langfuse_client is not None,
        "protocol": "A2A (JSON-RPC, Agent Card)",
    }


@app.get("/api/health/db", tags=["Health"])
def db_health_check():
    db_ok = check_db_connection()
    return {"database": "mysql", "connected": db_ok}


# Include All Procurement API Routers under /api
app.include_router(requests_router, prefix="/api")
app.include_router(workflows_router, prefix="/api")
app.include_router(vendors_router, prefix="/api")
app.include_router(quotes_router, prefix="/api")
app.include_router(evaluations_router, prefix="/api")
app.include_router(negotiations_router, prefix="/api")
app.include_router(approvals_router, prefix="/api")
app.include_router(purchase_orders_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")
app.include_router(replanning_router, prefix="/api")
app.include_router(observability_router, prefix="/api")

# Mount Frontend Static Directory
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_frontend_root():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "DealForge AI API is active. Access docs at /docs"}


if __name__ == "__main__":
    import uvicorn
    # Use direct app object or project root app_dir to guarantee clean startup in any terminal
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
