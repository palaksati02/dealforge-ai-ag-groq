import logging
import os
import tempfile

# Background exporter threads (Langfuse/OTel, abandoned timeout runs) may log after pytest has closed its
# streams; don't print 'Logging error' tracebacks for that teardown noise.
logging.raiseExceptions = False

# --- Environment for the whole test session: set BEFORE any app import -------------------------
# Isolated temp database (fixes the old bug where tests silently depended on a pre-created dealforge.db),
# deterministic agents (no network / API key), and no Langfuse export.
_TMP = tempfile.mkdtemp(prefix="dealforge-tests-")
# Override with DEALFORGE_TEST_DATABASE_URL to run the whole suite on MySQL/PostgreSQL.
os.environ["DATABASE_URL"] = os.environ.get("DEALFORGE_TEST_DATABASE_URL") or f"sqlite:///{_TMP}/test.db"
os.environ["USE_LLM_AGENTS"] = "false"
os.environ["DEMO_MODE"] = "True"
os.environ["ADK_AUTOSTART"] = "false"
os.environ["APP_ENV"] = "test"
os.environ["MCP_TRANSPORT"] = "inprocess"
os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
os.environ.pop("LANGFUSE_SECRET_KEY", None)
os.environ["LANGFUSE_ENABLED"] = "false"

import pytest
from decimal import Decimal
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import app.db.session
import app.mcp.tools.procurement_tools as pt
from app.db.base import Base
from app.db.seed import seed_database
from app.db.repositories.procurement_repo import ProcurementRepository

# In-memory SQLite engine with StaticPool for cross-session test isolation
TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    return engine

@pytest.fixture(scope="function")
def db_session(test_engine, monkeypatch):
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    monkeypatch.setattr(app.db.session, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(pt, "SessionLocal", TestingSessionLocal)
    session = TestingSessionLocal()
    seed_database(session)
    yield session
    session.rollback()
    session.close()

@pytest.fixture(scope="function")
def repo(db_session):
    return ProcurementRepository(db_session)

@pytest.fixture(scope="function")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def _database_and_a2a_service():
    """Create + seed the isolated DB once, and run the Google ADK A2A service on a free port so every
    test that crosses the CrewAI -> A2A -> ADK boundary uses REAL HTTP."""
    from app.db.session import engine as real_engine
    from app.db.base import Base as _Base
    import app.db.models  # noqa: F401  (register tables)
    _Base.metadata.create_all(bind=real_engine)
    seed_database()

    from app.adk_service.server import start_embedded_adk_server, find_free_port, stop_embedded_adk_servers
    from app.core.config import settings
    port = find_free_port()
    start_embedded_adk_server(host="127.0.0.1", port=port)
    settings.ADK_SERVICE_HOST = "127.0.0.1"
    settings.ADK_SERVICE_PORT = port
    yield
    stop_embedded_adk_servers()
