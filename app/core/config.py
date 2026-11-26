import os
from decimal import Decimal
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application
    APP_NAME: str = "DealForge AI"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    SECRET_KEY: str = "dealforge-secret-key-change-in-production"

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "debug", "dev", "development")
        return bool(v)

    # Database Configuration (MySQL Primary)
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str = "root"
    DB_PASSWORD: str = "root"
    DB_NAME: str = "dealforge_db"
    DB_ECHO: bool = False
    DATABASE_URL_OVERRIDE: Optional[str] = None

    # LLM Settings
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-3.5-flash"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    # Which provider serves the LLM-backed agents: "auto" (Groq if GROQ_API_KEY is set, else Gemini),
    # "groq" or "gemini".
    LLM_PROVIDER: str = "auto"
    # DEMO_MODE=True keeps the pipeline fully deterministic/offline unless
    # USE_LLM_AGENTS is explicitly set to true.
    DEMO_MODE: bool = True
    # Tri-state: None = auto (LLM agents run only if a Gemini key exists AND DEMO_MODE is off),
    # True = force LLM agents on, False = force deterministic fallback.
    USE_LLM_AGENTS: Optional[bool] = None

    # Model pricing used for REAL cost accounting from measured token usage.
    # USD per 1M tokens. Defaults are Gemini 2.5 Flash list prices -- verify against
    # https://ai.google.dev/pricing for the model you configure in GEMINI_MODEL.
    # Hard ceiling for one LLM-backed CrewAI agent run; on timeout the agent falls back to the deterministic path.
    LLM_TIMEOUT_SECONDS: int = 60
    # With Groq's free tier you pay nothing, so these are used as a PROJECTED list-price cost
    # (set them to the Groq list price of GROQ_MODEL, e.g. llama-3.3-70b-versatile ~ $0.59 in / $0.79 out).
    LLM_PRICE_INPUT_PER_M: Decimal = Decimal("0.59")
    LLM_PRICE_OUTPUT_PER_M: Decimal = Decimal("0.79")
    USD_INR_RATE: Decimal = Decimal("86.50")

    # MCP transport: "inprocess" (default, used by tests) or "stdio" (spawns the FastMCP
    # server as a separate process and talks the real MCP wire protocol).
    MCP_TRANSPORT: str = "inprocess"

    # Google ADK Service & A2A Protocol
    ADK_SERVICE_HOST: str = "127.0.0.1"
    ADK_SERVICE_PORT: int = 8001
    ADK_TIMEOUT_SECONDS: int = 15
    ADK_SERVICE_URL_OVERRIDE: Optional[str] = None
    # Start the A2A ADK service inside this process (real HTTP on loopback) when ADK_SERVICE_HOST is local.
    ADK_AUTOSTART: bool = True
    # Public base URL the A2A Agent Card advertises (set to the deployed ADK service URL).
    ADK_PUBLIC_URL: Optional[str] = None

    # Langfuse tracing. NO default keys: tracing exports only when both keys are supplied.
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_ENABLED: bool = True

    # Procurement Engine Business Rules
    AUTO_APPROVAL_THRESHOLD: Decimal = Decimal("1000000.00")  # ₹10,00,000
    OVERRUN_TOLERANCE_PERCENTAGE: Decimal = Decimal("0.0")
    DEFAULT_CURRENCY: str = "INR"

    # Configurable Evaluation Scoring Weights (Sum to 1.0)
    WEIGHT_PRICE: Decimal = Decimal("0.35")
    WEIGHT_DELIVERY: Decimal = Decimal("0.20")
    WEIGHT_WARRANTY: Decimal = Decimal("0.15")
    WEIGHT_COMPLIANCE: Decimal = Decimal("0.15")
    WEIGHT_SPECIFICATION: Decimal = Decimal("0.15")

    @staticmethod
    def _normalize_pg_url(url: str) -> str:
        """Pin the psycopg2 driver (SQLAlchemy 2.1 defaults postgresql:// to psycopg v3)."""
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg2://" + url[len("postgresql://"):]
        return url

    @property
    def database_url(self) -> str:
        # Standard cloud PaaS environment variable (Render, Railway, Heroku)
        env_db_url = os.environ.get("DATABASE_URL")
        if env_db_url:
            return self._normalize_pg_url(env_db_url)
        if self.DATABASE_URL_OVERRIDE:
            return self._normalize_pg_url(self.DATABASE_URL_OVERRIDE)
        # URL encode password if needed
        import urllib.parse
        encoded_pass = urllib.parse.quote_plus(self.DB_PASSWORD)
        return f"mysql+pymysql://{self.DB_USER}:{encoded_pass}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?charset=utf8mb4"

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() in ("production", "prod")

    @property
    def llm_provider(self) -> str:
        """Resolved provider: 'groq' or 'gemini'."""
        choice = (self.LLM_PROVIDER or "auto").strip().lower()
        if choice in ("groq", "gemini"):
            return choice
        return "groq" if self.GROQ_API_KEY else "gemini"

    @property
    def llm_api_key(self) -> Optional[str]:
        return self.GROQ_API_KEY if self.llm_provider == "groq" else self.GEMINI_API_KEY

    @property
    def llm_model_name(self) -> str:
        """Model id for the active provider (used for display, pricing and Langfuse)."""
        return self.GROQ_MODEL if self.llm_provider == "groq" else self.GEMINI_MODEL

    @property
    def llm_agents_enabled(self) -> bool:
        """Whether CrewAI agents should call a real LLM (otherwise deterministic fallback)."""
        if self.USE_LLM_AGENTS is not None:
            return bool(self.USE_LLM_AGENTS and self.llm_api_key)
        return bool(self.llm_api_key) and not self.DEMO_MODE

    @property
    def langfuse_configured(self) -> bool:
        return bool(self.LANGFUSE_ENABLED and self.LANGFUSE_PUBLIC_KEY and self.LANGFUSE_SECRET_KEY)

    @property
    def adk_service_url(self) -> str:
        if self.ADK_SERVICE_URL_OVERRIDE:
            return self.ADK_SERVICE_URL_OVERRIDE
        return f"http://{self.ADK_SERVICE_HOST}:{self.ADK_SERVICE_PORT}"


settings = Settings()
