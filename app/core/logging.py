import logging
import sys
from app.core.config import settings

def setup_logging():
    log_format = "%(asctime)s [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] - %(message)s"
    level = logging.DEBUG if settings.DEBUG else logging.INFO

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stderr)  # stderr: stdout is the MCP stdio protocol channel
        ]
    )

    # Silence overly verbose third party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    for noisy in ("a2a", "LiteLLM", "litellm", "opentelemetry", "mcp", "google_genai", "grpc"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO if settings.DB_ECHO else logging.WARNING)

    return logging.getLogger("dealforge")

logger = setup_logging()
