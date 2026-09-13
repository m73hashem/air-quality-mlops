import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone


correlation_id: ContextVar[str] = ContextVar(
    "correlation_id",
    default="-",
)


class JsonFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Convert a log record into a JSON string."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id.get(),
        }

        return json.dumps(log_data)


def configure_logging(level: str = "INFO") -> None:
    """Configure application-wide JSON logging."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

def set_correlation_id(request_id: str) -> None:
    """Set the correlation ID for the current execution context."""
    correlation_id.set(request_id)
    