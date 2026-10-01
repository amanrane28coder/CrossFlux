"""Small standard-library logging setup shared by Python entry points."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from src.config import LOG_FORMAT_ENV, LOG_LEVEL_ENV

_STANDARD_FIELDS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
_OUTPUT_FIELDS = _STANDARD_FIELDS | {"timestamp", "level", "logger", "exception"}


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per line, retaining useful structured extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _OUTPUT_FIELDS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging() -> None:
    """Configure root logging once using CROSSFLUX_LOG_LEVEL/LOG_FORMAT."""
    level_name = os.environ.get(LOG_LEVEL_ENV, "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"{LOG_LEVEL_ENV} must be a standard logging level")

    format_name = os.environ.get(LOG_FORMAT_ENV, "text").lower()
    if format_name not in {"text", "json"}:
        raise ValueError(f"{LOG_FORMAT_ENV} must be 'text' or 'json'")

    handler = logging.StreamHandler()
    if format_name == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s", datefmt="%H:%M:%S"
        ))
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    else:
        for existing in root.handlers:
            existing.setFormatter(handler.formatter)
    root.setLevel(level)
