"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

import structlog


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    service_name: str = "ad-ml",
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, emit JSON logs suitable for log aggregation.
        service_name: Service name added to every log record.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Standard library logging setup
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _add_service_name(service_name),
    ]

    if json_format:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)


def get_logger(name: str, **context: Any) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger with optional bound context.

    Args:
        name: Logger name (typically __name__).
        **context: Key-value pairs to bind to every log record.

    Returns:
        Bound structlog logger.
    """
    log = structlog.get_logger(name)
    if context:
        log = log.bind(**context)
    return log  # type: ignore[return-value]


def bind_request_context(
    request_id: Optional[str] = None,
    model_name: Optional[str] = None,
    **extra: Any,
) -> None:
    """Bind request-scoped context variables for structured logging.

    Call this at the start of each request handler. Context is stored in
    thread-local / contextvars storage and merged into every log record.

    Args:
        request_id: Unique request identifier.
        model_name: Model being served.
        **extra: Additional key-value context.
    """
    ctx: Dict[str, Any] = {}
    if request_id is not None:
        ctx["request_id"] = request_id
    if model_name is not None:
        ctx["model_name"] = model_name
    ctx.update(extra)
    structlog.contextvars.bind_contextvars(**ctx)


def clear_request_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


def _add_service_name(service_name: str) -> Any:
    """Return a structlog processor that injects service_name."""

    def processor(logger: Any, method: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        event_dict["service"] = service_name
        return event_dict

    return processor
