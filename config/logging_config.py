import logging
import structlog
import os
import sys
import structlog.contextvars
from typing import Optional


def setup_logging():
    """
    Configures logging for a cloud-native environment like Azure.
    All logs are structured (JSON) and written to stdout.
    """
    handlers = []
    print("Configuring logging to stdout for Azure production environment.")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    handlers.append(stream_handler)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=handlers,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def setup_experiment_logging(experiment_id: str) -> None:
    """
    Binds the experiment_id to the logging context.
    All subsequent logs within this context will have the 'experiment_id' field.
    """
    structlog.contextvars.bind_contextvars(experiment_id=experiment_id)

def remove_experiment_logging() -> None:
    """Removes the experiment_id from the logging context."""
    structlog.contextvars.unbind_contextvars("experiment_id")
