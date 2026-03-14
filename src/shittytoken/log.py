"""
Structured JSON logging factory using structlog.

Call configure_logging() once at process startup, then obtain loggers
with get_logger(name).

Logs go to both stderr (human-readable) and logs/orchestrator.log (JSON).
"""

import logging
import sys
from pathlib import Path

import structlog


def configure_logging(log_dir: str | Path | None = None) -> None:
    """Configure structlog for dual output: console (stderr) + JSON file.

    Args:
        log_dir: Directory for log files. Defaults to ``logs/`` in the
                 project root (next to config.yml).
    """
    if log_dir is None:
        # Put logs next to config.yml
        from .config import _find_config_yml
        log_dir = _find_config_yml().parent / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "orchestrator.log"

    # stdlib root logger → file (JSON lines)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # stderr handler (human-readable)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Silence noisy third-party loggers
    for noisy in ("neo4j", "vastai", "urllib3", "asyncio", "aiohttp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Shared pre-processing chain
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
    )

    # File formatter: JSON
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    file_handler.setFormatter(file_formatter)

    # Console formatter: human-readable key=value
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
    )
    console_handler.setFormatter(console_formatter)

    structlog.get_logger().info("logging_configured", log_file=str(log_file))


def get_logger(name: str | None = None):
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
