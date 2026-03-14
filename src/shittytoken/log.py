"""
Structured JSON logging factory using structlog.

Call configure_logging() once at process startup, then obtain loggers
with get_logger(name).

Log layout:
    logs/
      orchestrator/
        2026-03-14T04-21-50/       ← one dir per orchestrator run
          orchestrator.log         ← main orchestrator log
          worker-ssh2.vast.ai-28809.log  ← per-worker logs (future)
      gateway/
        2026-03-14T04-21-50/
          gateway.log
      stress-test/
        2026-03-14T04-40-00.log
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import structlog


def _make_run_dir(log_dir: Path, component: str) -> Path:
    """Create a timestamped run directory under logs/<component>/."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = log_dir / component / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # Maintain a "latest" symlink for convenience
    latest = log_dir / component / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)
    except OSError:
        pass  # symlinks may fail on some systems

    return run_dir


def configure_logging(
    log_dir: str | Path | None = None,
    component: str = "orchestrator",
) -> Path:
    """Configure structlog for dual output: console (stderr) + JSON file.

    Args:
        log_dir: Base directory for log files. Defaults to ``logs/`` in the
                 project root (next to config.yml).
        component: Component name (orchestrator, gateway, etc.). Creates
                   a timestamped subdirectory per run.

    Returns:
        Path to the run directory (for adding per-worker logs later).
    """
    if log_dir is None:
        from .config import _find_config_yml
        log_dir = _find_config_yml().parent / "logs"
    log_dir = Path(log_dir)

    run_dir = _make_run_dir(log_dir, component)
    log_file = run_dir / f"{component}.log"

    # stdlib root logger → file (JSON lines)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # stderr handler (human-readable)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Clear any handlers from previous configure_logging calls
    root.handlers.clear()
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
    return run_dir


def get_logger(name: str | None = None):
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
