"""
app/logger.py
─────────────
Centralised logging configuration.

Call setup_logging() once at startup (main.py does this).
Every other module calls get_logger(__name__) to get its own child logger.

Log destinations:
  Console  — INFO and above.  Clean operational output.
  File     — DEBUG and above. Everything, including verbose fetch progress.

Rotation policy (10 MB × 2 backups = 30 MB ceiling total):
  data/app.log        ← current
  data/app.log.1      ← first rollover
  data/app.log.2      ← second rollover (oldest, deleted on next rollover)

Log format:
  2024-01-15 19:00:01  INFO      app.fetcher          [ESPN] Season 2022-23 done.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Guard so setup_logging() is idempotent — safe to call multiple times
_configured: bool = False

_LOG_FMT  = "%(asctime)s  %(levelname)-8s  %(name)-24s  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_MB = 1024 * 1024  # bytes per megabyte


def setup_logging(log_dir: str = "data") -> logging.Logger:
    """
    Configure the root 'bball' logger with a rotating file handler and a
    console handler. Returns the root logger. Subsequent calls are no-ops
    — the same logger is returned without re-adding handlers.
    """
    global _configured
    root = logging.getLogger("bball")

    if _configured:
        return root

    Path(log_dir).mkdir(exist_ok=True)
    log_file  = Path(log_dir) / "app.log"
    formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)

    # Rotating file handler — 10 MB per file, 2 backups = 30 MB max
    fh = RotatingFileHandler(
        log_file,
        maxBytes    = 10 * _MB,
        backupCount = 2,
        encoding    = "utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console handler — INFO only, no debug spam in the terminal
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)

    _configured = True
    root.info("Logging initialised → %s  (10 MB × 2 backups)", log_file)
    return root


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'bball' namespace.
    Call this at module level:

        from app.logger import get_logger
        log = get_logger(__name__)

    The name will appear in the log line as e.g. 'app.fetcher'.
    setup_logging() must have been called first (main.py handles this).
    """
    return logging.getLogger(f"bball.{name}")